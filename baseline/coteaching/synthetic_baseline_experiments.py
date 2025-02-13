import argparse
import numpy as np
import time
import sys
import yaml
import json
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.data as torchdata

sys.path.append("../../")

from experiment_code.data_utils.dataloader_loaders import (
    get_train_data,
    get_test_data,
)

from experiment_code.utils.utils import ArgFake

parser = argparse.ArgumentParser()
args = parser.parse_args()

CONFIG_FILE = "../../synthetic_config.yaml"
DATA_DIR = "../../data/"
DATASET_NAMES = [
    "cifar10",
    "cifar100",
    "fmnist",
]
CORRUPTION_TYPES = [
    "no_c",
    "c_cs",
    "c_rl",
    "c_lbf",
    "c_ns",
    "c_lbs",
    "c_no",
]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_RUNS = 5

RESULTS_FILE = "../../outputs/synthetic_results/baseline/co-teaching/results.json"

results = {
    ds_name: {c_type: {} for c_type in CORRUPTION_TYPES} for ds_name in DATASET_NAMES
}

print("Saving to file:", RESULTS_FILE)


def batch_label_flipping(targets, sources, corrupt_sources):
    mask = torch.tensor(
        [source.item() in corrupt_sources for source in sources], device=targets.device
    )
    new_targets = targets.clone()
    new_targets[mask] = torch.tensor(
        np.random.choice(targets.cpu().numpy(), size=(1,)), device=targets.device
    )
    return new_targets, sources


def batch_label_shuffle(targets, sources, corrupt_sources):
    mask = torch.tensor(
        [source.item() in corrupt_sources for source in sources], device=targets.device
    )
    new_targets = targets.clone()
    new_targets[mask] = torch.tensor(
        np.random.permutation(targets.cpu().numpy()), device=targets.device
    )[mask]
    return new_targets, sources


# if results file exists, load it
try:
    with open(RESULTS_FILE, "r") as fp:
        result_loaded = json.load(fp)
    for ds_name in result_loaded.keys():
        if ds_name not in results.keys():
            results[ds_name] = {}
        for c_type in result_loaded[ds_name].keys():
            if c_type not in results[ds_name].keys():
                results[ds_name][c_type] = {}
            for run in result_loaded[ds_name][c_type].keys():
                run_int = int(run)
                if run_int not in results[ds_name][c_type].keys():
                    results[ds_name][c_type][run_int] = result_loaded[ds_name][c_type][
                        run
                    ]
except FileNotFoundError:
    pass


print(results)


## dataset


class ToMemory(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device="cpu"):
        """
        Wrapper for a dataset that stores the data
        in memory. This is useful for speeding up
        training when the dataset is small enough
        to fit in memory. Note that
        attributes of the dataset are not stored
        in memory and so will not be directly accessible.

        This class stores the data in memory after the
        first access. This means that the first epoch
        will be slow but subsequent epochs will be fast.


        Examples
        ---------

        .. code-block::

            >>> dataset = ToMemory(dataset)
            >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
            >>> for epoch in range(10):
            >>>     for batch in dataloader:
            >>>         # do something with batch


        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset that you want to store in memory.

        """
        self.dataset = dataset
        self.device = device
        self.memory_dataset = {}

    def __getitem__(self, index):
        if index in self.memory_dataset:
            return self.memory_dataset[index]
        output = self.dataset[index]

        output_on_device = []
        for i in output:
            try:
                output_on_device.append(i.to(self.device))
            except:
                output_on_device.append(i)

        self.memory_dataset[index] = output_on_device
        return output

    def __len__(self):
        return len(self.dataset)


class CoTeaching(torchdata.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        outputs = self.dataset[index]
        return *outputs, index

    def __len__(self):
        return len(self.dataset)


# the following is from https://discuss.pytorch.org/t/top-k-error-calculation/48815/3
# with edits to the documentation and commenting.
def accuracy_topk(output: torch.tensor, target: torch.tensor, topk: tuple = (1,)):
    """
    https://discuss.pytorch.org/t/top-k-error-calculation/48815/3

    Computes the accuracy over the k top predictions for the specified values of k.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch


    Arguments
    ---------

    - output: torch.tensor:
        The prediction of the model.

    - target: torch.tensor:
        The targets that each prediction corresponds to.

    - topk: tuple (optional):
        This is a tuple of values that represent the k values
        for which the accuracy should be calculated with.


    Returns
    ---------

    - topk_accuracies: list:
        This returns a list of the top k accuracies for
        the k values specified.

    """
    with torch.no_grad():
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)
        # get top maxk indicies that correspond to the most likely probability scores
        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B]

        # expand the target tensor to the same shape as y_pred
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        # compare the target to each of the top k predictions made by the model
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] for each example we know which topk prediction matched truth

        # get topk accuracy
        list_topk_accs = []
        for k in topk:
            # find which of the top k predictions were correct
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # calculate the number of correct predictions
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return (
            list_topk_accs  # list of topk accuracies for batch [topk1, topk2, ... etc]
        )


## model class definitions
class Conv3Net(nn.Module):
    def __init__(
        self,
        input_dim=32,
        in_channels=3,
        channels=32,
        n_out=10,
    ):
        super(Conv3Net, self).__init__()

        self.input_dim = input_dim
        self.channels = channels
        self.n_out = n_out

        # =============== Cov Network ===============
        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels, self.channels, 3, padding="valid"),
                    ),
                    ("relu1", nn.ReLU()),
                    ("mp1", nn.MaxPool2d(2, 2)),
                    (
                        "conv2",
                        nn.Conv2d(self.channels, self.channels * 2, 3, padding="valid"),
                    ),
                    ("relu2", nn.ReLU()),
                    ("mp2", nn.MaxPool2d(2, 2)),
                    (
                        "conv3",
                        nn.Conv2d(
                            self.channels * 2, self.channels * 2, 3, padding="valid"
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        # =============== Linear ===============
        self.pm_fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            self.size_of_dim_out(self.input_dim) ** 2
                            * (self.channels * 2),
                            64,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                ]
            )
        )

        # =============== Classifier ===============
        self.pm_clf = nn.Linear(64, n_out)

        return

    def _resolution_calc(
        self,
        dim_in: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        if padding == "valid":
            padding = 0

        if type(dim_in) == list or type(dim_in) == tuple:
            out_h = dim_in[0]
            out_w = dim_in[1]
            out_h = (
                out_h + 2 * padding - dilation * (kernel_size - 1) - 1
            ) / stride + 1
            out_w = (
                out_w + 2 * padding - dilation * (kernel_size - 1) - 1
            ) / stride + 1

            return (out_h, out_w)

        return int(
            np.floor((dim_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        )

    def _get_conv_params(self, layer):
        kernel_size = (
            layer.kernel_size[0]
            if type(layer.kernel_size) == tuple
            else layer.kernel_size
        )
        stride = layer.stride[0] if type(layer.stride) == tuple else layer.stride
        padding = layer.padding[0] if type(layer.padding) == tuple else layer.padding
        return {"kernel_size": kernel_size, "stride": stride, "padding": padding}

    def size_of_dim_out(self, dim_in):
        out = self._resolution_calc(
            dim_in=dim_in, **self._get_conv_params(self.net.conv1)
        )
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.mp1))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.conv2))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.mp2))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.conv3))

        return out

    def forward(self, X):
        out = self.pm_clf(self.pm_fc(self.net(X)))
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def resnet20(num_classes):
    ## https://github.com/akamaster/pytorch_resnet_cifar10
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


class VGG(nn.Module):
    def __init__(
        self,
        n_out=10,
        criterion=nn.CrossEntropyLoss(reduction="none"),
    ):
        super(VGG, self).__init__()

        import torchvision.models as vision_models

        self.net = vision_models.vgg16_bn(
            weights=None,
        )
        self.clf = nn.Linear(1000, n_out)
        self.criterion = criterion

    def forward(self, X, y=None, return_loss=False):
        out = self.net(X)
        out = self.clf(out)
        if return_loss:
            assert y is not None
            loss = self.criterion(out, y)
            return loss, out
        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features=100,
        out_features=100,
        hidden_layer_features=(100,),
        dropout=0.2,
    ):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.hidden_layer_features = hidden_layer_features

        in_out_list = [in_features] + list(self.hidden_layer_features) + [out_features]

        in_list = in_out_list[:-1][:-1]
        out_list = in_out_list[:-1][1:]

        # =============== Linear ===============
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_value, out_value),
                    nn.Dropout(self.dropout),
                    nn.ReLU(),
                )
                for in_value, out_value in zip(in_list, out_list)
            ]
        )

        # =============== Classifier ===============
        self.clf = nn.Linear(in_out_list[-2], in_out_list[-1])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        out = self.clf(out)
        out = self.softmax(out)

        return out


def get_model(args):
    ## model initialization for different datasets

    if args.dataset_name == "cifar10":
        model = resnet20(
            num_classes=10,
        )
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

    if args.dataset_name == "cifar100":
        model = resnet20(
            num_classes=100,
        )
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

    if args.dataset_name == "fmnist":
        model = MLP(
            in_features=784,
            out_features=10,
            hidden_layer_features=[
                16,
                16,
            ],
            dropout=0.2,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")
    return model, optimizer, criterion


exp_seed = 42
exp_number = 0
for run in range(N_RUNS):
    for dataset_name in DATASET_NAMES:
        for corruption_type in CORRUPTION_TYPES:
            exp_number += 1
            exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))
            print(
                "Performing experiment",
                exp_number,
                "of",
                N_RUNS * len(DATASET_NAMES) * len(CORRUPTION_TYPES),
                "with seed",
                exp_seed,
                "...",
            )
            print("Dataset:", dataset_name)
            print("Corruption type:", corruption_type)

            if run in results[dataset_name][corruption_type]:
                print("skipping this one as completed already")
                continue

            if corruption_type == "c_lbf":
                BATCH_FLIPPING = True
                BATCH_SHUFFLING = False
                corruption_type = "no_c"
            elif corruption_type == "c_lbs":
                BATCH_FLIPPING = False
                BATCH_SHUFFLING = True
                corruption_type = "no_c"
            else:
                BATCH_FLIPPING = False
                BATCH_SHUFFLING = False

            args = ArgFake(
                {
                    "seed": exp_seed,
                    "dataset_name": dataset_name,
                    "model_name": "",
                    "data_dir": DATA_DIR,
                    "verbose": False,
                    "config_file": CONFIG_FILE,
                    "device": "auto",
                    "corruption_type": corruption_type,
                }
            )

            # dataset dependent args

            if args.dataset_name == "cifar10":
                args.n_epochs = 40
                args.epoch_decay_start = 20
                args.num_gradual = 10
                args.exponent = 1
                model_name = "Conv3Net"

            elif args.dataset_name == "cifar100":
                args.n_epochs = 40
                args.epoch_decay_start = 20
                args.num_gradual = 10
                args.exponent = 1
                model_name = "Conv3Net_100"

            elif args.dataset_name == "fmnist":
                args.n_epochs = 40
                args.epoch_decay_start = 16
                args.num_gradual = 10
                args.exponent = 1
                model_name = "MLP"

            ## load data config files for different datasets and corruption types

            training_params = yaml.load(
                open(args.config_file, "r"), Loader=yaml.FullLoader
            )[f"{model_name}-{args.corruption_type}-drstd"]["train_params"]

            if BATCH_FLIPPING or BATCH_SHUFFLING:
                if BATCH_FLIPPING:
                    corruption_type_to_load_n_sources = "c_lbf"
                    corruption_type = "c_lbf"
                elif BATCH_SHUFFLING:
                    corruption_type_to_load_n_sources = "c_lbs"
                    corruption_type = "c_lbs"

                n_corrupt_sources = yaml.load(
                    open(args.config_file, "r"), Loader=yaml.FullLoader
                )[f"{model_name}-{corruption_type_to_load_n_sources}-drstd"][
                    "train_params"
                ][
                    "n_corrupt_sources"
                ]
                n_sources = yaml.load(
                    open(args.config_file, "r"), Loader=yaml.FullLoader
                )[f"{model_name}-{corruption_type_to_load_n_sources}-drstd"][
                    "train_params"
                ][
                    "n_sources"
                ]
                CORRUPT_SOURCES = np.random.choice(
                    n_sources, n_corrupt_sources, replace=False
                ).tolist()

            training_params["source_fit"] = False
            training_params["return_sources"] = True

            train_loader, _ = get_train_data(args, {"train_params": training_params})

            test_loader, _ = get_test_data(
                args,
                {
                    "test_method": "traditional",
                    "batch_size": training_params["source_size"],
                },
            )

            class BatchFlipShuffleDL(object):
                def __init__(self, dl):
                    self.dl = dl

                def __iter__(self):
                    for batch in self.dl:
                        outputs = batch
                        inputs, targets, sources = outputs[:3]
                        others = outputs[3:]
                        if BATCH_FLIPPING:
                            targets, sources = batch_label_flipping(
                                targets, sources, CORRUPT_SOURCES
                            )
                        elif BATCH_SHUFFLING:
                            targets, sources = batch_label_shuffle(
                                targets, sources, CORRUPT_SOURCES
                            )
                        yield inputs, targets, *others

                def __len__(self):
                    return len(self.dl)

            ## model initialization for different datasets

            train_loader_cot = torchdata.DataLoader(
                CoTeaching(ToMemory(train_loader.dataset, device=DEVICE)),
                shuffle=True,
                batch_size=training_params["source_size"],
            )

            train_loader_cot = BatchFlipShuffleDL(train_loader_cot)

            test_loader_cot = torchdata.DataLoader(
                CoTeaching(ToMemory(test_loader.dataset, device=DEVICE)),
                shuffle=False,
                batch_size=training_params["source_size"],
            )

            forget_rate = 0.2
            # define drop rate schedule
            rate_schedule = np.ones(args.n_epochs) * forget_rate
            rate_schedule[: args.num_gradual] = np.linspace(
                0, forget_rate**args.exponent, args.num_gradual
            )

            # Loss functions
            def loss_coteaching(y_1, y_2, t, criterion1, criterion2, forget_rate):
                loss_1 = criterion1(y_1, t)
                ind_1_sorted = torch.argsort(loss_1)
                loss_1_sorted = loss_1[ind_1_sorted]

                loss_2 = criterion2(y_2, t)
                ind_2_sorted = torch.argsort(loss_2)

                remember_rate = 1 - forget_rate
                num_remember = int(remember_rate * len(loss_1_sorted))

                ind_1_update = ind_1_sorted[:num_remember]
                ind_2_update = ind_2_sorted[:num_remember]
                # exchange
                loss_1_update = criterion1(y_1[ind_2_update], t[ind_2_update])
                loss_2_update = criterion2(y_2[ind_1_update], t[ind_1_update])

                return (
                    torch.sum(loss_1_update) / num_remember,
                    torch.sum(loss_2_update) / num_remember,
                )

            # Train the Model
            def train_epoch(
                train_loader, epoch, model1, optimizer1, model2, optimizer2, pbar
            ):
                train_total = 0
                train_correct = 0
                train_total2 = 0
                train_correct2 = 0

                for i, (images, labels, indexes) in enumerate(train_loader):
                    ind = indexes.cpu().numpy().transpose()

                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Forward + Backward + Optimize
                    logits1 = model1(images)
                    prec1, _ = accuracy_topk(logits1, labels, topk=(1, 5))
                    train_total += 1
                    train_correct += prec1

                    logits2 = model2(images)
                    prec2, _ = accuracy_topk(logits2, labels, topk=(1, 5))
                    train_total2 += 1
                    train_correct2 += prec2
                    loss_1, loss_2 = loss_coteaching(
                        logits1,
                        logits2,
                        labels,
                        criterion1,
                        criterion2,
                        rate_schedule[epoch],
                    )

                    optimizer1.zero_grad()
                    loss_1.backward()
                    optimizer1.step()
                    optimizer2.zero_grad()
                    loss_2.backward()
                    optimizer2.step()

                    pbar.update(1)

                train_acc1 = float(train_correct) / float(train_total)
                train_acc2 = float(train_correct2) / float(train_total2)

                return train_acc1, train_acc2

            # Evaluate the Model
            def evaluate(test_loader, model1, model2):
                model1.eval()  # Change model to 'eval' mode.
                correct1 = 0
                total1 = 0
                for images, labels, _ in test_loader:
                    images = images = images.to(DEVICE)
                    logits1 = model1(images)
                    outputs1 = F.softmax(logits1, dim=1)
                    _, pred1 = torch.max(outputs1.data, 1)
                    total1 += labels.size(0)
                    correct1 += (pred1.cpu() == labels).sum()

                model2.eval()  # Change model to 'eval' mode
                correct2 = 0
                total2 = 0
                for images, labels, _ in test_loader:
                    images = images = images.to(DEVICE)
                    logits2 = model2(images)
                    outputs2 = F.softmax(logits2, dim=1)
                    _, pred2 = torch.max(outputs2.data, 1)
                    total2 += labels.size(0)
                    correct2 += (pred2.cpu() == labels).sum()

                acc1 = 100 * float(correct1) / float(total1)
                acc2 = 100 * float(correct2) / float(total2)
                return acc1, acc2

            model1, optimizer1, criterion1 = get_model(args)
            model2, optimizer2, criterion2 = get_model(args)

            model1.to(DEVICE)
            model2.to(DEVICE)

            # training
            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(total=args.n_epochs * len(train_loader_cot))
            results[dataset_name][corruption_type][run] = {}
            for epoch in range(args.n_epochs):
                # train models
                model1.train()
                model2.train()
                train_acc1, train_acc2 = train_epoch(
                    train_loader_cot,
                    epoch,
                    model1,
                    optimizer1,
                    model2,
                    optimizer2,
                    pbar,
                )
                # evaluate models
                test_acc1, test_acc2 = evaluate(test_loader_cot, model1, model2)

                # performance on the test
                tqdm.tqdm._instances.clear()
                model1.eval()
                with torch.no_grad():
                    sums, total = 0, 0
                    predictions = []
                    targets = []
                    for batch in tqdm.tqdm(test_loader, desc="Predicting Test Labels"):
                        (
                            X,
                            y,
                        ) = batch
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        outputs = torch.softmax(model1(X), 1)
                        predictions.append(outputs.cpu())
                        targets.append(y.cpu())
                    predictions = torch.cat(predictions)
                    targets = torch.cat(targets)

                print(
                    "accuracy on the test set after robust learning",
                    accuracy_topk(predictions, targets, topk=(1, 5)),
                )

                results[dataset_name][corruption_type][run][epoch] = {
                    k: v.item()
                    for k, v in zip(
                        ["test_top1acc", "test_top5acc"],
                        accuracy_topk(predictions, targets, topk=(1, 5)),
                    )
                }

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)

            pbar.close()
