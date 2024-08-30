import argparse
import numpy as np
import time
import sys
import yaml
import json
import tqdm
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.data as torchdata

sys.path.append("../../")

from experiment_code.data_utils.dataloader_loaders import (
    get_train_data,
    get_test_data,
)
from experiment_code.testing_utils.testing_functions import accuracy_topk
from experiment_code.utils.utils import ArgFake

parser = argparse.ArgumentParser()
parser.add_argument("--multiple_sources_in_batch", action="store_true")
args = parser.parse_args()

CONFIG_FILE = "../../synthetic_config.yaml"
DATASET_NAMES = ["cifar10", "cifar100", "fmnist"]
CORRUPTION_TYPES = ["no_c", "c_cs", "c_rl", "c_lbf", "c_ns", "c_lbs", "c_no"]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
N_RUNS = 10
MULTIPLE_SOURCES_IN_BATCH = args.multiple_sources_in_batch
if not MULTIPLE_SOURCES_IN_BATCH:
    RESULTS_FILE = "../../outputs/synthetic_results/baseline/idpa/results.json"
else:
    RESULTS_FILE = "../../outputs/synthetic_results_batch_multiple_sources/baseline/idpa/results.json"

DATA_DIR = "../../data/"

eta_lr, eta_init = 5e-2, 0.01

results = {
    ds_name: {c_type: {} for c_type in CORRUPTION_TYPES} for ds_name in DATASET_NAMES
}


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


class InstanceDependentDataset(torchdata.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.psi = [0] * len(dataset)
        return

    def update_psi(self, idx, value):
        self.psi[idx] = value

    def __getitem__(self, index):
        outputs = self.dataset[index]
        return *outputs, index, self.psi[index]

    def __len__(self):
        return len(self.dataset)


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
        model = Conv3Net(
            input_dim=32,
            in_channels=3,
            channels=32,
            n_out=10,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss()

    if args.dataset_name == "cifar100":
        model = Conv3Net(
            input_dim=32,
            in_channels=3,
            channels=32,
            n_out=100,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss()

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
        criterion = nn.CrossEntropyLoss()
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

            if corruption_type == "c_lbf" and MULTIPLE_SOURCES_IN_BATCH:
                BATCH_FLIPPING = True
                BATCH_SHUFFLING = False
                corruption_type = "no_c"
            elif corruption_type == "c_lbs" and MULTIPLE_SOURCES_IN_BATCH:
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
                args.lr = 0.001
                args.n_epochs = 25
                model_name = "Conv3Net"

            elif args.dataset_name == "cifar100":
                args.lr = 0.001
                args.n_epochs = 25
                model_name = "Conv3Net_100"

            elif args.dataset_name == "fmnist":
                args.lr = 0.001
                args.n_epochs = 40
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
                )[f"Conv3Net-{corruption_type_to_load_n_sources}-drstd"][
                    "train_params"
                ][
                    "n_corrupt_sources"
                ]
                n_sources = yaml.load(
                    open(args.config_file, "r"), Loader=yaml.FullLoader
                )[f"Conv3Net-{corruption_type_to_load_n_sources}-drstd"][
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

            train_loader_idd = torchdata.DataLoader(
                InstanceDependentDataset(ToMemory(train_loader.dataset, device=DEVICE)),
                shuffle=MULTIPLE_SOURCES_IN_BATCH,
                batch_size=training_params["source_size"],
            )

            train_loader_idd = BatchFlipShuffleDL(train_loader_idd)

            test_loader_idd = torchdata.DataLoader(
                InstanceDependentDataset(ToMemory(test_loader.dataset, device=DEVICE)),
                shuffle=False,
                batch_size=training_params["source_size"],
            )

            # IDD Helper functions

            eye = torch.eye(
                len(np.unique(train_loader.dataset.dataset.dataset.targets))
            ).to(DEVICE)

            # + Class-conditional probability
            def predict_truelabel(etas, targets, pnl):
                # re-estimated posterior of true label given eta and posterior of noisy labels
                part1 = eye[targets.long()] * (1 - etas).view(-1, 1)
                part2 = (pnl * etas).view(-1, 1)
                return (part1 + part2).clamp(min=1e-5).log()

            def logs(msg):
                sys.stdout.write("\r" + msg)

            # class re-weighting
            def upsample(targets):
                return torch.ones_like(targets)

            model, optimizer, criterion = get_model(args)
            model.to(DEVICE)

            # training base model to get the psi values

            tqdm.tqdm._instances.clear()
            model.train()
            pbar = tqdm.tqdm(
                desc="Training", total=args.n_epochs * len(train_loader_idd)
            )
            for epoch in range(args.n_epochs):
                pbar.postfix = f"Epoch: {epoch+1}/{args.n_epochs}"
                for batch in train_loader_idd:
                    optimizer.zero_grad()

                    X, y, _, _ = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)

                    output = model(X)
                    loss = criterion(output, y)

                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
            pbar.close()

            # predicting labels that are used as posterior

            tqdm.tqdm._instances.clear()
            model.eval()
            with torch.no_grad():
                sums, total = 0, 0  #
                predictions = []
                targets = []
                for batch in tqdm.tqdm(train_loader_idd, desc="Predicting Labels"):
                    X, y, indies, _ = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    outputs = torch.softmax(model(X), 1)
                    psi = (outputs * eye[y]).sum(1).cpu().tolist()
                    for idx, p in zip(indies, psi):  #
                        train_loader_idd.dataset.update_psi(idx, p)  #
                    sums += sum(psi)  #
                    total += len(psi)  #
                    predictions.append(outputs.cpu())
                    targets.append(y.cpu())
                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

            print(
                "accuracy on the training set before robust learning",
                accuracy_topk(predictions, targets, topk=(1, 5)),
            )

            # performance on the test
            tqdm.tqdm._instances.clear()
            # predicting labels that are used as posterior
            model.eval()
            with torch.no_grad():
                sums, total = 0, 0  #
                predictions = []
                targets = []
                for batch in tqdm.tqdm(test_loader, desc="Predicting Test Labels"):
                    (
                        X,
                        y,
                    ) = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    outputs = torch.softmax(model(X), 1)
                    predictions.append(outputs.cpu())
                    targets.append(y.cpu())
                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

            print(
                "accuracy on the test set before robust learning",
                accuracy_topk(predictions, targets, topk=(1, 5)),
            )

            ETA = torch.zeros((len(train_loader_idd.dataset),)).to(DEVICE) + eta_init

            # double training loop to get robust model

            tqdm.tqdm._instances.clear()
            # TRAIN
            pbar = tqdm.tqdm(
                desc="Training Robust Model",
                total=args.n_epochs * len(train_loader_idd),
            )
            results[dataset_name][corruption_type][run] = {}
            for epoch in range(args.n_epochs):
                model.train()
                time_start = time.time()
                correct, total = 0, 0
                eta_hist = torch.Tensor([0] * 10).to(
                    DEVICE
                )  ##### was commented out for some reason #####
                pbar.postfix = f"Epoch: {epoch+1}/{args.n_epochs}"

                for inputs, targets, indies, pnl in train_loader_idd:
                    inputs, targets, indies, pnl = (
                        inputs.to(DEVICE),
                        targets.to(DEVICE),
                        indies.to(DEVICE),
                        pnl.to(DEVICE).float(),
                    )

                    optimizer.zero_grad()
                    outputs = torch.log_softmax(model(inputs), 1)

                    # ALTERNATING OPTIMIZATION
                    # ---------------------------------------------------- #
                    # + Prediction                                         #
                    pyz_x = (
                        predict_truelabel(ETA[indies], targets, pnl) + outputs.detach()
                    ).exp()
                    pz_x = pyz_x / pyz_x.sum(1).view(-1, 1)  #
                    # + Optimization                                       #
                    # |- classifier                                        #
                    loss = -(upsample(targets) * (pz_x * outputs).sum(1)).mean()
                    loss.backward()  #
                    optimizer.step()  #
                    # |- confusing                                         #
                    # For the simplicty of the updating rule, we actually assume pnl is close
                    # to 1. Directly assuming pnl=1 can lead to similar results in practice.
                    if epoch != 0:  #
                        disparities = (
                            pz_x
                            * (
                                1
                                + (pnl * ETA[indies] - ETA[indies] - 1).view(-1, 1)
                                * eye[targets]
                            )
                        ).sum(1)
                        ETA[indies] += (
                            eta_lr * disparities / ETA[indies].clamp(min=1e-5)
                        )
                        ETA[indies] = ETA[indies].clamp(min=0, max=1)  #
                    # ---------------------------------------------------- #

                    # ---------------------------------------------------- #
                    # + Classifier                                         #
                    predicts = outputs.detach().argmax(1)  #
                    correct += (predicts == targets).float().sum().item()  #
                    total += inputs.size(0)  #
                    # + Etas                                               #
                    etas = ETA[indies]  #
                    eta_hist += (
                        torch.eye(10)
                        .to(DEVICE)[(etas * 10).long().clamp(min=0, max=9)]
                        .sum(0)
                    )
                    # ---------------------------------------------------- #
                    pbar.update(1)

                # performance on the test
                tqdm.tqdm._instances.clear()
                # predicting labels that are used as posterior
                model.eval()
                with torch.no_grad():
                    sums, total = 0, 0  #
                    predictions = []
                    targets = []
                    for batch in tqdm.tqdm(test_loader, desc="Predicting Test Labels"):
                        (
                            X,
                            y,
                        ) = batch
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        outputs = torch.softmax(model(X), 1)
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

            pbar.close()

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)
