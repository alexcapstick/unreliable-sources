import os
from pathlib import Path
import argparse
import numpy as np
import yaml
import tqdm
import math
import typing
import json
import time
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from catalyst.metrics import AccuracyMetric
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from experiment_code.utils.utils import ArgFake
from experiment_code.data_utils.dataloader_loaders import get_train_data, get_test_data

from loss_adapted_plasticity import SourceLossWeighting

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="./outputs/synthetic_results_batch_multiple_sources/",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = [
    "cifar10",
]
CORRUPTION_TYPES = ["no_c", "c_cs", "c_rl", "c_lbf", "c_ns", "c_lbs", "c_no"]
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "./data/"
TEST_DIR = args.test_dir
DEPRESSION = [True, False]
RESULTS_FILE = os.path.join(TEST_DIR, "results.json")
CONFIG_FILE = "./synthetic_config.yaml"

exp_seed = args.seed


# --- model options
LAP_HISTORY_LENGTH = 25
DEPRESSION_STRENGTH = 1.0
LENIENCY = 0.8
HOLD_OFF = 0
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 0


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


results = {
    ds_name: {c_type: {run: {} for run in RUNS} for c_type in CORRUPTION_TYPES}
    for ds_name in DATASET_NAMES
}

# if results file exists, load it
try:
    with open(RESULTS_FILE, "r") as fp:
        result_loaded = json.load(fp)
    print("loaded previous results")
    for ds_name in result_loaded.keys():
        if ds_name not in results.keys():
            results[ds_name] = {}
        for c_type in result_loaded[ds_name].keys():
            if c_type not in results[ds_name].keys():
                results[ds_name][c_type] = {}
            for run in result_loaded[ds_name][c_type].keys():
                if run not in [int(k) for k in results[ds_name][c_type].keys()]:
                    results[ds_name][c_type][int(run)] = {}
                for depression in result_loaded[ds_name][c_type][run].keys():
                    depression_bool = depression == "true"
                    if depression_bool not in results[ds_name][c_type][int(run)].keys():
                        results[ds_name][c_type][int(run)][depression_bool] = (
                            result_loaded[ds_name][c_type][run][depression]
                        )
except FileNotFoundError:
    pass

## dataset


class ToMemory(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
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
        self.memory_dataset = {}

    def __getitem__(self, index):
        if index in self.memory_dataset:
            return self.memory_dataset[index]
        output = self.dataset[index]
        self.memory_dataset[index] = output
        return output

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
        criterion=nn.CrossEntropyLoss(reduction="none"),
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

        self.criterion = criterion

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

    def forward(self, X, y=None, return_loss=False):
        out = self.pm_clf(self.pm_fc(self.net(X)))
        if return_loss:
            assert y is not None
            loss = self.criterion(out, y)
            return loss, out
        return (out,)


class MLP(nn.Module):
    def __init__(
        self,
        in_features=100,
        out_features=100,
        hidden_layer_features=(100,),
        dropout=0.2,
        criterion=nn.CrossEntropyLoss(reduction="none"),
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
        self.criterion = criterion

    def forward(self, X, y=None, return_loss=False):
        out = X
        for layer in self.layers:
            out = layer(out)
        out = self.clf(out)
        out = self.softmax(out)
        if return_loss:
            assert y is not None
            loss = self.criterion(out, y)
            return loss, out

        return out


def get_model(args):
    ## model initialization for different datasets
    if args.dataset_name == "cifar10":
        model = Conv3Net(
            input_dim=32,
            in_channels=3,
            channels=32,
            n_out=10,
            criterion=nn.CrossEntropyLoss(reduction="none"),
        )
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

    if args.dataset_name == "cifar100":
        model = Conv3Net(
            input_dim=32,
            in_channels=3,
            channels=32,
            n_out=100,
            criterion=nn.CrossEntropyLoss(reduction="none"),
        )
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

    if args.dataset_name == "fmnist":
        model = MLP(
            in_features=784,
            out_features=10,
            hidden_layer_features=[
                16,
                16,
            ],
            dropout=0.2,
            criterion=nn.CrossEntropyLoss(reduction="none"),
        )
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

    return model, optimiser


def train_batch(
    model,
    optimiser,
    x,
    y,
    sources,
    label_loss_weighting,
    device,
    warmup=True,
    writer=None,
):
    model.to(device)
    model.train()

    optimiser.zero_grad()

    loss, outputs = model(x, y=y, return_loss=True)

    if not warmup:
        if label_loss_weighting is not None:
            loss = label_loss_weighting(
                losses=loss, sources=sources, writer=writer, writer_prefix="label"
            )

    loss = torch.mean(loss)
    loss.backward()
    optimiser.step()

    return loss.item(), outputs


def train_epoch(
    model,
    train_loader,
    optimiser,
    device,
    epoch_number,
    label_loss_weighting,
    pbar,
    writer=None,
):
    model.to(device)

    model.train()

    train_loss = 0
    train_total = 0
    train_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=10)
    lr = optimiser.param_groups[0]["lr"]

    for batch_idx, (inputs, targets, sources) in enumerate(train_loader):
        inputs, targets, sources = (
            inputs.to(device),
            targets.to(device),
            sources.to(device),
        )
        sources = sources.squeeze(-1)

        if BATCH_FLIPPING:
            targets, sources = batch_label_flipping(targets, sources, CORRUPT_SOURCES)
        if BATCH_SHUFFLING:
            targets, sources = batch_label_shuffle(targets, sources, CORRUPT_SOURCES)

        loss, outputs = train_batch(
            model,
            optimiser,
            inputs,
            targets,
            sources,
            label_loss_weighting=label_loss_weighting,
            device=device,
            warmup=False,
            writer=writer,
        )

        if writer is not None:
            writer.add_scalar(
                "Train Loss",
                loss,
                epoch_number * len(train_loader) + batch_idx,
            )

        train_acc_meter.update(outputs, targets)
        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

        pbar.update(1)

    metrics = train_acc_meter.compute_key_value()

    return train_loss / train_total, metrics["accuracy01"], metrics["accuracy05"]


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    test_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=10)
    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = model(inputs, y=targets, return_loss=True)

            loss = torch.mean(loss)
            test_acc_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        metrics = test_acc_meter.compute_key_value()
    return test_loss / test_total, metrics["accuracy01"], metrics["accuracy05"]


for ds_name in DATASET_NAMES:
    for c_type in CORRUPTION_TYPES:
        for run in RUNS:
            print(
                f"for dataset {ds_name}, no. corrupt sources {c_type}",
                f"and run {run} the following depression has been completed",
                results[ds_name][c_type][run].keys(),
            )


exp_number = 0
for run in RUNS:
    for dataset_name in DATASET_NAMES:
        for corruption_type in CORRUPTION_TYPES:
            # seed same for all depression types
            exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))
            for depression in DEPRESSION:
                exp_number += 1

                if depression in results[dataset_name][corruption_type][run].keys():
                    print(
                        "Skipping the following experiment as already completed:",
                        dataset_name,
                        corruption_type,
                        run,
                        depression,
                        "...",
                    )
                    continue

                print(
                    "Performing experiment",
                    exp_number,
                    "of",
                    len(RUNS)
                    * len(DATASET_NAMES)
                    * len(CORRUPTION_TYPES)
                    * len(DEPRESSION),
                    "with seed",
                    exp_seed,
                    "...",
                )
                print("Dataset:", dataset_name)
                print("Corruption type:", corruption_type)
                print("Run:", run)
                print("Depression:", depression)

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
                        "device": DEVICE,
                        "corruption_type": corruption_type,
                    }
                )

                # dataset dependent args

                # not hparam optimised for these experiments!
                # - performance could be improved further
                if args.dataset_name == "cifar10":
                    args.n_epochs = 25
                    model_name = "Conv3Net"

                elif args.dataset_name == "cifar100":
                    args.n_epochs = 25
                    WARMUP_ITERS = 100
                    model_name = "Conv3Net_100"

                elif args.dataset_name == "fmnist":
                    args.n_epochs = 40
                    LAP_HISTORY_LENGTH = 50
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

                training_params["return_sources"] = True

                train_loader, _ = get_train_data(
                    args, {"train_params": training_params}
                )
                test_loader, _ = get_test_data(
                    args,
                    {
                        "test_method": "traditional",
                        "batch_size": training_params["source_size"],
                    },
                )

                train_loader = torchdata.DataLoader(
                    dataset=ToMemory(train_loader.dataset),
                    batch_size=training_params["source_size"],
                    shuffle=True,
                )

                test_loader = torchdata.DataLoader(
                    dataset=ToMemory(test_loader.dataset),
                    batch_size=training_params["source_size"],
                    shuffle=False,
                )

                model, optimiser = get_model(args)

                if depression:
                    label_loss_weighting = SourceLossWeighting(
                        history_length=LAP_HISTORY_LENGTH,
                        warmup_iters=WARMUP_ITERS,
                        depression_strength=DEPRESSION_STRENGTH,
                        discrete_amount=DISCRETE_AMOUNT,
                        leniency=LENIENCY,
                    )

                else:
                    label_loss_weighting = None

                writer = SummaryWriter(
                    log_dir=os.path.join(
                        TEST_DIR,
                        "tb",
                        args.corruption_type,
                        dataset_name,
                        f"{run}",
                        f"{depression}",
                        f"{time.time().__str__().replace('.', '')}",
                    )
                )
                if BATCH_FLIPPING or BATCH_SHUFFLING:
                    corrupt_sources = CORRUPT_SOURCES
                else:
                    corrupt_sources = (
                        train_loader.dataset.dataset.corrupt_sources.tolist()
                    )

                results_this_train = {}

                pbar = tqdm.tqdm(
                    total=args.n_epochs * len(train_loader), desc="Training"
                )

                for epoch in range(args.n_epochs):
                    train_loss, train_top1acc, train_top5acc = train_epoch(
                        model=model,
                        train_loader=train_loader,
                        optimiser=optimiser,
                        device=DEVICE,
                        epoch_number=epoch,
                        label_loss_weighting=label_loss_weighting,
                        writer=writer,
                        pbar=pbar,
                    )
                    test_loss, test_top1acc, test_top5acc = test(
                        model, test_loader, DEVICE
                    )
                    if writer is not None:
                        writer.add_scalar("Test Loss", test_loss, epoch)
                        writer.add_scalar("Test Acc", test_top1acc, epoch)
                        writer.add_scalar("Test Top5Acc", test_top5acc, epoch)

                    results_this_train[epoch] = {
                        "train_loss": train_loss,
                        "train_top1acc": train_top1acc,
                        "train_top5acc": train_top5acc,
                        "test_loss": test_loss,
                        "test_top1acc": test_top1acc,
                        "test_top5acc": test_top5acc,
                    }
                pbar.close()

                results_this_train["corrupt_sources"] = corrupt_sources

                results[dataset_name][corruption_type][run][
                    depression
                ] = results_this_train

                print(results_this_train[epoch])

                writer.flush()
                writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
