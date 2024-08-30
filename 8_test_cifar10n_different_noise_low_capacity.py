import numpy as np
from pathlib import Path
import tqdm
import time
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision
import os
import requests
import typing as t
import torchvision.transforms.v2 as transforms
from catalyst.metrics import AccuracyMetric
from torch.utils.tensorboard import SummaryWriter
import json
import argparse

from loss_adapted_plasticity import SourceLossWeighting


parser = argparse.ArgumentParser()
parser.add_argument("--max_corruption_level", type=float, default=1.0)
parser.add_argument("--min_corruption_level", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="./outputs/cifar_10n_different_noise_results_low_capacity/",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = ["cifar-10n"]
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "./data/"
TEST_DIR = args.test_dir

# --- model options
DEPRESSION = [True, False]
LAP_HISTORY_LENGTH = 25
DEPRESSION_STRENGTH = 1
LENIENCY = 0.8
HOLD_OFF = 0
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 100

# --- data options
N_SOURCES = 10
N_CORRUPT_SOURCES = [6, 5, 4, 3, 2]
MIN_CORRUPTION_LEVEL = args.min_corruption_level
MAX_CORRUPTION_LEVEL = args.max_corruption_level
BATCH_SIZE = 128
N_EPOCHS = 25
LR = 0.001
CIFAR10_TRANSFORM = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)


exp_seed = args.seed


RESULTS_FILE = os.path.join(
    TEST_DIR,
    f"results_{MAX_CORRUPTION_LEVEL}_{MIN_CORRUPTION_LEVEL}_{''.join([str(r) for r in RUNS])}.json",
)

print(RESULTS_FILE)
results = {
    ds_name: {nc: {run: {} for run in RUNS} for nc in N_CORRUPT_SOURCES}
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
        for nc in result_loaded[ds_name].keys():
            if nc not in [int(k) for k in results[ds_name].keys()]:
                results[ds_name][int(nc)] = {}
            for run in result_loaded[ds_name][nc].keys():
                if run not in [int(k) for k in results[ds_name][int(nc)].keys()]:
                    results[ds_name][int(nc)][int(run)] = {}
                for depression in result_loaded[ds_name][nc][run].keys():
                    depression_bool = depression == "true"
                    if (
                        depression_bool
                        not in results[ds_name][int(nc)][int(run)].keys()
                    ):
                        results[ds_name][int(nc)][int(run)][depression_bool] = (
                            result_loaded[ds_name][nc][run][depression]
                        )
except FileNotFoundError:
    pass

print(results)
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
        output_device = []
        for item in output:
            try:
                output_device.append(item.to(DEVICE))
            except:
                output_device.append(item)
        self.memory_dataset[index] = output_device
        return output

    def __len__(self):
        return len(self.dataset)


# cifar-10N dataset
class CIFAR10N(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "./",
        train: bool = True,
        n_sources: int = 10,
        n_corrupt_sources: int = 8,
        noise_level: t.Union[None, float, t.List[float]] = None,
        seed: int = None,
        download: bool = False,
        **cifar10_kwargs,
    ):
        """
        This dataset wraps the CIFAR-10 observations and clean
        labels with erroneous labels generated by human labelling.
        These labels are found here: https://github.com/UCSC-REAL/cifar-10-100n.

        Arguments
        ---------

        - root: str
            The root directory to store the data.
        - train: bool
            Whether to use the training or test set.
        - n_sources: int
            The number of sources to use.
        - n_corrupt_sources: int
            The number of sources to corrupt.
        - noise_level: Union[None, float, List[float]]
            The noise level to use for each source. If None, then
            no noise is added.
        - seed: int
            The seed to use for reproducibility.
        - download: bool
            Whether to download the data.
        - **cifar10_kwargs:
            Keyword arguments to pass to the CIFAR-10 dataset.

        """

        # creating seed for random operations
        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        else:
            self.seed = seed

        # getting CIFAR100 data
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            **cifar10_kwargs,
        )

        self.train = train

        if self.train:
            self.root = root
            self.n_sources = n_sources
            self.n_corrupt_sources = n_corrupt_sources
            self.corrupt_sources = np.random.default_rng(self.seed).choice(
                n_sources, size=(self.n_corrupt_sources,), replace=False
            )
            self.source_size = len(self.cifar_dataset) // self.n_sources

            self.seed = np.random.default_rng(self.seed).integers(
                low=1, high=1e9, size=1
            )[0]

            # setting the noise level
            if noise_level is None:
                self._noise_level = [0] * len(self.corrupt_sources)
            elif type(noise_level) == float:
                self._noise_level = [noise_level] * len(self.corrupt_sources)
            elif hasattr(noise_level, "__iter__"):
                if hasattr(noise_level, "__len__"):
                    if hasattr(self.corrupt_sources, "__len__"):
                        assert len(noise_level) == len(self.corrupt_sources), (
                            "Please ensure that the noise level "
                            "is the same length as the corrupt sources. "
                            f"Expected {len(self.corrupt_sources)} noise levels, "
                            f"got {len(noise_level)}."
                        )
                self._noise_level = noise_level
            else:
                raise TypeError(
                    "Please ensure that the noise level is a float, iterable or None"
                )

            self.noise_level = np.zeros(self.n_sources)
            self.noise_level[self.corrupt_sources] = self._noise_level

            if download:
                self._download_noisy_labels()

            self.clean_labels = np.array(self.cifar_dataset.targets)
            self.noisy_labels = self._get_noisy_labels()

            wrong_labels = self.clean_labels != self.noisy_labels
            idx_wrong_labels = np.argwhere(wrong_labels)[:, 0]

            correct_labels = self.clean_labels == self.noisy_labels
            idx_correct_labels = np.argwhere(correct_labels)[:, 0]

            np.random.default_rng(self.seed).shuffle(idx_wrong_labels)
            self.seed = np.random.default_rng(self.seed).integers(
                low=1, high=1e9, size=1
            )[0]

            np.random.default_rng(self.seed).shuffle(idx_correct_labels)
            self.seed = np.random.default_rng(self.seed).integers(
                low=1, high=1e9, size=1
            )[0]

            sources = []
            targets = []
            wl_idxs = []
            cl_idxs = []

            wl_i = 0
            for source in np.arange(self.n_sources):
                nl = self.noise_level[source]
                source_n_wl = int(self.source_size * nl)

                if wl_i + source_n_wl >= len(idx_wrong_labels):
                    raise TypeError(
                        "There are not enough noisy labels available for the noise_level and "
                        "n_corrupt_sources that you have supplied."
                    )

                source_wl_idxs = idx_wrong_labels[wl_i : wl_i + source_n_wl]

                sources_temp = np.zeros(len(source_wl_idxs))
                sources_temp[:] = source

                sources.append(sources_temp)
                wl_idxs.append(source_wl_idxs)
                targets.append(self.noisy_labels[source_wl_idxs])

                wl_i = wl_i + source_n_wl

            wl_idxs = np.concatenate(wl_idxs)

            # now create an index array with all the left over indices and
            # only get them from the clean labels

            idxs_left = np.arange(len(self.cifar_dataset))
            idxs_left = idxs_left[~np.isin(idxs_left, wl_idxs)]

            cl_i = 0
            for source in np.arange(self.n_sources):
                nl = self.noise_level[source]
                source_n_wl = int(self.source_size * nl)
                source_n_cl = int(self.source_size - source_n_wl)

                source_cl_idxs = idxs_left[cl_i : cl_i + source_n_cl]

                sources_temp = np.zeros(len(source_cl_idxs))
                sources_temp[:] = source

                sources.append(sources_temp)
                cl_idxs.append(source_cl_idxs)
                targets.append(self.clean_labels[source_cl_idxs])

                cl_i = cl_i + source_n_cl

            cl_idxs = np.concatenate(cl_idxs)

            idx = np.concatenate([wl_idxs, cl_idxs])
            targets = np.concatenate(
                [self.noisy_labels[wl_idxs], self.clean_labels[cl_idxs]]
            )
            sources = np.concatenate(sources)

            data_order = np.argsort(idx)
            self.sources = sources[data_order].astype(np.int64)
            self.targets = targets[data_order].astype(np.int64)

        else:
            self.targets = np.array(self.cifar_dataset.targets)

        return

    def _download_noisy_labels(self):
        url = (
            "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
        )
        file_dir = os.path.join(self.root, "cifar-n")
        file_name = os.path.join(file_dir, "CIFAR-10_human.pt")
        if not os.path.exists(file_name):
            print("Noisy labels file will be downloaded")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            r = requests.get(url)
            open(file_name, "wb").write(r.content)
        else:
            print("Noisy labels file exists")
        return

    def _get_noisy_labels(self):
        file_dir = os.path.join(self.root, "cifar-n")
        file_name = os.path.join(file_dir, "CIFAR-10_human.pt")

        labels = torch.load(file_name)

        return labels["worse_label"]

    def __getitem__(self, index):
        # im is a tensor
        im = self.cifar_dataset[index][0].float()
        # targets is an integer
        target = self.targets[index].astype(np.int64)
        if self.train:
            source = self.sources[index]
            return im, target, source
        return im, target

    def __len__(self):
        return len(self.cifar_dataset)


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


def Conv3Net32(input_dim=32, in_channels=3, channels=32, n_out=10):
    return Conv3Net(
        input_dim=input_dim,
        in_channels=in_channels,
        channels=channels,
        n_out=n_out,
        criterion=nn.CrossEntropyLoss(reduction="none"),
    )


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
    scheduler,
    device,
    epoch_number,
    label_loss_weighting,
    writer=None,
):
    model.to(device)

    model.train()
    if scheduler is not None:
        scheduler.step()

    train_loss = 0
    train_total = 0
    train_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=10)
    lr = optimiser.param_groups[0]["lr"]
    pbar = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch_number+1}, LR {lr:.4f}")

    for batch_idx, (inputs, targets, sources) in enumerate(pbar):
        inputs, targets, sources = (
            inputs.to(device),
            targets.to(device),
            sources.to(device),
        )
        sources = sources.squeeze(-1)

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
        metrics = train_acc_meter.compute_key_value()

        pbar.set_postfix(
            {
                "Train Loss": train_loss / train_total,
                "Train Acc": metrics["accuracy01"],
                "Train Top5Acc": metrics["accuracy05"],
            }
        )

    pbar.close()

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
            label_loss, outputs = model(inputs, y=targets, return_loss=True)

            loss = torch.mean(label_loss)
            test_acc_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        metrics = test_acc_meter.compute_key_value()
        print(
            "Test Loss",
            test_loss / test_total,
            "Test Acc",
            metrics["accuracy01"],
            "Test Top5Acc",
            metrics["accuracy05"],
        )
    return test_loss / test_total, metrics["accuracy01"], metrics["accuracy05"]


for dataset_name in DATASET_NAMES:
    for nc in N_CORRUPT_SOURCES:
        for run in RUNS:
            print(
                f"for dataset {dataset_name}, no. corrupt sources {nc}",
                f"and run {run} the following depression has been completed",
                results[dataset_name][nc][run].keys(),
            )


exp_number = 0
for run in RUNS:
    for dataset_name in DATASET_NAMES:
        for nc in N_CORRUPT_SOURCES:
            # seed same for all depression types
            exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))
            for depression in DEPRESSION:
                exp_number += 1

                if depression in results[dataset_name][nc][run].keys():
                    print(
                        "Skipping the following experiment as already completed:",
                        dataset_name,
                        nc,
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
                    * len(N_CORRUPT_SOURCES)
                    * len(DEPRESSION),
                    "with seed",
                    exp_seed,
                    "...",
                )
                print("Dataset:", dataset_name)
                print("No. Corrupt Sources:", nc)
                print("Run:", run)
                print("Depression:", depression)

                corruption_level = np.linspace(
                    MIN_CORRUPTION_LEVEL, MAX_CORRUPTION_LEVEL, nc, endpoint=True
                )
                print("Corruption Level:", corruption_level)

                train_dataset = CIFAR10N(
                    root=DATA_DIR,
                    train=True,
                    n_sources=N_SOURCES,
                    n_corrupt_sources=nc,
                    noise_level=corruption_level,
                    seed=exp_seed,
                    download=True,
                    transform=CIFAR10_TRANSFORM,
                )
                test_dataset = CIFAR10N(
                    root=DATA_DIR,
                    train=False,
                    download=True,
                    transform=CIFAR10_TRANSFORM,
                )

                # dataset dependent args

                train_dataset_memory = ToMemory(train_dataset)
                test_dataset_memory = ToMemory(test_dataset)

                aug_train_loader = torchdata.DataLoader(
                    dataset=train_dataset_memory,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )

                test_loader = torchdata.DataLoader(
                    dataset=test_dataset_memory,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                model = Conv3Net32(
                    input_dim=32,
                    in_channels=3,
                    channels=32,
                    n_out=10,
                )

                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=LR,
                )

                scheduler = None

                if depression:
                    label_loss_weighting = SourceLossWeighting(
                        history_length=LAP_HISTORY_LENGTH,
                        warmup_iters=WARMUP_ITERS,
                        depression_strength=DEPRESSION_STRENGTH,
                        discrete_amount=DISCRETE_AMOUNT,
                        leniency=LENIENCY,
                        device=DEVICE,
                    )

                else:
                    label_loss_weighting = None

                # writer = SummaryWriter(
                #     log_dir=os.path.join(
                #         TEST_DIR,
                #         "tb",
                #         dataset_name,
                #         f"{run}",
                #         f"{depression}",
                #         f"{time.time().__str__().replace('.', '')}",
                #     )
                # )
                writer = None

                corrupt_sources = train_dataset.corrupt_sources.tolist()

                results_this_train = {}

                for epoch in range(N_EPOCHS):
                    train_loss, train_top1acc, train_top5acc = train_epoch(
                        model=model,
                        train_loader=aug_train_loader,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        device=DEVICE,
                        epoch_number=epoch,
                        label_loss_weighting=label_loss_weighting,
                        writer=writer,
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

                results_this_train["corrupt_sources"] = corrupt_sources

                results[dataset_name][nc][run][depression] = results_this_train

                if writer is not None:
                    writer.flush()
                    writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
