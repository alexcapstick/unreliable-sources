import os
from pathlib import Path
import argparse
import json
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as torchdatasets
import torch.utils.data as torchdata
import torchvision.transforms.v2 as transforms
import torchvision.models as torchmodels
from catalyst.metrics import AccuracyMetric
from torch.utils.tensorboard import SummaryWriter

from loss_adapted_plasticity import SourceLossWeighting


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--scheduler", action="store_true")
parser.add_argument("--results_name", type=str, default="results.json")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="./outputs/tiny_imagenet_random_label/",
)
parser.add_argument("--depression", action="store_true")
parser.add_argument("--no_depression", action="store_true")
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = [
    "tiny-imagenet-200",
]
CORRUPTION_TYPE = [
    "random_label",
    "original",
]
N_SOURCES = 100
N_CORRUPT_SOURCES = 40
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "./data/"
TEST_DIR = args.test_dir
if (not args.depression) and (not args.no_depression):
    DEPRESSION = [True, False]
else:
    if args.depression:
        DEPRESSION = [True]
        args.results_name = args.results_name.replace(".json", "_dep.json")
    if args.no_depression:
        DEPRESSION = [False]
        args.results_name = args.results_name.replace(".json", "_no_dep.json")
if args.scheduler:
    args.results_name = args.results_name.replace(".json", "_scheduler.json")
RESULTS_FILE = os.path.join(TEST_DIR, args.results_name)

# --- training options
BATCH_SIZE = 32 * 8
N_EPOCHS = 90
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SCHEDULER = args.scheduler
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

exp_seed = args.seed

# --- model options
RESNET_MODEL = "resnet50"
LAP_HISTORY_LENGTH = 25
DEPRESSION_STRENGTH = 1.0
LENIENCY = 0.8
HOLD_OFF = 0
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 0


print(args.results_name)


results = {
    ds_name: {c_type: {run: {} for run in RUNS} for c_type in CORRUPTION_TYPE}
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
                run_int = int(run)
                if run_int not in results[ds_name][c_type].keys():
                    results[ds_name][c_type][run_int] = {}
                for depression in result_loaded[ds_name][c_type][run].keys():
                    depression_bool = depression == "true"
                    if depression_bool not in results[ds_name][c_type][run_int].keys():
                        results[ds_name][c_type][run_int][depression_bool] = (
                            result_loaded[ds_name][c_type][run][depression]
                        )

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


class TinyImagenet(torchdata.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform

        if train:
            self.dataset = torchdatasets.ImageFolder(
                root + "/tiny-imagenet-200/train", transform=transform
            )
            self.targets = self.dataset.targets

        else:
            class_to_idx = torchdatasets.ImageFolder(
                root + "/tiny-imagenet-200/train"
            ).class_to_idx

            self.dataset = torchdatasets.ImageFolder(
                root + "/tiny-imagenet-200/val", transform=transform
            )
            target_names = pd.read_table(
                root + "tiny-imagenet-200/val/val_annotations.txt", header=None
            )[1].values

            img_indices = [
                int(
                    self.dataset.imgs[idx][0].split("/")[-1].split("_")[1].split(".")[0]
                )
                for idx in range(len(self.dataset))
            ]

            target_names = [target_names[idx] for idx in img_indices]

            self.targets = [class_to_idx[i] for i in target_names]

        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx][0], self.targets[idx]

        x = self.transform(x)

        return x, y


class SourceDataset(torchdata.Dataset):
    def __init__(self, dataset, n_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        self.dataset = dataset
        self.n_sources = n_sources
        self.sources = self.rng.choice(n_sources, len(dataset), replace=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        source = self.sources[idx]
        x, y = self.dataset[idx]

        # return data and source
        return x, y, source


class SourceDatasetRandomLabel(torchdata.Dataset):
    def __init__(self, dataset, targets, n_sources, n_corrupt_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        self.dataset = dataset
        self.n_sources = n_sources
        self.unique_targets = np.unique(targets)

        self.sources = self.rng.choice(n_sources, len(dataset), replace=True)
        self.corrupt_sources = self.rng.choice(
            n_sources, n_corrupt_sources, replace=False
        )

        self.corrupt_targets = self.rng.choice(
            self.unique_targets, len(dataset), replace=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        source = self.sources[idx]
        x, y = self.dataset[idx]

        # if source is corrupt, return random label
        if source in self.corrupt_sources:
            y = self.corrupt_targets[idx]

        # return data and source
        return x, y, source


class TransformSourceDataset(torchdata.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y, source = self.dataset[idx]
        return self.transform(x), y, source


class ResNet(nn.Module):

    def __init__(
        self,
        n_classes=200,
        criterion=nn.CrossEntropyLoss(reduce="none"),
        model_name="resnet18",
        **resnet_kwargs,
    ):
        super(ResNet, self).__init__()

        assert model_name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ], "Please ensure you are using a valid ResNet model."

        self.model_class = getattr(torchmodels, model_name)
        self.model = self.model_class(**resnet_kwargs, weights=None)

        # change the last layer to output n_classes classes
        final_layer_features_in = self.model.fc.in_features
        self.model.fc = nn.Linear(final_layer_features_in, n_classes)
        self.criterion = criterion

    def forward(self, X, y=None, return_loss=False):

        out = X

        out = self.model(out)

        if return_loss:
            assert y is not None
            loss = self.criterion(out, y)
            return loss, out

        return (out,)


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
                losses=loss,
                sources=sources,
                # writer=writer, writer_prefix="label"
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
    scheduler=None,
):
    model.to(device)

    model.train()

    train_loss = 0
    train_total = 0
    train_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=200)

    for batch_idx, (inputs, targets, sources) in enumerate(train_loader):
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
            if batch_idx % int(len(train_loader) / 5) == 0:
                writer.add_scalar(
                    "Train Loss",
                    loss,
                    epoch_number * len(train_loader) + batch_idx,
                )

        train_acc_meter.update(outputs, targets)
        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

        pbar.update(1)

    if scheduler is not None:
        scheduler.step()

    metrics = train_acc_meter.compute_key_value()

    return train_loss / train_total, metrics["accuracy01"], metrics["accuracy05"]


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    test_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=200)
    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = model(inputs, y=targets, return_loss=True)

            loss = torch.mean(loss)
            test_acc_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        metrics = test_acc_meter.compute_key_value()
    return test_loss / test_total, metrics["accuracy01"], metrics["accuracy05"]


for dataset_name in DATASET_NAMES:
    for c_type in CORRUPTION_TYPE:
        for run in RUNS:
            print(
                f"for dataset {dataset_name}",
                f"and corruption type {c_type}",
                f"and run {run} the following depression has been completed",
                results[dataset_name][c_type][run].keys(),
            )


transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

train_ds_base = TinyImagenet(
    DATA_DIR,
    transform=transform,
    train=True,
)

test_ds = TinyImagenet(
    DATA_DIR,
    transform=transform,
    train=False,
)

exp_number = 0
for run in RUNS:
    for dataset_name in DATASET_NAMES:
        for corruption_type in CORRUPTION_TYPE:
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
                    len(DEPRESSION)
                    * len(RUNS)
                    * len(CORRUPTION_TYPE)
                    * len(DATASET_NAMES),
                    "with seed",
                    exp_seed,
                    "...",
                )

                print("Dataset:", dataset_name)
                print("Corruption Type:", corruption_type)
                print("Run:", run)
                print("Depression:", depression)

                if corruption_type == "random_label":
                    train_ds = SourceDatasetRandomLabel(
                        dataset=train_ds_base,
                        targets=train_ds_base.targets,
                        n_sources=N_SOURCES,
                        n_corrupt_sources=N_CORRUPT_SOURCES,
                        seed=exp_seed,
                    )
                    corrupt_sources = [int(s) for s in train_ds.corrupt_sources]
                elif corruption_type == "original":
                    train_ds = SourceDataset(
                        dataset=train_ds_base, n_sources=N_SOURCES, seed=exp_seed
                    )
                    corrupt_sources = []
                else:
                    raise ValueError("Unknown corruption type")

                train_ds = ToMemory(train_ds, device=DEVICE)
                test_ds = ToMemory(test_ds, device=DEVICE)

                train_ds = TransformSourceDataset(
                    train_ds,
                    transform=transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                )

                train_ds, val_ds = torchdata.random_split(
                    train_ds,
                    [
                        int(0.9 * len(train_ds)),
                        len(train_ds) - int(0.9 * len(train_ds)),
                    ],
                )

                print("The corrupt sources are:", corrupt_sources)

                train_dl = torchdata.DataLoader(
                    train_ds, batch_size=BATCH_SIZE, shuffle=True
                )
                val_dl = torchdata.DataLoader(
                    val_ds, batch_size=BATCH_SIZE, shuffle=False
                )
                test_dl = torchdata.DataLoader(
                    test_ds, batch_size=BATCH_SIZE, shuffle=False
                )

                model = ResNet(
                    n_classes=200,
                    criterion=nn.CrossEntropyLoss(reduction="none"),
                    model_name=RESNET_MODEL,
                )
                model = model.to(DEVICE)

                # Observe that all parameters are being optimized
                optimiser = torch.optim.SGD(
                    model.parameters(),
                    lr=LR,
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY,
                )

                if SCHEDULER:
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
                    )
                else:
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

                writer = SummaryWriter(
                    log_dir=os.path.join(
                        TEST_DIR,
                        "tb",
                        dataset_name,
                        f"{corruption_type}",
                        f"{run}",
                        f"{depression}",
                    )
                )
                # writer = None

                results_this_train = {}

                for epoch in range(N_EPOCHS):
                    pbar = tqdm.tqdm(
                        total=len(train_dl), desc=f"Training epoch {epoch+1}/{N_EPOCHS}"
                    )
                    train_loss, train_top1acc, train_top5acc = train_epoch(
                        model=model,
                        train_loader=train_dl,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        device=DEVICE,
                        epoch_number=epoch,
                        label_loss_weighting=label_loss_weighting,
                        writer=writer,
                        pbar=pbar,
                    )

                    val_loss, val_top1acc, val_top5acc = test(model, val_dl, DEVICE)
                    test_loss, test_top1acc, test_top5acc = test(model, test_dl, DEVICE)

                    if writer is not None:

                        writer.add_scalar("Validation Loss", val_loss, epoch)
                        writer.add_scalar("Validation Acc", val_top1acc, epoch)
                        writer.add_scalar("Validation Top5Acc", val_top5acc, epoch)
                        writer.add_scalar("Test Loss", test_loss, epoch)
                        writer.add_scalar("Test Acc", test_top1acc, epoch)
                        writer.add_scalar("Test Top5Acc", test_top5acc, epoch)

                    results_this_train[epoch] = {
                        "train_loss": train_loss,
                        "train_top1acc": train_top1acc,
                        "train_top5acc": train_top5acc,
                        "val_loss": val_loss,
                        "val_top1acc": val_top1acc,
                        "val_top5acc": val_top5acc,
                        "test_loss": test_loss,
                        "test_top1acc": test_top1acc,
                        "test_top5acc": test_top5acc,
                    }

                    pbar.set_postfix(
                        {
                            "tr_l": train_loss,
                            "te_l": test_loss,
                            "te_1a": test_top1acc,
                            "te_5a": test_top5acc,
                        }
                    )

                    pbar.refresh()

                    pbar.close()

                results_this_train["corrupt_sources"] = corrupt_sources

                results[dataset_name][corruption_type][run][
                    depression
                ] = results_this_train

                print(results_this_train[epoch])

                if writer is not None:
                    writer.flush()
                    writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
