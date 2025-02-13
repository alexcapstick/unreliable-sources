import os
from pathlib import Path
import argparse
import time
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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
    default="./outputs/california_housing/",
)
parser.add_argument("--depression", action="store_true")
parser.add_argument("--no_depression", action="store_true")
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = [
    "california_housing",
]
CORRUPTION_TYPE = [
    "original",
    "random_label",
]
N_SOURCES = 10
N_CORRUPT_SOURCES = 4
NOISE_LEVELS = np.ones((N_CORRUPT_SOURCES,))
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
    if args.no_depression:
        DEPRESSION = [False]
if args.scheduler:
    args.results_name = args.results_name.replace(".json", "_scheduler.json")
RESULTS_FILE = os.path.join(TEST_DIR, args.results_name)


# --- training options
BATCH_SIZE = 256
N_EPOCHS = 200
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SCHEDULER = args.scheduler
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5


exp_seed = args.seed

# --- model options
RESNET_MODEL = "resnet50"
LAP_HISTORY_LENGTH = 25
DEPRESSION_STRENGTH = 1.0
LENIENCY = 0.8
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 0


print(args.results_name)

time.sleep(5)


results = {
    ds_name: {c: {run: {} for run in RUNS} for c in CORRUPTION_TYPE}
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
        for corruption in result_loaded[ds_name].keys():
            if corruption not in results[ds_name].keys():
                results[ds_name][corruption] = {}
            for run in result_loaded[ds_name][corruption].keys():
                run_int = int(run)
                if run_int not in results[ds_name][corruption].keys():
                    results[ds_name][corruption][run_int] = {}
                for depression in result_loaded[ds_name][corruption][run].keys():
                    depression_bool = depression == "true"
                    if (
                        depression_bool
                        not in results[ds_name][corruption][run_int].keys()
                    ):
                        results[ds_name][corruption][run_int][depression_bool] = (
                            result_loaded[ds_name][corruption][run][depression]
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


class SourceDatasetRandomContinuousLabel(torchdata.Dataset):
    def __init__(
        self,
        dataset,
        targets,
        sources,
        corrupt_sources,
        noise_levels,
        seed=None,
    ):
        # targets must be a tensor

        self.seed = np.random.default_rng(seed=seed).integers(1e6)
        self.rng = np.random.default_rng(self.seed)

        self.dataset = dataset
        self.lower_bound_targets = torch.min(targets)
        self.upper_bound_targets = torch.max(targets)

        self.corrupt_sources = corrupt_sources

        self.corrupt_targets = torch.empty_like(targets).uniform_(
            self.lower_bound_targets,
            self.upper_bound_targets,
            generator=torch.Generator(device=targets.device).manual_seed(
                int(self.rng.integers(1e6))
            ),
        )

        self.noise_levels = {
            source: noise_level
            for source, noise_level in zip(corrupt_sources, noise_levels)
        }
        self.noisy_point = torch.zeros(len(dataset), device=targets.device).to(bool)
        for source in corrupt_sources:
            self.noisy_point[sources == source] = (
                torch.rand(
                    len(self.noisy_point[sources == source]),
                    device=targets.device,
                    generator=torch.Generator(device=targets.device).manual_seed(
                        int(self.rng.integers(1e6))
                    ),
                )
                < self.noise_levels[source]
            )

        self.sources = sources
        self.targets = torch.where(
            self.noisy_point.reshape(*targets.shape), self.corrupt_targets, targets
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, _, _ = self.dataset[idx]
        # get target
        y = self.targets[idx]
        # get source
        source = self.sources[idx]

        # return data and source
        return x, y, source


### model


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, criterion=nn.MSELoss(reduction="none")):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

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
    grad_clip=None,
):
    model.to(device)
    model.train()

    optimiser.zero_grad()

    loss, outputs = model(x, y=y, return_loss=True)

    if not warmup:
        if label_loss_weighting is not None:
            loss = label_loss_weighting(
                losses=loss.reshape(-1),
                sources=sources.reshape(-1),
                # writer=writer,
                # writer_prefix="label",
            )

    loss = torch.mean(loss)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    grad_clip=None,
    writer=None,
    scheduler=None,
):
    model.to(device)

    model.train()

    train_loss = 0
    train_total = 0

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
            grad_clip=grad_clip,
            warmup=False,
            writer=writer,
        )

        if writer is not None:
            writer.add_scalar(
                "Train Loss",
                loss,
                epoch_number * len(train_loader) + batch_idx,
            )

        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

        pbar.update(1)

    if scheduler is not None:
        scheduler.step()

    return train_loss / train_total


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = model(inputs, y=targets, return_loss=True)

            loss = torch.mean(loss)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

    return test_loss / test_total


for dataset_name in DATASET_NAMES:
    for c in CORRUPTION_TYPE:
        for run in RUNS:
            print(
                f"for dataset {dataset_name}",
                f"and corruption type {c}",
                f"and run {run} the following depression has been completed",
                results[dataset_name][c][run].keys(),
            )


X, y = fetch_california_housing(return_X_y=True)
X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


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
                    len(RUNS)
                    * len(DATASET_NAMES)
                    * len(CORRUPTION_TYPE)
                    * len(DEPRESSION),
                    "with seed",
                    exp_seed,
                    "...",
                )

                print("Dataset:", dataset_name)
                print("Corruption Type:", corruption_type)
                print("Run:", run)
                print("Depression:", depression)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=exp_seed
                )

                sources_train = np.random.default_rng(exp_seed).choice(
                    N_SOURCES, size=len(y_train)
                )

                train_ds = torchdata.TensorDataset(
                    torch.from_numpy(X_train),
                    torch.from_numpy(y_train),
                    torch.from_numpy(sources_train),
                )
                test_ds = torchdata.TensorDataset(
                    torch.from_numpy(X_test), torch.from_numpy(y_test)
                )

                corrupt_sources = np.random.default_rng(exp_seed).choice(
                    N_SOURCES, size=N_CORRUPT_SOURCES, replace=False
                )

                if corruption_type == "random_label":

                    train_ds = SourceDatasetRandomContinuousLabel(
                        dataset=train_ds,
                        targets=torch.from_numpy(y_train),
                        sources=torch.from_numpy(sources_train),
                        corrupt_sources=corrupt_sources,
                        noise_levels=torch.ones(len(corrupt_sources)),
                        seed=exp_seed,
                    )

                train_ds = ToMemory(train_ds, device=DEVICE)
                test_ds = ToMemory(test_ds, device=DEVICE)

                train_dl = torchdata.DataLoader(train_ds, batch_size=128, shuffle=True)
                test_dl = torchdata.DataLoader(test_ds, batch_size=128, shuffle=False)

                model = MLP(
                    input_dim=X_train.shape[1],
                    output_dim=1,
                    criterion=nn.MSELoss(reduction="none"),
                )
                model = model.to(DEVICE)

                # Loss Function
                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=LR,
                )

                if SCHEDULER:
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimiser,
                        step_size=SCHEDULER_STEP_SIZE,
                        gamma=SCHEDULER_GAMMA,
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
                        corruption_type,
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
                    train_loss = train_epoch(
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

                    test_loss = test(model, test_dl, DEVICE)

                    if writer is not None:
                        writer.add_scalar("Test Loss", test_loss, epoch)

                    results_this_train[epoch] = {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }

                    pbar.set_postfix(
                        {
                            "tr_l": train_loss,
                            "te_l": test_loss,
                        }
                    )

                    pbar.refresh()

                    pbar.close()

                results_this_train["corrupt_sources"] = corrupt_sources.tolist()

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
