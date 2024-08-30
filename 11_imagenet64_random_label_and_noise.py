import os
from pathlib import Path
import argparse
import json
import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
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
    default="./outputs/imagenet64_random_label_and_noise/",
)
parser.add_argument("--depression", action="store_true")
parser.add_argument("--no_depression", action="store_true")
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = [
    "imagenet64",
]
N_SOURCES = 10
N_CORRUPT_SOURCES = 5
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


# https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
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
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 0


torch.set_float32_matmul_precision("high")


print(args.results_name)
print("Running on device:", DEVICE)


results = {ds_name: {run: {} for run in RUNS} for ds_name in DATASET_NAMES}


# if results file exists, load it
try:
    with open(RESULTS_FILE, "r") as fp:
        result_loaded = json.load(fp)
    print("loaded previous results")
    for ds_name in result_loaded.keys():
        if ds_name not in results.keys():
            results[ds_name] = {}
        for run in result_loaded[ds_name].keys():
            run_int = int(run)
            if run_int not in results[ds_name].keys():
                results[ds_name][run_int] = {}
            for depression in result_loaded[ds_name][run].keys():
                depression_bool = depression == "true"
                if depression_bool not in results[ds_name][run_int].keys():
                    results[ds_name][run_int][depression_bool] = result_loaded[ds_name][
                        run
                    ][depression]

except FileNotFoundError:
    pass


print(results)


## dataset


class ImageNet64(torchdata.Dataset):
    def __init__(self, root, train, file_index=None, transform=None, device="cpu"):

        self.root = os.path.join(root, "imagenet", "raw")
        self.train = train
        self.transform = transform

        files_to_load = [
            file for file in os.listdir(self.root) if "train_data_batch" in file
        ]

        if file_index is not None:
            files_to_load = [files_to_load[file_index]]

        if self.train:
            self.data = []
            self.targets = []
            for file in tqdm.tqdm(files_to_load, desc="Loading training data"):
                batch = self.load_data_file(os.path.join(self.root, file))
                self.data.append(torch.tensor(batch[0]).type(torch.uint8).to(device))
                self.targets.append(torch.tensor(batch[1]).long().to(device))
            self.data = torch.concat(self.data)
            self.targets = torch.concat(self.targets)
        else:
            batch = self.load_data_file(os.path.join(self.root, "val_data"))
            self.data = torch.tensor(batch[0]).type(torch.uint8).to(device)
            self.targets = torch.tensor(batch[1]).long().to(device)

    def unpickle(self, file_name):
        with open(file_name, "rb") as fo:
            dict = pickle.load(fo)
        return dict

    @staticmethod
    def get_filenames(root):
        root = os.path.join(root, "imagenet", "raw")
        return [file for file in os.listdir(root) if "train_data_batch" in file]

    def load_data_file(self, file_name):
        # edited from https://patrykchrabaszcz.github.io/Imagenet32/
        d = self.unpickle(file_name)
        x = d["data"]
        y = d["labels"]

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = np.array(y) - 1
        data_size = x.shape[0]

        img_size2 = 64 * 64

        x = np.dstack(
            (x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :])
        )
        x = x.reshape((x.shape[0], 64, 64, 3))
        x = x.transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        y_train = y[0:data_size]

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train.astype("uint8"), y_train.astype("int64")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


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
        self.length = len(dataset)
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
        return self.length


class SourceDataset(torchdata.Dataset):
    def __init__(self, dataset, n_sources, device="cpu", seed=None):
        self.rng = np.random.default_rng(seed)
        self.dataset = dataset
        self.n_sources = n_sources
        self.sources = self.rng.choice(n_sources, len(dataset), replace=True)
        self.sources = torch.tensor(self.sources, device=device).long()

    def __getitem__(self, index):
        x, y = self.dataset[index]
        source = self.sources[index]
        return x, y, source

    def __len__(self):
        return len(self.dataset)


class SourceDatasetRandomLabel(torchdata.Dataset):
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
        self.unique_targets = np.unique(targets.cpu().numpy())

        self.corrupt_sources = corrupt_sources

        self.corrupt_targets = self.rng.choice(
            self.unique_targets, len(dataset), replace=True
        )

        self.corrupt_targets = torch.tensor(
            self.corrupt_targets,
            device=targets.device,
        ).long()

        self.noise_levels = {
            source: noise_level
            for source, noise_level in zip(corrupt_sources, noise_levels)
        }
        self.noisy_point = np.zeros(len(dataset)).astype(bool)
        for source in corrupt_sources:
            self.noisy_point[sources.cpu().numpy() == source] = (
                self.rng.random(len(self.noisy_point[sources.cpu().numpy() == source]))
                < self.noise_levels[source]
            )
        self.noisy_point = torch.tensor(self.noisy_point, device=targets.device)

        self.sources = sources
        self.targets = torch.where(self.noisy_point, self.corrupt_targets, targets)

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


class SourceDatasetAddRandomInputNoise(torchdata.Dataset):
    def __init__(self, dataset, sources, corrupt_sources, noise_levels, seed=None):
        # dataset x must be uint8

        self.seed = np.random.default_rng(seed=seed).integers(1e6)
        self.rng = np.random.default_rng(self.seed)

        self.dataset = dataset

        self.corrupt_sources = corrupt_sources

        self.noise_levels = {
            source: noise_level
            for source, noise_level in zip(corrupt_sources, noise_levels)
        }
        self.noisy_point = np.zeros(len(dataset)).astype(bool)
        for source in corrupt_sources:
            self.noisy_point[sources.cpu().numpy() == source] = (
                self.rng.random(len(self.noisy_point[sources.cpu().numpy() == source]))
                < self.noise_levels[source]
            )

        self.noisy_point = torch.tensor(self.noisy_point, device=sources.device)
        self.sources = sources
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _, _ = self.dataset[idx]
        y = self.targets[idx]
        source = self.sources[idx]

        noisy_point = self.noisy_point[idx]
        if noisy_point:
            # x = x + 0.5 * torch.empty_like(x).normal_(
            #     generator=torch.Generator(device=x.device).manual_seed(
            #         int(self.seed + idx)
            #     )
            # )
            noise = torch.randint(-64, 64, x.shape, device=x.device)
            x = (x + noise).to(torch.uint8)
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
        n_classes=1000,
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
                writer=None,
                # writer=writer,
                # writer_prefix="label",
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
    train_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=1000)

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
            writer.add_scalar(
                "Train Loss",
                loss,
                epoch_number * len(train_loader) + batch_idx,
            )

        train_acc_meter.update(outputs, targets)
        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

        pbar.update(1)

        if batch_idx % int(len(train_loader) / 5) == 0:
            pbar.set_postfix(
                {
                    "tr_l": train_loss / train_total,
                    "tr_1a": train_acc_meter.compute_key_value()["accuracy01"],
                    "tr_5a": train_acc_meter.compute_key_value()["accuracy05"],
                }
            )
            if writer is not None:
                writer.add_scalar(
                    "Train Acc",
                    train_acc_meter.compute_key_value()["accuracy01"],
                    epoch_number * len(train_loader) + batch_idx,
                )

    if scheduler is not None:
        scheduler.step()

    metrics = train_acc_meter.compute_key_value()

    return train_loss / train_total, metrics["accuracy01"], metrics["accuracy05"]


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    test_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=1000)
    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, batch in tqdm.tqdm(
            enumerate(test_loader), desc="Testing", total=len(test_loader)
        ):
            inputs, targets = batch[0], batch[1]  # excluding source if there is one
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = model(inputs, y=targets, return_loss=True)

            loss = torch.mean(loss)
            test_acc_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        metrics = test_acc_meter.compute_key_value()
    return test_loss / test_total, metrics["accuracy01"], metrics["accuracy05"]


for dataset_name in DATASET_NAMES:
    for run in RUNS:
        print(
            f"for dataset {dataset_name}",
            f"and run {run} the following depression has been completed",
            results[dataset_name][run].keys(),
        )


before_transform = transforms.Compose(
    [
        transforms.ToImage(),
    ]
)

after_transform = transforms.Compose(
    [
        transforms.ToDtype(torch.float32, scale=True),
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html#torchvision.models.resnet152
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


exp_number = 0
for run in RUNS:
    for dataset_name in DATASET_NAMES:
        # seed same for all depression types
        exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))
        for depression in DEPRESSION:

            exp_number += 1

            # only run experiments that have not been completed
            # this should only be true if the whole training run for an experiment
            # has not been completed
            if depression in results[dataset_name][run].keys():
                print(
                    "Skipping the following experiment as already completed:",
                    dataset_name,
                    run,
                    depression,
                    "...",
                )
                continue

            print(
                "Performing experiment",
                exp_number,
                "of",
                len(RUNS) * len(DATASET_NAMES) * len(DEPRESSION),
                "with seed",
                exp_seed,
                "...",
            )

            print("Dataset:", dataset_name)
            print("Run:", run)
            print("Depression:", depression)

            corrupt_sources = np.random.default_rng(exp_seed).choice(
                N_SOURCES, N_CORRUPT_SOURCES, replace=False
            )

            random_input_corrupt_sources = corrupt_sources[: N_CORRUPT_SOURCES // 2]
            random_label_corrupt_sources = corrupt_sources[N_CORRUPT_SOURCES // 2 :]

            print("Corrupt sources:", corrupt_sources)
            print("Random input corrupt sources:", random_input_corrupt_sources)
            print("Random label corrupt sources:", random_label_corrupt_sources)

            ### adding random labels and input noise
            ### and then saving data to gpu for faster training

            train_file_indices = ImageNet64.get_filenames(DATA_DIR)

            train_ds_list = []

            for file_index in range(len(train_file_indices)):
                print("File index:", file_index)
                single_imagenet_ds = ImageNet64(
                    root=DATA_DIR,
                    file_index=file_index,
                    train=True,
                    transform=before_transform,
                    device=DEVICE,
                )

                source_dataset = SourceDataset(
                    dataset=single_imagenet_ds,
                    n_sources=N_SOURCES,
                    device=DEVICE,
                    seed=exp_seed + file_index,
                )

                source_random_label_dataset = SourceDatasetRandomLabel(
                    dataset=source_dataset,
                    targets=single_imagenet_ds.targets,
                    sources=source_dataset.sources,
                    corrupt_sources=random_label_corrupt_sources,
                    noise_levels=NOISE_LEVELS,
                    seed=exp_seed + 1 + file_index,
                )

                source_random_label_and_noise_dataset = (
                    SourceDatasetAddRandomInputNoise(
                        dataset=source_random_label_dataset,
                        sources=source_dataset.sources,
                        corrupt_sources=random_input_corrupt_sources,
                        noise_levels=NOISE_LEVELS,
                        seed=exp_seed + 2 + file_index,
                    )
                )

                train_ds = TransformSourceDataset(
                    source_random_label_and_noise_dataset,
                    transform=transforms.Compose(
                        [
                            after_transform,
                        ]
                    ),
                )

                train_ds = ToMemory(
                    train_ds,
                    device=DEVICE,
                )

                # putting training data in memory
                memory_train_dl = torchdata.DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                for b in tqdm.tqdm(
                    memory_train_dl, desc="Loading training data into gpu memory"
                ):
                    pass

                train_ds.dataset = None
                del source_random_label_and_noise_dataset
                del source_random_label_dataset
                del source_dataset
                del single_imagenet_ds

                if "cuda" in DEVICE.type:
                    torch.cuda.empty_cache()

                train_ds_list.append(train_ds)

            train_ds = torchdata.ConcatDataset(train_ds_list)

            train_ds, val_ds = torchdata.random_split(
                train_ds,
                lengths=[
                    len(train_ds) - int(0.1 * len(train_ds)),
                    int(0.1 * len(train_ds)),
                ],
                generator=torch.Generator().manual_seed(exp_seed),
            )

            test_ds = ImageNet64(
                root=DATA_DIR,
                train=False,
                transform=transforms.Compose([before_transform, after_transform]),
                device=DEVICE,
            )

            train_ds = TransformSourceDataset(
                train_ds,
                transform=transforms.Compose(
                    [
                        # after_transform,
                        transforms.RandomHorizontalFlip(),
                    ]
                ),
            )

            train_dl = torchdata.DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            val_dl = torchdata.DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            test_dl = torchdata.DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            # make checkpoint directory and load model, optimiser, scheduler and source loss weighting
            # if it exists
            checkpoint_dir = os.path.join(
                TEST_DIR,
                "checkpoints",
                dataset_name,
                f"{run}",
                f"{depression}",
            )
            if not Path(checkpoint_dir).exists():
                os.makedirs(checkpoint_dir)

            checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pth")

            if Path(checkpoint_file).exists():
                # load torch items
                checkpoint = torch.load(checkpoint_file)
                model = checkpoint["model"]
                optimiser = checkpoint["optimiser"]
                if SCHEDULER:
                    scheduler = checkpoint["scheduler"]
                else:
                    scheduler = None
                if depression:
                    label_loss_weighting = checkpoint["label_loss_weighting"]
                else:
                    label_loss_weighting = None

                # load results
                with open(os.path.join(checkpoint_dir, "results.json"), "r") as fp:
                    results_this_train = json.load(fp)

                prev_epoch = max([int(e) for e in results_this_train.keys()]) + 1

                print("Loaded checkpoint from epoch", prev_epoch)
            else:
                prev_epoch = 0

                model = ResNet(
                    n_classes=1000,
                    criterion=nn.CrossEntropyLoss(reduction="none"),
                    model_name=RESNET_MODEL,
                )
                model = model.to(DEVICE)

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

                results_this_train = {}

            writer = SummaryWriter(
                log_dir=os.path.join(
                    TEST_DIR,
                    "tb",
                    dataset_name,
                    f"{run}",
                    f"{depression}",
                )
            )
            # writer = None

            for epoch in range(prev_epoch, N_EPOCHS):
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
                        "va_l": val_loss,
                        "te_l": test_loss,
                        "te_1a": test_top1acc,
                        "va_1a": val_top1acc,
                        "te_5a": test_top5acc,
                    }
                )

                to_save = {
                    "model": model,
                    "optimiser": optimiser,
                    "label_loss_weighting": label_loss_weighting,
                }
                if scheduler is not None:
                    to_save["scheduler"] = scheduler

                torch.save(
                    to_save,
                    checkpoint_file,
                )

                writer.flush()

                with open(os.path.join(checkpoint_dir, "results.json"), "w") as fp:
                    json.dump(results_this_train, fp)

                pbar.refresh()
                pbar.close()

            results_this_train["corrupt_sources"] = corrupt_sources.tolist()

            results[dataset_name][run][depression] = results_this_train

            print(results_this_train[epoch])

            if writer is not None:
                writer.flush()
                writer.close()

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)
