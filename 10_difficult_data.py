from pathlib import Path
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torchvision.datasets as vision_datasets
import torchvision.transforms.v2 as transforms

from loss_adapted_plasticity import SourceLossWeighting


SEED = 42

RUNS = [0, 1, 2, 3, 4]

DATA_DIR = "./data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("outputs", "difficult_data")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_EPOCHS = 25


# this includes the final source, which will be CIFAR and not noisy
N_SOURCES = 100
CORRUPTION_TYPE = [
    "random_label",
    "original",
]

# this is the number of sources that will be noisy
# -- should be less than N_SOURCES - 1
N_CORRUPT_SOURCES = 95

HISTORY_LENGTH = 50
LENIENCY = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
DEPRESSION_STRENGTH = 1.0
WARMUP_ITERS = 0

SAVE_DIR.mkdir(parents=True, exist_ok=True)

image_transformations = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28), antialias=True),
    ]
)

common_args = dict(
    root=DATA_DIR,
    transform=image_transformations,
    download=True,
)

mnist_dataset_train = vision_datasets.MNIST(train=True, **common_args)
mnist_dataset_test = vision_datasets.MNIST(train=False, **common_args)

cifar_dataset_train = vision_datasets.CIFAR10(train=True, **common_args)
cifar_dataset_test = vision_datasets.CIFAR10(train=False, **common_args)


class SubsetNewLabel(torchdata.Dataset):
    def __init__(self, dataset, indices, target_transform):
        self.dataset = dataset
        self.indices = indices
        self.subset_dataset = torchdata.Subset(dataset, indices)
        self.target_transform = target_transform

    def __getitem__(self, idx):
        data, target = self.subset_dataset[idx]
        return data, self.target_transform(target)

    def __len__(self):
        return len(self.subset_dataset)


class SourceDatasetOriginal(torchdata.Dataset):
    def __init__(self, dataset, targets, sources):

        self.dataset = dataset
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, y, source = self.dataset[idx][0], self.targets[idx], self.sources[idx]

        x = torch.tensor(x) if not torch.is_tensor(x) else x
        y = torch.tensor(y) if not torch.is_tensor(y) else y
        source = torch.tensor(source) if not torch.is_tensor(source) else source

        # return data and source
        return x, y, source


class SourceDatasetRandomLabel(torchdata.Dataset):
    def __init__(self, dataset, targets, sources, n_corrupt_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        targets = torch.tensor(targets) if not torch.is_tensor(targets) else targets
        sources = torch.tensor(sources) if not torch.is_tensor(sources) else sources

        self.dataset = dataset

        n_sources = len(torch.unique(sources))
        self.unique_targets = torch.unique(targets)

        self.sources = sources

        self.corrupt_sources = torch.tensor(
            self.rng.choice(n_sources, n_corrupt_sources, replace=False),
            device=sources.device,
        )

        self.targets = torch.tensor(
            self.rng.choice(self.unique_targets, len(dataset), replace=True),
            device=targets.device,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, y = self.dataset[idx]
        source = self.sources[idx]

        # if source is corrupt, return random label
        if source in self.corrupt_sources:
            y = self.targets[idx]

        x = torch.tensor(x) if not torch.is_tensor(x) else x
        y = torch.tensor(y) if not torch.is_tensor(y) else y
        source = torch.tensor(source) if not torch.is_tensor(source) else source

        # return data and source
        return x, y, source


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
                output = torch.tensor(output)
            except:
                pass
            try:
                output_on_device.append(i.to(self.device))
            except:
                output_on_device.append(i)

        self.memory_dataset[index] = output_on_device
        return output

    def __len__(self):
        return len(self.dataset)


mnist_dataset_test = ToMemory(mnist_dataset_test)
cifar_dataset_test = ToMemory(cifar_dataset_test)


for c_type in CORRUPTION_TYPE:
    exp_seed = SEED
    results_standard = {}
    results_lap = {}
    results_standard["mnist_and_cifar"] = {}
    results_standard["mnist_and_cifar"][c_type] = {}
    results_lap["mnist_and_cifar"] = {}
    results_lap["mnist_and_cifar"][c_type] = {}
    for run in RUNS:
        exp_seed += 1

        print(f"Running {c_type} corruption, run {run}")

        results_standard["mnist_and_cifar"][c_type][run] = {}
        results_lap["mnist_and_cifar"][c_type][run] = {}

        mnist_dataset_test.targets = mnist_dataset_test.dataset.targets
        cifar_dataset_test.targets = cifar_dataset_test.dataset.targets

        unique_cifar_targets = np.unique(cifar_dataset_train.targets)
        cifar_targets_to_keep = np.random.default_rng(exp_seed).choice(
            unique_cifar_targets, 2, replace=False
        )

        print("Using CIFAR classes:", cifar_targets_to_keep)

        n_mnist_targets = len(torch.unique(mnist_dataset_train.targets))

        train_class_idx = torch.where(
            torch.isin(
                torch.tensor(cifar_dataset_train.targets),
                torch.tensor(cifar_targets_to_keep),
            )
        )[0]
        test_class_idx = torch.where(
            torch.isin(
                torch.tensor(cifar_dataset_test.targets),
                torch.tensor(cifar_targets_to_keep),
            )
        )[0]

        cifar_class_mapping = {c: i for i, c in enumerate(cifar_targets_to_keep)}

        print("CIFAR class mapping:", cifar_class_mapping)

        cifar_dataset_subset_train = SubsetNewLabel(
            cifar_dataset_train,
            indices=train_class_idx,
            target_transform=lambda x: cifar_class_mapping[x] + n_mnist_targets,
        )
        cifar_dataset_subset_test = SubsetNewLabel(
            cifar_dataset_test,
            indices=test_class_idx,
            target_transform=lambda x: cifar_class_mapping[x] + n_mnist_targets,
        )

        new_cifar_targets_train = torch.tensor(
            [
                cifar_class_mapping[c.item()] + n_mnist_targets
                for c in torch.tensor(cifar_dataset_train.targets)[train_class_idx]
            ]
        )
        new_cifar_targets_test = torch.tensor(
            [
                cifar_class_mapping[c.item()] + n_mnist_targets
                for c in torch.tensor(cifar_dataset_test.targets)[test_class_idx]
            ]
        )

        print(
            "Original CIFAR targets:",
            np.array(cifar_dataset_train.targets)[train_class_idx],
        )
        print("New CIFAR targets:", new_cifar_targets_train)

        cifar_dataset_subset_train.targets = new_cifar_targets_train
        cifar_dataset_subset_test.targets = new_cifar_targets_test

        mnist_train_sources = torch.randint(
            0,
            N_SOURCES - 1,
            size=(len(mnist_dataset_train),),
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(exp_seed),
        )

        if c_type == "random_label":
            # make mnist have random label noise

            mnist_dataset_noisy_train = SourceDatasetRandomLabel(
                mnist_dataset_train,
                targets=mnist_dataset_train.targets,
                sources=mnist_train_sources,
                n_corrupt_sources=N_CORRUPT_SOURCES,
                seed=exp_seed,
            )

            corrupt_sources = mnist_dataset_noisy_train.corrupt_sources.tolist()
            print("Corrupt sources:", corrupt_sources)

        elif c_type == "original":
            mnist_dataset_noisy_train = SourceDatasetOriginal(
                mnist_dataset_train,
                targets=mnist_dataset_train.targets,
                sources=mnist_train_sources,
            )

            corrupt_sources = []

        cifar_dataset_non_noisy_train = SourceDatasetOriginal(
            cifar_dataset_subset_train,
            targets=cifar_dataset_subset_train.targets,
            sources=torch.zeros(len(cifar_dataset_subset_train), device=DEVICE)
            + N_SOURCES
            - 1,
        )

        dataset_train = torchdata.ConcatDataset(
            [mnist_dataset_noisy_train, cifar_dataset_non_noisy_train]
        )
        dataset_test = torchdata.ConcatDataset(
            [mnist_dataset_test, cifar_dataset_subset_test]
        )

        dataset_train.targets = torch.cat(
            [mnist_dataset_noisy_train.targets, cifar_dataset_non_noisy_train.targets]
        )
        dataset_test.targets = torch.cat(
            [mnist_dataset_test.targets, cifar_dataset_subset_test.targets]
        )

        dataset_train = ToMemory(dataset_train, device=DEVICE)
        dataset_test = ToMemory(dataset_test, device=DEVICE)

        dataset_train.targets = dataset_train.dataset.targets
        dataset_test.targets = dataset_test.dataset.targets

        dataset_train_train, dataset_train_val = torchdata.random_split(
            dataset_train,
            [
                int(0.9 * len(dataset_train)),
                len(dataset_train) - int(0.9 * len(dataset_train)),
            ],
            generator=torch.Generator().manual_seed(exp_seed),
        )

        unique_targets = torch.unique(dataset_train.targets)
        train_dl = torchdata.DataLoader(
            dataset_train_train, batch_size=BATCH_SIZE, shuffle=True
        )
        train_full_dl = torchdata.DataLoader(
            dataset_train, batch_size=BATCH_SIZE, shuffle=True
        )
        val_dl = torchdata.DataLoader(
            dataset_train_val, batch_size=BATCH_SIZE, shuffle=False
        )
        test_dl = torchdata.DataLoader(
            dataset_test, batch_size=BATCH_SIZE, shuffle=False
        )

        print("Unique targets:", unique_targets)

        def get_model():

            model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
                nn.Linear(9216, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, len(unique_targets)),
            )

            model.to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss(reduction="none")

            return model, optimizer, criterion

        print("\n", "Training model in standard way", "\n")

        model, optimizer, criterion = get_model()

        # standard training
        standard_results = {}
        for epoch in range(N_EPOCHS):
            model.train()
            for x, y, _ in tqdm.tqdm(train_full_dl, desc=f"Epoch {epoch}"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.mean().backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            class_correct = torch.zeros(len(unique_targets), device=DEVICE)
            class_total = torch.zeros(len(unique_targets), device=DEVICE)
            with torch.no_grad():
                for b in test_dl:
                    x, y = b[:2]
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    y_hat = model(x)
                    _, predicted = torch.max(y_hat, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    # performance on each class
                    for c in unique_targets:
                        class_total[c] += (y == c).sum()
                        class_correct[c] += ((predicted == c) * (y == c)).sum()

                standard_results[epoch] = {
                    f"accuracy": correct / total,
                    f"class_accuracy": (class_correct / class_total).tolist(),
                }

            print(f"Epoch {epoch}: {correct/total}")

        results_standard["mnist_and_cifar"][c_type][run]["standard"] = standard_results

        print("\n", "Training model with loss adapted plasticity", "\n")

        for l in LENIENCY:

            print(f"Leniency: {l}")
            # lap training with validation
            model, optimizer, criterion = get_model()

            source_loss_weighting = SourceLossWeighting(
                history_length=HISTORY_LENGTH,
                leniency=l,
                depression_strength=DEPRESSION_STRENGTH,
                device=DEVICE,
                warmup_iters=WARMUP_ITERS,
            )

            lap_results = {}
            for epoch in range(N_EPOCHS):
                model.train()
                for x, y, s in tqdm.tqdm(train_dl, desc=f"Epoch {epoch}"):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    s = s.to(DEVICE)

                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    loss = source_loss_weighting(losses=loss, sources=s)
                    loss.mean().backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                class_correct = torch.zeros(len(unique_targets), device=DEVICE)
                class_total = torch.zeros(len(unique_targets), device=DEVICE)
                with torch.no_grad():
                    for b in val_dl:
                        x, y = b[:2]
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)

                        y_hat = model(x)
                        _, predicted = torch.max(y_hat, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                        # performance on each class
                        for c in unique_targets:
                            class_total[c] += (y == c).sum()
                            class_correct[c] += ((predicted == c) * (y == c)).sum()

                    lap_results[epoch] = {
                        f"accuracy": correct / total,
                        f"class_accuracy": (class_correct / class_total).tolist(),
                        "source_weights": source_loss_weighting.get_source_unrelaibility(),
                        "source_values": source_loss_weighting.get_source_order(),
                        "corrupt_sources": corrupt_sources,
                    }

                print(f"Epoch {epoch}: {correct/total}")
                print(source_loss_weighting.get_source_unrelaibility())
                print(source_loss_weighting.get_source_order())

            results_lap["mnist_and_cifar"][c_type][run][
                f"lap_{l}_validation"
            ] = lap_results

            print(f"Leniency: {l}")
            # lap training on full dataset
            model, optimizer, criterion = get_model()

            source_loss_weighting = SourceLossWeighting(
                history_length=HISTORY_LENGTH,
                leniency=l,
                depression_strength=DEPRESSION_STRENGTH,
                device=DEVICE,
                warmup_iters=WARMUP_ITERS,
            )

            lap_results = {}
            for epoch in range(N_EPOCHS):
                model.train()
                for x, y, s in tqdm.tqdm(train_full_dl, desc=f"Epoch {epoch}"):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    s = s.to(DEVICE)

                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    loss = source_loss_weighting(losses=loss, sources=s)
                    loss.mean().backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                class_correct = torch.zeros(len(unique_targets), device=DEVICE)
                class_total = torch.zeros(len(unique_targets), device=DEVICE)
                with torch.no_grad():
                    for b in test_dl:
                        x, y = b[:2]
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)

                        y_hat = model(x)
                        _, predicted = torch.max(y_hat, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                        # performance on each class
                        for c in unique_targets:
                            class_total[c] += (y == c).sum()
                            class_correct[c] += ((predicted == c) * (y == c)).sum()

                    lap_results[epoch] = {
                        f"accuracy": correct / total,
                        f"class_accuracy": (class_correct / class_total).tolist(),
                        "source_weights": source_loss_weighting.get_source_unrelaibility(),
                        "source_values": source_loss_weighting.get_source_order(),
                        "corrupt_sources": corrupt_sources,
                    }

                print(f"Epoch {epoch}: {correct/total}")
                print(source_loss_weighting.get_source_unrelaibility())
                print(source_loss_weighting.get_source_order())

            results_lap["mnist_and_cifar"][c_type][run][f"lap_{l}_test"] = lap_results

        # save results
        with open(SAVE_DIR / f"results_standard_{c_type}.json", "w") as f:
            json.dump(results_standard, f)

        with open(SAVE_DIR / f"results_lap_{c_type}.json", "w") as f:
            json.dump(results_lap, f)
