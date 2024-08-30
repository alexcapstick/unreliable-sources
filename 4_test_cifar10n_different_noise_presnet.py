import numpy as np
from pathlib import Path
import tqdm
import time
import torch
import faiss
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
    default="./outputs/cifar_10n_different_noise_results/",
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
N_CORRUPT_SOURCES = [2, 3, 4, 5, 6]
MIN_CORRUPTION_LEVEL = args.min_corruption_level
MAX_CORRUPTION_LEVEL = args.max_corruption_level
BATCH_SIZE = 128
N_EPOCHS = 25
LR = 0.02
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


## augmentation dataset wrapper
class AugmentDataset(torchdata.Dataset):
    def __init__(self, dataset: torchdata.Dataset, fmnist=False):
        self.dataset = dataset
        self.augment = transforms.AugMix(
            severity=1,
            alpha=1.0,
            mixture_width=3,
            chain_depth=-1,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        self.to_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        self.fmnist = fmnist

    def __getitem__(self, index):
        img_orig, target, source = self.dataset[index]
        if self.fmnist:
            img_orig = img_orig.reshape(28, 28)

        img_min, img_max = img_orig.min(), img_orig.max()
        img_norm = (img_orig - img_min) / (img_max - img_min)

        img_unit8 = (img_norm * 255).to(torch.uint8)
        img_aug = self.augment(img_unit8)
        img_aug = (img_aug / 255) * (
            img_max - img_min
        ) + img_min  # rescale to original range

        if self.fmnist:
            img_orig = img_orig.reshape(-1)
            img_aug = img_aug.reshape(-1)

        return img_orig, target, img_aug, index, source

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


# presnet architecture


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
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


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
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
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        low_dim=50,
        w_recon=1,
        w_instance=1,
        w_proto=5,
        temperature=0.3,
        beta_alpha=8,
        low_th=0.1,
        high_th=0.9,
        n_neighbours=200,
    ):
        super(PreResNet, self).__init__()

        self.in_planes = 64
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes

        self.net = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(4),
            nn.Flatten(start_dim=1),
        )

        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)
        self.recon = nn.Linear(low_dim, 512 * block.expansion)

        self.w_recon = w_recon
        self.w_instance = w_instance
        self.w_proto = w_proto
        self.temperature = temperature
        self.beta_alpha = beta_alpha
        self.low_th = low_th
        self.high_th = high_th
        self.n_neighbours = n_neighbours
        self.criterion_instance = nn.CrossEntropyLoss(reduction="none")

        # PCL on lower dimensional feature space
        self.reset_prototypes()
        self.soft_labels = None
        self.hard_labels = None
        self.clean_idx = None

    def reset_prototypes(self):
        self.prototypes = []
        self.features = []
        self.labels = []
        self.probs = []

    def compute_features(self, x, y):
        if self.training:
            raise ValueError("model is in training mode and compute_features is called")
        with torch.no_grad():
            feature = self.net(x)
            feature_low_dim = self.fc(feature)
            feature_low_dim = self.l2norm(feature_low_dim)

            self.features.append(feature_low_dim)
            self.labels.append(y)
            self.probs.append(F.softmax(self.classifier(feature), dim=1))
        return

    def _ensure_features_tensor(self):
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.cat(self.features, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
            self.probs = torch.cat(self.probs, dim=0)
        return

    def _label_clean(self):
        if self.soft_labels is None:
            gt_score = self.probs[self.labels >= 0, self.labels]
            gt_clean = gt_score > self.low_th
            self.soft_labels = self.probs.clone()
            self.soft_labels[gt_clean] = torch.zeros(
                gt_clean.sum(), self.num_classes, device=gt_clean.device
            ).scatter_(1, self.labels[gt_clean].view(-1, 1), 1)

        N = self.features.shape[0]
        k = self.n_neighbours
        index = faiss.IndexFlatIP(self.features.shape[1])

        index.add(self.features.cpu().numpy())
        D, I = index.search(self.features.cpu().numpy(), k + 1)
        # find k nearest neighbors excluding itself
        neighbours = torch.from_numpy(I).to(self.features.device).long()

        score = torch.zeros(
            N, self.num_classes, device=self.features.device
        )  # holds the score from weighted-knn
        weights = torch.exp(
            torch.tensor(D[:, 1:], device=self.features.device) / self.temperature
        )  # weight is calculated by embeddings' similarity

        for n in range(N):
            neighbour_labels = self.soft_labels[neighbours[n, 1:]]
            score[n] = (neighbour_labels * weights[n].unsqueeze(-1)).sum(
                0
            )  # aggregate soft labels from neighbors

        self.soft_labels = (
            score / score.sum(1).unsqueeze(-1) + self.probs
        ) / 2  # combine with model's prediction as the new soft labels

        # consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
        gt_score = self.soft_labels[self.labels >= 0, self.labels]
        gt_clean = gt_score > self.low_th
        self.soft_labels[gt_clean] = torch.zeros(
            gt_clean.sum(), self.num_classes, device=gt_clean.device
        ).scatter_(1, self.labels[gt_clean].view(-1, 1), 1)

        # get the hard pseudo label and the clean subset used to calculate supervised loss
        max_score, self.hard_labels = torch.max(self.soft_labels, 1)
        self.clean_idx = max_score > self.high_th

        return

    def update_prototypes(self, knn=False):
        if self.training:
            raise ValueError(
                "model is in training mode and update_prototypes is called"
            )
        self._ensure_features_tensor()
        with torch.no_grad():
            prototypes = []

            if knn:
                self._label_clean()
                features = self.features[self.clean_idx]
                labels = self.hard_labels[self.clean_idx]
            else:
                features = self.features
                labels = self.labels

            for c in range(self.num_classes):
                prototypes.append(torch.mean(features[labels == c], axis=0))
            self.prototypes = torch.stack(prototypes, dim=0)
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        return

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _reconstruction_error(self, feature, feature_low_dim):
        recon = F.relu(self.recon(feature_low_dim))
        error = torch.mean((recon - feature) ** 2, dim=1)
        return error

    def forward(
        self,
        x,
        y=None,
        x_aug=None,
        index=None,
        return_loss=False,
        warmup=True,
        knn=False,
    ):
        feature = self.net(x)

        # return logits
        out = self.classifier(feature)

        if return_loss:
            if y is None:
                raise ValueError("y is None and return_loss is True")

            batch_size = len(x)

            feature_low_dim = self.fc(feature)

            ##### label loss #####
            if knn:
                label_loss = torch.zeros_like(y, dtype=torch.float32)
                clean_idx = self.clean_idx[index]
                target = self.hard_labels[index]
                label_loss[clean_idx] = self.criterion(
                    out[clean_idx], target[clean_idx]
                )
            else:
                label_loss = self.criterion(out, y)

            ##### reconstruction loss #####
            recon_loss = self._reconstruction_error(feature, feature_low_dim)

            input_loss = recon_loss

            if not warmup:
                if x_aug is None:
                    raise ValueError("x_aug is None and warmup is False")

                ##### augmentation reconstruction loss #####

                feature_aug = self.net(x_aug)
                feature_aug_low_dim = self.fc(feature)
                recon_aug_loss = self._reconstruction_error(
                    feature_aug, feature_aug_low_dim
                )
                feature_aug = feature_aug

                input_loss += recon_aug_loss

                ##### instance contrastive loss #####

                feature_low_dim = self.l2norm(feature_low_dim)
                feature_aug_low_dim = self.l2norm(feature_aug_low_dim)

                sim_clean = torch.mm(feature_low_dim, feature_low_dim.t())
                sim_aug = torch.mm(feature_low_dim, feature_aug_low_dim.t())

                mask = (
                    torch.ones_like(sim_clean)
                    - torch.eye(batch_size, device=sim_clean.device)
                ).bool()

                sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)
                sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

                logits_pos = torch.bmm(
                    feature_low_dim.view(batch_size, 1, -1),
                    feature_aug_low_dim.view(batch_size, -1, 1),
                ).squeeze(-1)
                logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

                logits = torch.cat([logits_pos, logits_neg], dim=1)
                instance_labels = torch.zeros(batch_size, device=logits.device).long()

                loss_instance = self.criterion_instance(
                    logits / self.temperature, instance_labels
                )

                label_loss += self.w_instance * loss_instance

                ##### mixup prototypical contrastive loss #####

                L = np.random.beta(self.beta_alpha, self.beta_alpha)
                labels = torch.zeros(
                    batch_size,
                    self.num_classes,
                    device=feature_low_dim.device,
                ).scatter_(1, y.view(-1, 1), 1)

                if knn:
                    labels = labels[clean_idx]
                    inputs = torch.cat([x[clean_idx], x_aug[clean_idx]], dim=0)
                    idx = torch.randperm(clean_idx.sum() * 2)
                else:
                    inputs = torch.cat([x, x_aug], dim=0)
                    idx = torch.randperm(batch_size * 2)

                labels = torch.cat([labels, labels], dim=0)

                input_mix = L * inputs + (1 - L) * inputs[idx]
                labels_mix = L * labels + (1 - L) * labels[idx]

                feat_mix = self.net(input_mix)
                feat_mix = self.fc(feat_mix)
                feat_mix = self.l2norm(feat_mix)

                logits_proto = (
                    torch.mm(feat_mix, self.prototypes.t()) / self.temperature
                )

                loss_proto_stack = -torch.sum(
                    F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1
                )

                if knn:
                    loss_proto = torch.zeros_like(y, dtype=torch.float32)
                    loss_proto[clean_idx] = (
                        loss_proto_stack[: clean_idx.sum()]
                        + loss_proto_stack[clean_idx.sum() :]
                    ) / 2
                else:
                    loss_proto = (
                        loss_proto_stack[:batch_size] + loss_proto_stack[batch_size:]
                    ) / 2

                label_loss += self.w_proto * loss_proto

            return label_loss, input_loss, out

        return out


def PreResNet18(num_class=10, low_dim=20, **kwargs):
    return PreResNet(
        PreActBlock, [2, 2, 2, 2], num_classes=num_class, low_dim=low_dim, **kwargs
    )


def PreResNet34(num_class=10, low_dim=20):
    return PreResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_class, low_dim=low_dim)


def train_batch(
    model,
    optimiser,
    x,
    y,
    sources,
    label_loss_weighting,
    input_loss_weighting,
    device,
    x_aug=None,
    index=None,
    warmup=True,
    knn=False,
    writer=None,
):
    model.to(device)
    model.train()

    optimiser.zero_grad()

    label_loss, input_loss, outputs = model(
        x, y, x_aug=x_aug, index=index, return_loss=True, warmup=warmup, knn=knn
    )

    if not warmup:
        if label_loss_weighting is not None:
            label_loss = label_loss_weighting(
                losses=label_loss, sources=sources, writer=writer, writer_prefix="label"
            )
        if input_loss_weighting is not None:
            input_loss = input_loss_weighting(
                losses=input_loss, sources=sources, writer=writer, writer_prefix="input"
            )

    loss = torch.mean(label_loss) + model.w_recon * torch.mean(input_loss)
    loss.backward()
    optimiser.step()

    return loss.item(), outputs


def train_epoch(
    model,
    train_loader,
    train_cf_loader,
    optimiser,
    scheduler,
    device,
    epoch_number,
    label_loss_weighting,
    input_loss_weighting,
    knn=False,
    writer=None,
):
    model.to(device)

    model.eval()
    # updating prototypes for PCL
    desc = f"Computing Features with KNN" if knn else f"Computing Features"
    pbar = tqdm.tqdm(train_cf_loader, desc=desc)
    with torch.no_grad():
        model.reset_prototypes()
        for batch_idx, (inputs, targets, sources) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            model.compute_features(inputs, targets)
        model.update_prototypes(knn=knn)

    model.train()
    scheduler.step()

    train_loss = 0
    train_total = 0
    train_acc_meter = AccuracyMetric(topk=[1, 5], num_classes=10)
    lr = optimiser.param_groups[0]["lr"]
    pbar = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch_number+1}, LR {lr:.4f}")

    for batch_idx, (inputs, targets, inputs_aug, index, sources) in enumerate(pbar):
        inputs, targets, inputs_aug, index, sources = (
            inputs.to(device),
            targets.to(device),
            inputs_aug.to(device),
            index.to(device),
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
            input_loss_weighting=input_loss_weighting,
            device=device,
            x_aug=inputs_aug,
            index=index,
            knn=knn,
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
            label_loss, input_loss, outputs = model(
                inputs, y=targets, return_loss=True, warmup=True
            )

            loss = torch.mean(label_loss) + model.w_recon * torch.mean(input_loss)
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
                    dataset=AugmentDataset(
                        train_dataset_memory,
                        fmnist=False,
                    ),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )

                train_cf_loader = torchdata.DataLoader(
                    dataset=train_dataset_memory,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                test_loader = torchdata.DataLoader(
                    dataset=test_dataset_memory,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                model = PreResNet18(
                    num_class=10,
                    low_dim=50,
                )

                optimiser = torch.optim.SGD(
                    params=model.parameters(),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=5e-4,
                    nesterov=False,
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser,
                    T_max=N_EPOCHS,
                    eta_min=0.0002,
                )

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

                input_loss_weighting = None

                warmup_iters = 100
                knn_start_epoch = 5

                writer = SummaryWriter(
                    log_dir=os.path.join(
                        TEST_DIR,
                        "tb",
                        dataset_name,
                        f"{run}",
                        f"{depression}",
                        f"{time.time().__str__().replace('.', '')}",
                    )
                )

                corrupt_sources = train_dataset.corrupt_sources.tolist()

                # warmup
                pbar = tqdm.tqdm(aug_train_loader, desc="Warm-Up")
                train_loss = 0
                train_total = 0

                results_this_train = {}

                # move data to GPU
                for _ in tqdm.tqdm(train_cf_loader, desc="Moving Data to Memory"):
                    pass

                for batch_idx, (inputs, targets, _, _, sources) in enumerate(pbar):
                    if batch_idx >= warmup_iters:
                        break

                    inputs, targets, sources = (
                        inputs.to(DEVICE),
                        targets.to(DEVICE),
                        sources.to(DEVICE),
                    )

                    loss, outputs = train_batch(
                        model,
                        optimiser,
                        inputs,
                        targets,
                        sources,
                        label_loss_weighting=label_loss_weighting,
                        input_loss_weighting=input_loss_weighting,
                        device=DEVICE,
                        warmup=True,
                        knn=False,
                        writer=writer,
                    )

                    train_loss += loss * targets.size(0)
                    train_total += targets.size(0)

                    pbar.set_postfix({"Warm-Up Loss": train_loss / train_total})
                pbar.close()

                test(model, test_loader, DEVICE)

                for epoch in range(N_EPOCHS):
                    if epoch >= knn_start_epoch:
                        knn = True
                    else:
                        knn = False
                    train_loss, train_top1acc, train_top5acc = train_epoch(
                        model=model,
                        train_loader=aug_train_loader,
                        train_cf_loader=train_cf_loader,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        device=DEVICE,
                        epoch_number=epoch,
                        label_loss_weighting=label_loss_weighting,
                        input_loss_weighting=input_loss_weighting,
                        knn=knn,
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

                writer.flush()
                writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
