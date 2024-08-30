import os
from pathlib import Path
import argparse
import numpy as np
import yaml
import tqdm
import math
import time
import torch
import faiss
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.transforms.v2 as transforms
from catalyst.metrics import AccuracyMetric
from torch.utils.tensorboard import SummaryWriter
import json
from collections import OrderedDict

from experiment_code.utils.utils import ArgFake
from experiment_code.data_utils.dataloader_loaders import get_train_data, get_test_data

from loss_adapted_plasticity import SourceLossWeighting

parser = argparse.ArgumentParser()
parser.add_argument("--corruption_type", type=str, default="c_lbf")
parser.add_argument("--corruption_level", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="./outputs/cifar_different_noise_results/",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)

# --- experiment options
DATASET_NAMES = ["cifar10"]  # only cifar10 is implemented
CORRUPTION_TYPE = args.corruption_type
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "./data/"
TEST_DIR = args.test_dir
DEPRESSION = [True, False]
CORRUPTION_LEVEL = args.corruption_level
N_CORRUPT_SOURCES = [2, 4, 6]
CONFIG_FILE = "./synthetic_config.yaml"


# --- model options
LAP_HISTORY_LENGTH = 25
DEPRESSION_STRENGTH = 1.0
LENIENCY = 0.8
HOLD_OFF = 0
DISCRETE_AMOUNT = 0.005
WARMUP_ITERS = 0

# --- data options
BATCH_SIZE = 128

RESULTS_FILE = os.path.join(
    TEST_DIR,
    f"results_presnet_{CORRUPTION_TYPE}_{CORRUPTION_LEVEL}_{''.join([str(r) for r in RUNS])}.json",
)


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


exp_seed = args.seed

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
    def __init__(self, dataset: torchdata.Dataset):
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

    def __getitem__(self, index):
        img_orig, target, source = self.dataset[index]

        img_min, img_max = img_orig.min(), img_orig.max()
        img_norm = (img_orig - img_min) / (img_max - img_min)

        img_unit8 = (img_norm * 255).to(torch.uint8)
        img_aug = self.augment(img_unit8)
        img_aug = (img_aug / 255) * (
            img_max - img_min
        ) + img_min  # rescale to original range

        return img_orig, target, img_aug, index, source

    def __len__(self):
        return len(self.dataset)


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

                if CORRUPTION_TYPE == "c_lbf":
                    BATCH_FLIPPING = True
                    BATCH_SHUFFLING = False
                    CORRUPTION_TYPE = "no_c"
                elif CORRUPTION_TYPE == "c_lbs":
                    BATCH_FLIPPING = False
                    BATCH_SHUFFLING = True
                    CORRUPTION_TYPE = "no_c"
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
                        "corruption_type": CORRUPTION_TYPE,
                        "n_corrupt_sources": nc,
                        "corruption_level": CORRUPTION_LEVEL,
                    }
                )

                # dataset dependent args

                if args.dataset_name == "cifar10":
                    args.n_epochs = 25

                ## load data config files for different datasets and corruption types

                if args.dataset_name == "cifar10":
                    training_params = yaml.load(
                        open(args.config_file, "r"), Loader=yaml.FullLoader
                    )[f"Conv3Net-{args.corruption_type}-drstd"]["train_params"]

                if BATCH_FLIPPING or BATCH_SHUFFLING:
                    if BATCH_FLIPPING:
                        corruption_type_to_load_n_sources = "c_lbf"
                        CORRUPTION_TYPE = "c_lbf"
                    elif BATCH_SHUFFLING:
                        corruption_type_to_load_n_sources = "c_lbs"
                        CORRUPTION_TYPE = "c_lbs"

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

                training_params["return_sources"] = True
                training_params["corruption_level"] = [
                    args.corruption_level
                ] * args.n_corrupt_sources
                training_params["n_corrupt_sources"] = args.n_corrupt_sources

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

                dataset_memory = ToMemory(train_loader.dataset)

                aug_train_loader = torchdata.DataLoader(
                    dataset=AugmentDataset(
                        dataset_memory,
                    ),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )

                train_cf_loader = torchdata.DataLoader(
                    dataset=dataset_memory,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                test_loader = torchdata.DataLoader(
                    dataset=ToMemory(test_loader.dataset),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                model = PreResNet18(
                    num_class=10,
                    low_dim=50,
                )

                optimiser = torch.optim.SGD(
                    params=model.parameters(),
                    lr=0.02,
                    momentum=0.9,
                    weight_decay=5e-4,
                    nesterov=False,
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser,
                    T_max=args.n_epochs,
                    eta_min=0.0002,
                )

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

                input_loss_weighting = None

                warmup_iters = 100
                knn_start_epoch = 5

                # writer = SummaryWriter(
                #     log_dir=os.path.join(
                #         TEST_DIR,
                #         "tb",
                #         CORRUPTION_TYPE,
                #         args.dataset_name,
                #         f"{run}",
                #         f"{depression}",
                #         f"{time.time().__str__().replace('.', '')}",
                #     )
                # )
                writer = None

                # warmup
                pbar = tqdm.tqdm(aug_train_loader, desc="Warm-Up")
                train_loss = 0
                train_total = 0

                if BATCH_FLIPPING or BATCH_SHUFFLING:
                    corrupt_sources = CORRUPT_SOURCES
                else:
                    corrupt_sources = (
                        aug_train_loader.dataset.dataset.dataset.corrupt_sources.tolist()
                    )

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

                    if BATCH_FLIPPING:
                        targets, sources = batch_label_flipping(
                            targets, sources, CORRUPT_SOURCES
                        )
                    if BATCH_SHUFFLING:
                        targets, sources = batch_label_shuffle(
                            targets, sources, CORRUPT_SOURCES
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

                for epoch in range(args.n_epochs):
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

                if writer is not None:

                    writer.flush()
                    writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
