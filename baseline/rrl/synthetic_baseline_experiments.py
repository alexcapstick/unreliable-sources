import numpy as np
import numpy as np
import sys
import yaml
import json
import tqdm
from catalyst.metrics import AccuracyMetric
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.transforms import v2 as v2_transforms
import torch.utils.data as torchdata
import faiss

sys.path.append("../../")

from experiment_code.data_utils.dataloader_loaders import (
    get_train_data,
    get_test_data,
)
from experiment_code.utils.utils import ArgFake

CONFIG_FILE = "../../synthetic_config.yaml"
DATA_DIR = "../../data/"
DATASET_NAMES = ["cifar10"]
CORRUPTION_TYPES = ["no_c", "c_cs", "c_rl", "c_lbf", "c_ns", "c_lbs", "c_no"][::-1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS = 5
RESULTS_FILE = "../../outputs/presnet_results/baseline/rrl/results.json"
exp_seed = 2

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
    print("loaded previous results")
    for ds_name in result_loaded.keys():
        if ds_name not in results.keys():
            results[ds_name] = {}
        for c_type in result_loaded[ds_name].keys():
            if c_type not in results[ds_name].keys():
                results[ds_name][c_type] = {}
            for run in result_loaded[ds_name][c_type].keys():
                if run not in results[ds_name][c_type].keys():
                    results[ds_name][c_type][run] = result_loaded[ds_name][c_type][run]
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


## augmentation dataset wrapper


class AugmentDataset(torchdata.Dataset):
    def __init__(self, dataset: torchdata.Dataset):
        self.dataset = dataset
        self.augment = v2_transforms.AugMix(
            severity=1,
            alpha=1.0,
            mixture_width=3,
            chain_depth=-1,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img_orig, target, source = self.dataset[index]

        img_min, img_max = img_orig.min(), img_orig.max()
        img_norm = (img_orig - img_min) / (img_max - img_min)

        img_unit8 = (img_norm * 255).to(torch.uint8)
        img_aug = self.augment(img_unit8)
        img_aug = (img_aug / 255) * (
            img_max - img_min
        ) + img_min  # rescale to original range

        return img_orig, target, index, img_aug, source

    def __len__(self):
        return len(self.dataset)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


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
    def __init__(self, block, num_blocks, num_classes=10, low_dim=128):
        super(PreResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)
        self.recon = nn.Linear(low_dim, 512 * block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5, do_recon=False, has_out=True):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            feat = out.view(out.size(0), -1)
            feat_lowD = self.fc(feat)
            if has_out:
                logits = self.classifier(feat)
            if do_recon:
                recon = F.relu(self.recon(feat_lowD))
                error = torch.mean((recon - feat) ** 2, dim=1)
                feat_lowD = self.l2norm(feat_lowD)
                if has_out:
                    return logits, feat_lowD, error
                else:
                    return feat_lowD, error
            else:
                feat_lowD = self.l2norm(feat_lowD)
                if has_out:
                    return logits, feat_lowD
                else:
                    return feat_lowD


def PreResNet18(num_class=10, low_dim=20):
    return PreResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_class, low_dim=low_dim)


def PreResNet34(num_class=10, low_dim=20):
    return PreResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_class, low_dim=low_dim)


def get_model(args):
    ## model initialization for different datasets
    if args.dataset_name == "cifar10":

        model = PreResNet18(
            num_class=10,
            low_dim=50,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.02,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        )

        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError("Dataset not supported")

    return model, optimizer, criterion


class ClassErrorMeter(AccuracyMetric):
    def __init__(self, topk=(1,), accuracy=True):
        assert accuracy, "ClassErrorMeter works only with accuracy=True"
        super().__init__(topk=topk)

    def add(self, output, targets):
        super().update(output, targets)
        return

    def value(self):
        return super().compute()[0]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if batch_size == 0:
        return [torch.tensor(0.0)] * len(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# from utils.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, values):
        assert isinstance(values, dict)
        for key, val in values.items():
            if not (key in self.values):
                self.values[key] = AverageMeter()
            self.values[key].update(val)

    def average(self):
        average = {}
        for key, val in self.values.items():
            average[key] = val.avg
        return average

    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()


# model


class Model(object):
    def __init__(self, args, config):
        self.opt = config
        self.args = args
        self.curr_epoch = 0
        self.acc_meter = ClassErrorMeter(topk=[1, 5], accuracy=True)
        self.temperature = self.opt["data_train_opt"]["temperature"]
        self.alpha = self.opt["data_train_opt"]["alpha"]
        self.w_inst = self.opt["data_train_opt"]["w_inst"]
        self.w_recon = self.opt["data_train_opt"]["w_recon"]
        self.model, self.optimizer, self.criterion = get_model(self.args)
        self.optimizers = {"model": self.optimizer}
        self.networks = {"model": self.model}
        self.networks["model"].to(DEVICE)
        self.criterion_instance = nn.CrossEntropyLoss()
        self.results = {
            "num_clean": [],
            "knn_acc": [],
            "test_acc": [],
        }

    def train_step(self, batch, warmup):
        if (
            self.opt["knn"]
            and self.curr_epoch >= self.opt["knn_start_epoch"]
            and not warmup
        ):
            return self.train_pseudo(batch)
        else:
            return self.train(batch, warmup=warmup)

    def train_naive(self, batch):
        data = batch[0].to(DEVICE)
        target = batch[1].to(DEVICE)
        record = {}

        output, _ = self.networks["model"](data)
        loss = self.criterion(output, target)
        record["loss"] = loss.item()
        record["train_accuracy"] = accuracy(output, target)[0].item()

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        return record

    def train(self, batch, warmup=True):
        data = batch[0].to(DEVICE)
        target = batch[1].to(DEVICE)
        batch_size = data.size(0)
        record = {}

        output, feat, error_recon = self.networks["model"](data, do_recon=True)
        loss = self.criterion(output, target)
        record["loss"] = loss.item()
        record["train_accuracy"] = accuracy(output, target)[0].item()

        loss_recon = error_recon.mean()
        loss += self.w_recon * loss_recon
        record["loss_recon"] = loss_recon.item()

        if not warmup:
            data_aug = batch[3].to(DEVICE)

            shuffle_idx = torch.randperm(batch_size)
            mapping = {k: v for (v, k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])
            feat_aug, error_recon_aug = self.networks["model"](
                data_aug[shuffle_idx], do_recon=True, has_out=False
            )
            feat_aug = feat_aug[reverse_idx]

            ##**************Reconstruction loss****************
            loss_recon_aug = error_recon_aug.mean()
            loss += self.w_recon * loss_recon_aug
            record["loss_recon"] += loss_recon_aug.item()

            ##**************Instance contrastive loss****************
            sim_clean = torch.mm(feat, feat.t())
            mask = (
                torch.ones_like(sim_clean)
                - torch.eye(batch_size, device=sim_clean.device)
            ).bool()
            sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

            sim_aug = torch.mm(feat, feat_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

            logits_pos = torch.bmm(
                feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)
            ).squeeze(-1)
            logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

            logits = torch.cat([logits_pos, logits_neg], dim=1)
            instance_labels = torch.zeros(batch_size).long().to(DEVICE)

            loss_instance = self.criterion_instance(
                logits / self.temperature, instance_labels
            )
            loss += self.w_inst * loss_instance
            record["loss_inst"] = loss_instance.item()
            record["acc_inst"] = accuracy(logits, instance_labels)[0].item()

            ##**************Mixup Prototypical contrastive loss****************
            L = np.random.beta(self.alpha, self.alpha)
            labels = (
                torch.zeros(batch_size, self.opt["data_train_opt"]["num_class"])
                .to(DEVICE)
                .scatter_(1, target.view(-1, 1), 1)
            )

            inputs = torch.cat([data, data_aug], dim=0)
            idx = torch.randperm(batch_size * 2)
            labels = torch.cat([labels, labels], dim=0)

            input_mix = L * inputs + (1 - L) * inputs[idx]
            labels_mix = L * labels + (1 - L) * labels[idx]

            feat_mix = self.networks["model"](input_mix, has_out=False)

            logits_proto = torch.mm(feat_mix, self.prototypes.t()) / self.temperature
            loss_proto = -torch.mean(
                torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1)
            )
            record["loss_proto"] = loss_proto.item()
            loss += self.w_proto * loss_proto

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()

        return record

    def train_pseudo(self, batch):
        data = batch[0].to(DEVICE)
        data_aug = batch[3].to(DEVICE)
        index = batch[2]
        batch_size = data.size(0)
        target = self.hard_labels[index].to(DEVICE)
        clean_idx = self.clean_idx[index]

        record = {}

        output, feat, error_recon = self.networks["model"](data, do_recon=True)

        loss = self.criterion(output[clean_idx], target[clean_idx])
        record["loss"] = loss.item()
        record["train_accuracy"] = accuracy(output[clean_idx], target[clean_idx])[
            0
        ].item()

        shuffle_idx = torch.randperm(batch_size)
        mapping = {k: v for (v, k) in enumerate(shuffle_idx)}
        reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])
        feat_aug, error_recon_aug = self.networks["model"](
            data_aug[shuffle_idx], do_recon=True, has_out=False
        )
        feat_aug = feat_aug[reverse_idx]

        ##**************Recon loss****************
        loss_recon = error_recon.mean() + error_recon_aug.mean()
        loss += self.w_recon * loss_recon
        record["loss_recon"] = loss_recon.item()

        ##**************Instance contrastive loss****************
        sim_clean = torch.mm(feat, feat.t())
        mask = (
            torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)
        ).bool()
        sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

        sim_aug = torch.mm(feat, feat_aug.t())
        sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

        logits_pos = torch.bmm(
            feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)
        ).squeeze(-1)
        logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

        logits = torch.cat([logits_pos, logits_neg], dim=1)
        instance_labels = torch.zeros(batch_size).long().to(DEVICE)

        loss_instance = self.criterion_instance(
            logits / self.temperature, instance_labels
        )
        loss += self.w_inst * loss_instance
        record["loss_inst"] = loss_instance.item()
        record["acc_inst"] = accuracy(logits, instance_labels)[0].item()

        ##**************Mixup Prototypical contrastive loss****************
        L = np.random.beta(self.alpha, self.alpha)

        labels = (
            torch.zeros(batch_size, self.opt["data_train_opt"]["num_class"])
            .to(DEVICE)
            .scatter_(1, target.view(-1, 1), 1)
        )
        labels = labels[clean_idx]

        inputs = torch.cat([data[clean_idx], data_aug[clean_idx]], dim=0)
        idx = torch.randperm(clean_idx.sum() * 2)
        labels = torch.cat([labels, labels], dim=0)

        input_mix = L * inputs + (1 - L) * inputs[idx]
        labels_mix = L * labels + (1 - L) * labels[idx]

        feat_mix = self.networks["model"](input_mix, has_out=False)

        logits_proto = torch.mm(feat_mix, self.prototypes.t()) / self.temperature
        loss_proto = -torch.mean(
            torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1)
        )
        record["loss_proto"] = loss_proto.item()
        loss += self.w_proto * loss_proto

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()

        return record

    def label_clean(
        self,
        features,
        labels,
        probs,
    ):
        # initalize knn search
        N = features.shape[0]
        k = self.opt["n_neighbors"]
        index = faiss.IndexFlatIP(features.shape[1])

        index.add(features)
        D, I = index.search(features, k + 1)
        neighbors = torch.LongTensor(I)  # find k nearest neighbors excluding itself

        score = torch.zeros(
            N, self.opt["data_train_opt"]["num_class"]
        )  # holds the score from weighted-knn
        weights = torch.exp(
            torch.Tensor(D[:, 1:]) / self.temperature
        )  # weight is calculated by embeddings' similarity
        for n in range(N):
            neighbor_labels = self.soft_labels[neighbors[n, 1:]]
            score[n] = (neighbor_labels * weights[n].unsqueeze(-1)).sum(
                0
            )  # aggregate soft labels from neighbors
        self.soft_labels = (
            score / score.sum(1).unsqueeze(-1) + probs
        ) / 2  # combine with model's prediction as the new soft labels

        # consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
        gt_score = self.soft_labels[labels >= 0, labels]
        gt_clean = gt_score > self.opt["low_th"]
        self.soft_labels[gt_clean] = torch.zeros(
            gt_clean.sum(), self.opt["data_train_opt"]["num_class"]
        ).scatter_(1, labels[gt_clean].view(-1, 1), 1)

        # get the hard pseudo label and the clean subset used to calculate supervised loss
        max_score, self.hard_labels = torch.max(self.soft_labels, 1)
        self.clean_idx = max_score > self.opt["high_th"]
        self.results["num_clean"].append(
            {
                "value": self.clean_idx.sum().item(),
                "epoch": "self.curr_epoch",
            }
        )

        return

    def run_train_warmup(self, data_loader, epoch):
        self.networks["model"].train()
        disp_step = self.opt["disp_step"] if ("disp_step" in self.opt) else 50
        train_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm.tqdm(data_loader)):
            if idx > self.opt["data_train_opt"]["warmup_iters"]:
                break
            train_stats_this = self.train_step(batch, True)
            train_stats.update(train_stats_this)
        return train_stats.average()

    def run_train_epoch(self, data_loader, epoch):
        self.networks["model"].train()
        disp_step = self.opt["disp_step"] if ("disp_step" in self.opt) else 50
        train_stats = DAverageMeter()

        self.scheduler.step()
        for param_group in self.optimizers["model"].param_groups:
            lr = param_group["lr"]
            break
        for idx, batch in enumerate(
            tqdm.tqdm(data_loader, desc=f"Training Epoch {epoch+1}")
        ):
            train_stats_this = self.train_step(batch, False)
            train_stats.update(train_stats_this)
        return train_stats.average()

    def compute_features(self, dataloader, model, N):
        print("Compute features")
        model.eval()
        batch_size = dataloader.batch_size
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                inputs = batch[0].to(DEVICE)
                output, feat = model(inputs)
                feat = feat.data.cpu().numpy()
                prob = F.softmax(output, dim=1)
                prob = prob.data.cpu()
            if i == 0:
                features = np.zeros((N, feat.shape[1]), dtype="float32")
                labels = torch.zeros(N, dtype=torch.long)
                probs = torch.zeros(N, self.opt["data_train_opt"]["num_class"])
            if i < len(dataloader) - 1:
                features[i * batch_size : (i + 1) * batch_size] = feat
                labels[i * batch_size : (i + 1) * batch_size] = batch[1]
                probs[i * batch_size : (i + 1) * batch_size] = prob
            else:
                # special treatment for final batch
                features[i * batch_size :] = feat
                labels[i * batch_size :] = batch[1]
                probs[i * batch_size :] = prob
        return features, labels, probs

    def test_knn(self, model, test_loader, features):
        k = self.opt["n_neighbors"]
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        with torch.no_grad():
            self.acc_meter.reset()
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(DEVICE)
                outputs, test_feat = model(inputs)
                batch_size = inputs.size(0)
                dist = np.zeros((batch_size, k))
                neighbors = np.zeros((batch_size, k))
                D, I = index.search(test_feat.data.cpu().numpy(), k)
                neighbors = torch.LongTensor(I)
                weights = torch.exp(torch.Tensor(D) / self.temperature).unsqueeze(-1)
                score = torch.zeros(batch_size, self.opt["data_train_opt"]["num_class"])
                for n in range(batch_size):
                    neighbor_labels = self.soft_labels[neighbors[n]]
                    score[n] = (neighbor_labels * weights[n]).sum(0)
                self.acc_meter.add(score, targets)
            accuracy = self.acc_meter.value()
            self.results["knn_acc"].append(
                {
                    "value": {"top1": accuracy[0], "top5": accuracy[1]},
                    "epoch": self.curr_epoch - 1,
                }
            )
        return

    def test(self, model, test_loader):
        with torch.no_grad():
            self.acc_meter.reset()
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs, _ = model(inputs)
                self.acc_meter.add(outputs, targets)
            accuracy = self.acc_meter.value()
            self.results["test_acc"].append(
                {
                    "value": {"top1": accuracy[0], "top5": accuracy[1]},
                    "epoch": self.curr_epoch - 1,
                }
            )
            print("Test accuracy: ", accuracy)
        return

    def solve(self, data_loader_train, data_loader_eval, test_loader):
        tqdm.tqdm._instances.clear()
        self.max_num_epochs = self.opt["max_num_epochs"]
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizers["model"],
            T_max=self.max_num_epochs - start_epoch,
            eta_min=0.0002,
        )
        # **********************************************************
        if self.curr_epoch == 0:
            self.run_train_warmup(
                data_loader_train, self.curr_epoch
            )  # warm-up for several iterations to initalize prototypes

        train_stats = {}
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            features, labels, probs = self.compute_features(
                data_loader_eval, self.networks["model"], len(data_loader_eval.dataset)
            )

            old_prototypes = (
                self.prototypes.clone().cpu() if self.curr_epoch > 0 else None
            )
            self.prototypes = []
            if self.opt["knn"] and self.curr_epoch >= self.opt["knn_start_epoch"]:
                if self.curr_epoch == self.opt["knn_start_epoch"]:
                    # initalize the soft label as model's softmax prediction
                    gt_score = probs[labels >= 0, labels]
                    gt_clean = gt_score > self.opt["low_th"]
                    self.soft_labels = probs.clone()
                    self.soft_labels[gt_clean] = torch.zeros(
                        gt_clean.sum(), self.opt["data_train_opt"]["num_class"]
                    ).scatter_(1, labels[gt_clean].view(-1, 1), 1)

                self.label_clean(features, labels, probs)
                self.test_knn(self.networks["model"], test_loader, features)

                features = features[self.clean_idx]
                pseudo_labels = self.hard_labels[self.clean_idx]
                for c in range(self.opt["data_train_opt"]["num_class"]):
                    if len(features[np.where(pseudo_labels.numpy() == c)]) == 0:
                        print(
                            f"prototype not found for class {c}, using old prototype instead"
                        )
                        self.prototypes.append(old_prototypes[c].clone())
                        # if no clean sample is assigned to this class, use identity vector as prototype
                        # prototype = torch.zeros(features.shape[1])
                        # prototype[c] = 1.0
                    else:
                        prototype = features[np.where(pseudo_labels.numpy() == c)].mean(
                            0
                        )  # compute prototypes with pseudo-label
                        self.prototypes.append(torch.Tensor(prototype))
            else:
                for c in range(self.opt["data_train_opt"]["num_class"]):
                    if len(features[np.where(labels.numpy() == c)]) == 0:
                        print(
                            f"prototype not found for class {c}, using old prototype instead"
                        )
                        self.prototypes.append(old_prototypes[c].clone())
                        # if no clean sample is assigned to this class, use identity vector as prototype
                        # prototype = torch.zeros(features.shape[1])
                        # prototype[c] = 1.0
                    else:
                        prototype = features[np.where(labels.numpy() == c)].mean(
                            0
                        )  # compute prototypes as mean embeddings
                        self.prototypes.append(torch.Tensor(prototype))
            self.prototypes = torch.stack(self.prototypes).to(DEVICE)
            self.prototypes = F.normalize(
                self.prototypes, p=2, dim=1
            )  # normalize the prototypes

            if self.opt["data_train_opt"][
                "ramp_epoch"
            ]:  # ramp up the weights for prototypical loss (optional)
                self.w_proto = min(
                    1
                    + self.curr_epoch
                    * (self.opt["data_train_opt"]["w_proto"] - 1)
                    / self.opt["data_train_opt"]["ramp_epoch"],
                    self.opt["data_train_opt"]["w_proto"],
                )
            else:
                self.w_proto = self.opt["data_train_opt"]["w_proto"]

            # perform training for 1 epoch
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            for k, v in train_stats.items():
                if k not in self.results:
                    self.results[k] = []
                self.results[k].append({"value": v, "epoch": self.curr_epoch})

            self.test(self.networks["model"], test_loader)


for dataset_name in DATASET_NAMES:
    for corruption_type in CORRUPTION_TYPES:
        print(
            f"for dataset {dataset_name} and corruption {corruption_type} the following runs have been completed",
            [int(k) for k in results[dataset_name][corruption_type].keys()],
        )

exp_number = 0
for run in range(N_RUNS):
    for dataset_name in DATASET_NAMES:
        for corruption_type in CORRUPTION_TYPES:
            exp_number += 1
            exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))

            if run in [int(k) for k in results[dataset_name][corruption_type].keys()]:
                print(
                    "Skipping the following experiment as already completed:",
                    dataset_name,
                    corruption_type,
                    run,
                    "...",
                )
                continue

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
            print("Run:", run)

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
            args.lr = 0.02
            args.n_epochs = 25

            training_params = yaml.load(
                open(args.config_file, "r"), Loader=yaml.FullLoader
            )[f"Conv3Net-{args.corruption_type}-drstd"]["train_params"]

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
                        inputs, targets = outputs[:2]
                        sources = outputs[-1]
                        if BATCH_FLIPPING:
                            targets, sources = batch_label_flipping(
                                targets, sources, CORRUPT_SOURCES
                            )
                        elif BATCH_SHUFFLING:
                            targets, sources = batch_label_shuffle(
                                targets, sources, CORRUPT_SOURCES
                            )
                        yield inputs, targets

                def __len__(self):
                    return len(self.dl)

            baseline_train_loader = torchdata.DataLoader(
                dataset=AugmentDataset(ToMemory(train_loader.dataset)),
                batch_size=128,
                shuffle=True,
            )
            baseline_train_loader = BatchFlipShuffleDL(baseline_train_loader)

            baseline_eval_loader = torchdata.DataLoader(
                dataset=ToMemory(train_loader.dataset),
                batch_size=128,
                shuffle=False,
            )

            baseline_eval_loader = BatchFlipShuffleDL(baseline_eval_loader)

            baseline_test_loader = torchdata.DataLoader(
                dataset=ToMemory(test_loader.dataset),
                batch_size=128,
                shuffle=False,
            )

            baseline_config = {}

            num_classes = 10
            ramp_epoch = 0
            knn_start_epoch = 5
            low_th = 0.1
            high_th = 0.9

            data_train_opt = {}
            data_train_opt["batch_size"] = training_params["source_size"]
            data_train_opt["temperature"] = 0.3
            data_train_opt["num_class"] = num_classes
            data_train_opt["alpha"] = 8
            data_train_opt["w_inst"] = 1
            data_train_opt["w_proto"] = 5
            data_train_opt["w_recon"] = 1
            data_train_opt["low_dim"] = 50
            data_train_opt["warmup_iters"] = 100
            data_train_opt["ramp_epoch"] = ramp_epoch

            baseline_config["data_train_opt"] = data_train_opt
            baseline_config["max_num_epochs"] = args.n_epochs
            baseline_config["test_knn"] = True
            baseline_config["knn_start_epoch"] = knn_start_epoch
            baseline_config["knn"] = True
            baseline_config["n_neighbors"] = 200
            baseline_config["low_th"] = low_th
            baseline_config["high_th"] = high_th

            temperature = baseline_config["data_train_opt"]["temperature"]
            alpha = baseline_config["data_train_opt"]["alpha"]
            w_inst = baseline_config["data_train_opt"]["w_inst"]
            w_recon = baseline_config["data_train_opt"]["w_recon"]

            m = Model(args, baseline_config)
            m.solve(baseline_train_loader, baseline_eval_loader, baseline_test_loader)

            results[dataset_name][corruption_type][run] = m.results

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)
