import sys
import os
import math
from pathlib import Path
import argparse
import time
import json
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.utils.data as torchdata
from catalyst.metrics import AccuracyMetric

sys.path.append("../../")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--results_name", type=str, default="results_cdr.json")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="../../outputs/goemotions",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)

# --- experiment options
DATASET_NAMES = [
    "goemotions",
]
CORRUPTION_TYPE = [
    # "flipped_label",
    "random_label",
    "random_token",
    "random_permute",
    "original",
]
N_CORRUPT_SOURCES = 30
TOKEN_NOISE_AMOUNT = 0.5
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
print("Using device", DEVICE)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "../../data/"
TEST_DIR = args.test_dir
RESULTS_FILE = os.path.join(TEST_DIR, args.results_name)

# --- training options
BATCH_SIZE = 256
N_EPOCHS = 25
LR = 0.001
GRAD_CLIP = None


NUM_GRADUAL = 10
NOISE_RATE = 0.4


exp_seed = args.seed

EMBEDDING_DIM = 256  # embedding dimension
D_HIDDEN = (
    256  # dimension of the feedforward network model in ``nn.TransformerEncoder``
)
N_LAYERS = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
N_HEADS = 4  # number of heads in ``nn.MultiheadAttention``
DROPOUT = 0.1  # dropout probability
MAX_LENGTH = 128  # the max length of the sequence


SMOOTH_RATE = 0.1
WA = 0
WB = 1

print(args.results_name)

time.sleep(5)


results = {ds_name: {c: {} for c in CORRUPTION_TYPE} for ds_name in DATASET_NAMES}


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
                    results[ds_name][corruption][run_int] = result_loaded[ds_name][
                        corruption
                    ][run]

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


class GoEmotions(torch.utils.data.Dataset):
    def __init__(
        self,
        root="./data/",
        max_length=512,
    ):
        """

        The GoEmotions dataset is a dataset of Reddit comments
        that have been labelled with 28 different emotions.
        The dataset is available from the GoEmotions paper
        (https://arxiv.org/abs/2005.00547) and is available
        for download from the Google Research website.

        The dataset is split into three files, each containing
        a different set of comments. The dataset is loaded
        into memory and the comments are tokenised using the
        basic_english tokenizer from the torchtext library.

        This dataset is downloaded automatically if it is not
        found in the root directory.

        Parameters
        ----------
        root: str
            The root directory where the dataset should be stored.
            Defaults to :code:`'./data/'`.

        max_length: int
            The maximum length of the text sequence.
            Defaults to :code:`512`.

        """
        self.root = root
        self.max_length = max_length

        if not os.path.exists(os.path.join(self.root, "goemotions")):
            os.makedirs(os.path.join(self.root, "goemotions"))

        files_exist = np.all(
            [
                os.path.exists(
                    os.path.join(self.root, "goemotions", f"goemotions_{i}.csv")
                )
                for i in range(1, 4)
            ]
        )

        if not files_exist:
            download_url(
                "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
                os.path.join(self.root, "goemotions"),
            )
            download_url(
                "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
                os.path.join(self.root, "goemotions"),
            )
            download_url(
                "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
                os.path.join(self.root, "goemotions"),
            )

        self.data = pd.concat(
            [
                pd.read_csv(
                    os.path.join(self.root, "goemotions", f"goemotions_{i}.csv")
                )
                for i in range(1, 4)
            ]
        )

        self.y_unique = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        X = self.data["text"].values
        y = self.data[self.y_unique].values
        sources = self.data["rater_id"].values

        rows_with_no_labels = y.sum(axis=1) == 0
        X = X[~rows_with_no_labels]
        y = y[~rows_with_no_labels]
        sources = sources[~rows_with_no_labels]

        rows_with_too_many_labels = y.sum(axis=1) > 1
        X = X[~rows_with_too_many_labels]
        y = y[~rows_with_too_many_labels]
        sources = sources[~rows_with_too_many_labels]

        y = np.argmax(y, axis=1)

        self.inputs = X
        self.targets = y
        self.sources = sources

        tokeniser = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            map(tokeniser, X),
            specials=["<pad>", "<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])

        self.tokeniser = tokeniser
        self.vocab = vocab

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text = self.inputs[idx]

        text_tensor = (
            torch.zeros(self.max_length, dtype=torch.long) * self.vocab["<pad>"]
        )
        text_split = self.tokeniser(text)[: self.max_length]
        text_tokens = [
            torch.tensor(self.vocab(self.tokeniser(item)), dtype=torch.long)
            for item in text_split
        ]
        text_tensor_raw = torch.cat(tuple(filter(lambda t: t.numel() > 0, text_tokens)))
        text_tensor[: text_tensor_raw.size(0)] = text_tensor_raw

        label = self.targets[idx]
        source = self.sources[idx]

        return (
            text_tensor,
            label,
            source,
        )


class SourceDatasetOriginal(torchdata.Dataset):
    def __init__(self, dataset, sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        self.dataset = dataset

        self.sources = sources
        self.corrupt_sources = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, y, source = self.dataset[idx]

        # return data and source
        return x, y, source


class SourceDatasetRandomLabel(torchdata.Dataset):
    def __init__(self, dataset, targets, sources, n_corrupt_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        self.dataset = dataset
        n_sources = len(np.unique(sources))
        self.unique_targets = np.unique(targets)

        self.sources = sources
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
        x, y, source = self.dataset[idx]

        # if source is corrupt, return random label
        if source in self.corrupt_sources:
            y = self.corrupt_targets[idx]

        # return data and source
        return x, y, source


class SourceDatasetRandomPermute(torchdata.Dataset):
    def __init__(self, dataset, sources, n_corrupt_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        self.seed = self.rng.integers(1e9)

        self.dataset = dataset
        n_sources = len(np.unique(sources))

        self.sources = sources
        self.corrupt_sources = self.rng.choice(
            n_sources, n_corrupt_sources, replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, y, source = self.dataset[idx]

        if source in self.corrupt_sources:
            x = x[
                torch.randperm(
                    x.shape[0],
                    generator=torch.Generator(device=x.device).manual_seed(
                        int(self.seed + idx)
                    ),
                    device=x.device,
                )
            ]

        # return data and source
        return x, y, source


class SourceDatasetRandomToken(torchdata.Dataset):
    def __init__(
        self, dataset, sources, n_corrupt_sources, num_tokens, amount_noise, seed=None
    ):
        self.rng = np.random.default_rng(seed=seed)

        self.seed = self.rng.integers(1e9)

        self.dataset = dataset
        n_sources = len(np.unique(sources))

        self.sources = sources
        self.corrupt_sources = self.rng.choice(
            n_sources, n_corrupt_sources, replace=False
        )

        self.num_tokens = num_tokens
        self.amount_noise = amount_noise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        x, y, source = self.dataset[idx]

        if source in self.corrupt_sources:
            mask = (
                torch.rand(
                    x.shape[0],
                    device=x.device,
                    generator=torch.Generator(device=x.device).manual_seed(
                        int(self.seed + idx)
                    ),
                )
                < self.amount_noise
            )

            random_tokens = torch.randint(
                0,
                self.num_tokens,
                (x.shape[0],),
                generator=torch.Generator(device=x.device).manual_seed(
                    int(self.seed + idx)
                ),
                device=x.device,
            )

            x = torch.where(mask, random_tokens, x)

        # return data and source
        return x, y, source


class PositionalEncoding(nn.Module):

    def __init__(
        self, embedding_dim: int = 256, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_classes=2,
        ntokens=100,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        max_len=5000,
        criterion=nn.CrossEntropyLoss(reduction="none"),
    ):

        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(ntokens, d_model)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=d_model, dropout=dropout, max_len=max_len
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        self.net = nn.Sequential(
            self.embedding, self.positional_encoding, self.transformer_encoder
        )

        self.prediction_head = nn.Linear(d_model, n_classes)

        self.criterion = criterion

    def forward(self, X):

        out = X
        out = self.net(out)
        out = out.mean(dim=1)
        out = self.prediction_head(out)

        return out


class SentimentAnalysisModel(nn.Module):
    def __init__(
        self,
        ntokens,
        n_classes=2,
        hidden_size=512,
        embedding_size=256,
        n_layers=2,
        dropout=0.25,
        criterion=nn.CrossEntropyLoss(reduction="none"),
    ):
        # adapted from https://www.kaggle.com/code/affand20/imdb-with-pytorch
        super(SentimentAnalysisModel, self).__init__()

        self.embedding = nn.Embedding(ntokens, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

        self.criterion = criterion

    def forward(self, x):

        # map input to vector
        x = self.embedding(x)

        # pass forward to lstm
        out, _ = self.lstm(x)

        # get last sequence output
        out = out[:, -1, :]

        # apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)

        return out


def cdr_on_grads(net, nonzero_ratio, clip):
    """
    This is used after calculating the loss and
    calling loss.backward(). It clips the gradients.
    Then after this, optimizer.step() should be called.
    """

    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).float()
            # .type(torch.cuda.FloatTensor) ## changed so that code does not require gpu
            mask = mask * clip
            param.grad.data = mask * param.grad.data

    return


def train_batch(
    inputs,
    targets,
    model,
    nonzero_ratio,
    clip,
    criterion,
    optimiser,
    pbar,
):
    optimiser.zero_grad()

    # get loss on batch
    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    loss.backward()

    # CDR on gradients
    cdr_on_grads(
        model,
        nonzero_ratio,
        clip,
    )

    # optimiser step
    optimiser.step()

    pbar.update(1)

    return loss.item(), outputs


def train_epoch(
    model,
    train_loader,
    criterion,
    optimiser,
    device,
    pbar,
    noise_rate,
    num_gradual,
):
    model.to(device)

    model.train()

    train_loss = 0
    train_total = 0
    train_acc_meter = AccuracyMetric(topk=[1, 5])

    for batch_idx, (inputs, targets, sources) in enumerate(train_loader):
        inputs, targets, sources = (
            inputs.to(device),
            targets.to(device),
            sources.to(device),
        )
        sources = sources.squeeze(-1)

        clip_narry = np.linspace(1 - noise_rate, 1, num=num_gradual)
        clip_narry = clip_narry[::-1]
        if epoch < num_gradual:
            clip = clip_narry[epoch]

        clip = 1 - noise_rate

        loss, outputs = train_batch(
            inputs,
            targets,
            model,
            clip,
            clip,
            criterion,
            optimiser,
            pbar,
        )

        train_acc_meter.update(outputs, targets)
        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

    metrics = train_acc_meter.compute_key_value()

    return train_loss / train_total, metrics["accuracy01"], metrics["accuracy05"]


def test(test_loader, model, criterion, device):
    model.to(device)
    model.eval()

    test_acc_meter = AccuracyMetric(topk=[1, 5])
    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, (inputs, targets, sources) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()

            loss = torch.mean(loss)
            test_acc_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        metrics = test_acc_meter.compute_key_value()
    return test_loss / test_total, metrics["accuracy01"], metrics["accuracy05"]


for dataset_name in DATASET_NAMES:
    for c in CORRUPTION_TYPE:
        print(
            f"for dataset {dataset_name}",
            f"and corruption type {c}",
            f"the following run has been completed",
            results[dataset_name][c].keys(),
        )


dataset = GoEmotions(
    root=DATA_DIR,
    max_length=128,
)

dataset = ToMemory(dataset, device=DEVICE)

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
    generator=torch.Generator().manual_seed(exp_seed),
)
train_idx = train_dataset.indices
test_idx = test_dataset.indices


exp_number = 0
for run in RUNS:
    for dataset_name in DATASET_NAMES:
        for corruption_type in CORRUPTION_TYPE:
            exp_seed = int(np.random.default_rng(exp_seed).integers(0, 1e9))
            exp_number += 1

            if run in results[dataset_name][corruption_type].keys():
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
                len(RUNS) * len(DATASET_NAMES) * len(CORRUPTION_TYPE),
                "with seed",
                exp_seed,
                "...",
            )

            print("Dataset:", dataset_name)
            print("Corruption Type:", corruption_type)
            print("Run:", run)

            if corruption_type == "random_label":

                train_ds = SourceDatasetRandomLabel(
                    dataset=train_dataset,
                    targets=dataset.dataset.targets[train_idx],
                    sources=dataset.dataset.sources[train_idx],
                    n_corrupt_sources=N_CORRUPT_SOURCES,
                    seed=exp_seed,
                )

            elif corruption_type == "random_permute":
                train_ds = SourceDatasetRandomPermute(
                    dataset=train_dataset,
                    sources=dataset.dataset.sources[train_idx],
                    n_corrupt_sources=N_CORRUPT_SOURCES,
                    seed=exp_seed,
                )

            elif corruption_type == "random_token":
                train_ds = SourceDatasetRandomToken(
                    dataset=train_dataset,
                    sources=dataset.dataset.sources[train_idx],
                    n_corrupt_sources=N_CORRUPT_SOURCES,
                    num_tokens=len(dataset.dataset.vocab),
                    amount_noise=TOKEN_NOISE_AMOUNT,
                    seed=exp_seed,
                )

            elif corruption_type == "original":
                train_ds = SourceDatasetOriginal(
                    dataset=train_dataset,
                    sources=dataset.dataset.sources[train_idx],
                    seed=exp_seed,
                )

            else:
                raise ValueError("Invalid corruption type")

            train_dl = torchdata.DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )

            test_dl = torchdata.DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False
            )

            def get_model():

                model = TransformerEncoder(
                    d_model=EMBEDDING_DIM,
                    n_classes=len(dataset.dataset.y_unique),
                    ntokens=len(dataset.dataset.vocab),
                    nhead=N_HEADS,
                    num_layers=N_LAYERS,
                    dropout=DROPOUT,
                    max_len=MAX_LENGTH,
                    criterion=nn.CrossEntropyLoss(reduction="none"),
                )

                model = model.to(DEVICE)

                # Observe that all parameters are being optimized
                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=LR,
                )
                criterion = nn.CrossEntropyLoss()

                return model, optimiser, criterion

            model, optimiser, criterion = get_model()

            pbar = tqdm.tqdm(total=N_EPOCHS * len(train_dl), desc="Training")

            max_test_top1acc = 0
            max_test_top5acc = 0

            results_this_train = {}
            for epoch in range(N_EPOCHS):

                train_loss, train_top1acc, train_top5acc = train_epoch(
                    model=model,
                    train_loader=train_dl,
                    criterion=criterion,
                    optimiser=optimiser,
                    device=DEVICE,
                    pbar=pbar,
                    noise_rate=NOISE_RATE,
                    num_gradual=NUM_GRADUAL,
                )

                test_loss, test_top1acc, test_top5_acc = test(
                    test_dl, model, criterion, DEVICE
                )

                if test_top1acc > max_test_top1acc:
                    max_test_top1acc = test_top1acc

                if test_top5_acc > max_test_top5acc:
                    max_test_top5acc = test_top5_acc

                results_this_train[epoch] = {
                    "train_loss": train_loss,
                    "train_top1acc": train_top1acc,
                    "train_top5acc": train_top5acc,
                    "test_loss": test_loss,
                    "test_top1acc": test_top1acc,
                    "test_top5acc": test_top5_acc,
                }

                pbar.set_postfix(
                    {
                        "test_acc": test_top1acc,
                        "max_test_acc": max_test_top1acc,
                        "test_top5_acc": test_top5_acc,
                    }
                )

            pbar.close()

            results[dataset_name][corruption_type][run] = results_this_train

            print(results_this_train[epoch])

            # save results to json
            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)
