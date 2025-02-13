import sys
import os
import math
from pathlib import Path
import argparse
import time
import json
import tqdm
import numpy as np
import datasets
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as torchdata

sys.path.append("../../")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--results_name", type=str, default="results_cot.json")
parser.add_argument(
    "--test-dir",
    help="The directory to save the model test results",
    type=str,
    default="../../outputs/imdb_random_label",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# --- experiment options
DATASET_NAMES = [
    "imdb",
]
CORRUPTION_TYPE = [
    "random_label",
    "random_permute",
    "original",
]
N_SOURCES = 10
N_CORRUPT_SOURCES = 4
TOKEN_NOISE_AMOUNT = 0.5
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
)
RUNS = [int(k) for k in args.runs]
DATA_DIR = "../../data/"
TEST_DIR = args.test_dir
RESULTS_FILE = os.path.join(TEST_DIR, args.results_name)

# --- training options
BATCH_SIZE = 256
N_EPOCHS = 160
LR = 0.001


exp_seed = args.seed

EMBEDDING_DIM = 256  # embedding dimension
D_HIDDEN = (
    512  # dimension of the feedforward network model in ``nn.TransformerEncoder``
)
N_LAYERS = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
N_HEADS = 2  # number of heads in ``nn.MultiheadAttention``
DROPOUT = 0.25  # dropout probability
MAX_LENGTH = 256  # the max length of the sequence


## CoTeaching hyperparameters
NUM_GRADUAL = 10
EXPONENT = 1
EPOCH_DECAY_START = 10


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


class IMDBDataset(torchdata.Dataset):
    def __init__(
        self,
        dataset,
        tokeniser,
        vocab,
        max_length=512,
    ):
        self.dataset = dataset
        self.target = dataset["label"]
        self.max_length = max_length
        self.tokeniser = tokeniser
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text_tensor = (
            torch.zeros(self.max_length, dtype=torch.long) * self.vocab["<pad>"]
        )
        text_split = self.tokeniser(self.dataset[idx]["text"])[: self.max_length]
        text_tokens = [
            torch.tensor(self.vocab(self.tokeniser(item)), dtype=torch.long)
            for item in text_split
        ]
        text_tensor_raw = torch.cat(tuple(filter(lambda t: t.numel() > 0, text_tokens)))
        text_tensor[: text_tensor_raw.size(0)] = text_tensor_raw
        return text_tensor, self.dataset[idx]["label"]


class SourceDatasetOriginal(torchdata.Dataset):
    def __init__(self, dataset, n_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)

        self.dataset = dataset
        self.n_sources = n_sources

        self.sources = self.rng.choice(n_sources, len(dataset), replace=True)
        self.corrupt_sources = []

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


class SourceDatasetRandomPermute(torchdata.Dataset):
    def __init__(self, dataset, n_sources, n_corrupt_sources, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        self.seed = self.rng.integers(1e9)

        self.dataset = dataset
        self.n_sources = n_sources

        self.sources = self.rng.choice(n_sources, len(dataset), replace=True)
        self.corrupt_sources = self.rng.choice(
            n_sources, n_corrupt_sources, replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data
        source = self.sources[idx]
        x, y = self.dataset[idx]

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


class CoTeaching(torchdata.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        point = self.dataset[index]
        if len(point) == 3:
            x, y, s = point
        else:
            x, y = point

        return x, y, index

    def __len__(self):
        return len(self.dataset)


# the following is from https://discuss.pytorch.org/t/top-k-error-calculation/48815/3
# with edits to the documentation and commenting.
def accuracy_topk(output: torch.tensor, target: torch.tensor, topk: tuple = (1,)):
    """
    https://discuss.pytorch.org/t/top-k-error-calculation/48815/3

    Computes the accuracy over the k top predictions for the specified values of k.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch


    Arguments
    ---------

    - output: torch.tensor:
        The prediction of the model.

    - target: torch.tensor:
        The targets that each prediction corresponds to.

    - topk: tuple (optional):
        This is a tuple of values that represent the k values
        for which the accuracy should be calculated with.


    Returns
    ---------

    - topk_accuracies: list:
        This returns a list of the top k accuracies for
        the k values specified.

    """
    with torch.no_grad():
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)
        # get top maxk indicies that correspond to the most likely probability scores
        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B]

        # expand the target tensor to the same shape as y_pred
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        # compare the target to each of the top k predictions made by the model
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] for each example we know which topk prediction matched truth

        # get topk accuracy
        list_topk_accs = []
        for k in topk:
            # find which of the top k predictions were correct
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # calculate the number of correct predictions
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return (
            list_topk_accs  # list of topk accuracies for batch [topk1, topk2, ... etc]
        )


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


for dataset_name in DATASET_NAMES:
    for c in CORRUPTION_TYPE:
        print(
            f"for dataset {dataset_name}",
            f"and corruption type {c}",
            f"the following run has been completed",
            results[dataset_name][c].keys(),
        )


train_hf_data, test_hf_data = datasets.load_dataset(
    path="imdb", cache_dir=DATA_DIR, split=["train", "test"], keep_in_memory=True
)

tokeniser = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(
    map(tokeniser, train_hf_data["text"]),
    specials=["<pad>", "<unk>"],
)
vocab.set_default_index(vocab["<unk>"])

train_dataset = IMDBDataset(
    train_hf_data,
    tokeniser=tokeniser,
    vocab=vocab,
    max_length=MAX_LENGTH,
)
test_dataset = IMDBDataset(
    test_hf_data,
    tokeniser=tokeniser,
    vocab=vocab,
    max_length=MAX_LENGTH,
)

train_dataset_memory = ToMemory(train_dataset, device=DEVICE)
test_dataset_memory = ToMemory(test_dataset, device=DEVICE)


# load all data into memory
train_memory_dl = torchdata.DataLoader(
    train_dataset_memory, batch_size=BATCH_SIZE, shuffle=False
)
test_memory_dl = torchdata.DataLoader(
    test_dataset_memory, batch_size=BATCH_SIZE, shuffle=False
)

# for x, y in tqdm.tqdm(train_memory_dl, desc="Loading training data into memory"):
#     pass

# for x, y in tqdm.tqdm(test_memory_dl, desc="Loading validation data into memory"):
#     pass


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
                    dataset=train_dataset_memory,
                    targets=train_dataset.target,
                    n_sources=N_SOURCES,
                    n_corrupt_sources=N_CORRUPT_SOURCES,
                    seed=exp_seed,
                )

            elif corruption_type == "random_permute":
                train_ds = SourceDatasetRandomPermute(
                    dataset=train_dataset_memory,
                    n_sources=N_SOURCES,
                    n_corrupt_sources=N_CORRUPT_SOURCES,
                    seed=exp_seed,
                )

            elif corruption_type == "original":
                train_ds = SourceDatasetOriginal(
                    dataset=train_dataset_memory,
                    n_sources=N_SOURCES,
                    seed=exp_seed,
                )

            else:
                raise ValueError("Invalid corruption type")

            train_loader_cot = torchdata.DataLoader(
                CoTeaching(train_ds),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            test_loader_cot = torchdata.DataLoader(
                CoTeaching(test_dataset_memory), batch_size=BATCH_SIZE, shuffle=False
            )
            test_loader = torchdata.DataLoader(
                test_dataset_memory, batch_size=BATCH_SIZE, shuffle=False
            )

            def get_model():

                model = SentimentAnalysisModel(
                    ntokens=len(train_ds.dataset.dataset.vocab),
                    n_classes=2,
                    hidden_size=D_HIDDEN,
                    embedding_size=EMBEDDING_DIM,
                    n_layers=N_LAYERS,
                    dropout=DROPOUT,
                )

                model = model.to(DEVICE)

                # Observe that all parameters are being optimized
                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=LR,
                )
                criterion = nn.CrossEntropyLoss()

                return model, optimiser, criterion

            corrupt_sources = [int(s) for s in train_ds.corrupt_sources]

            ## model initialization for different datasets

            forget_rate = 0.2
            learning_rate = LR

            # Adjust learning rate and betas for Adam Optimizer
            mom1 = 0.9
            mom2 = 0.1
            alpha_plan = [learning_rate] * N_EPOCHS
            beta1_plan = [mom1] * N_EPOCHS
            for i in range(EPOCH_DECAY_START, N_EPOCHS):
                alpha_plan[i] = (
                    float(N_EPOCHS - i) / (N_EPOCHS - EPOCH_DECAY_START) * learning_rate
                )
                beta1_plan[i] = mom2

            # define drop rate schedule
            rate_schedule = np.ones(N_EPOCHS) * forget_rate
            rate_schedule[:NUM_GRADUAL] = np.linspace(
                0, forget_rate**EXPONENT, NUM_GRADUAL
            )

            # Loss functions
            def loss_coteaching(y_1, y_2, t, forget_rate, ind):
                loss_1 = F.cross_entropy(y_1, t, reduction="none")
                ind_1_sorted = np.argsort(loss_1.data.cpu()).to(DEVICE)

                loss_1_sorted = loss_1[ind_1_sorted]

                loss_2 = F.cross_entropy(y_2, t, reduction="none")
                ind_2_sorted = np.argsort(loss_2.data.cpu()).to(DEVICE)

                remember_rate = 1 - forget_rate
                num_remember = int(remember_rate * len(loss_1_sorted))

                ind_1_update = ind_1_sorted[:num_remember]
                ind_2_update = ind_2_sorted[:num_remember]
                # exchange
                loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
                loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

                return (
                    torch.sum(loss_1_update) / num_remember,
                    torch.sum(loss_2_update) / num_remember,
                )

            # Train the Model
            def train_epoch(
                train_loader, epoch, model1, optimizer1, model2, optimizer2, pbar
            ):
                train_total = 0
                train_correct = 0
                train_total2 = 0
                train_correct2 = 0

                for i, (images, labels, indexes) in enumerate(train_loader):
                    ind = indexes.cpu().numpy().transpose()

                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Forward + Backward + Optimize
                    logits1 = model1(images)
                    prec1 = accuracy_topk(logits1, labels, topk=(1,))[0]
                    train_total += 1
                    train_correct += prec1

                    logits2 = model2(images)
                    prec2 = accuracy_topk(logits2, labels, topk=(1,))[0]
                    train_total2 += 1
                    train_correct2 += prec2
                    loss_1, loss_2 = loss_coteaching(
                        logits1, logits2, labels, rate_schedule[epoch], ind
                    )

                    optimizer1.zero_grad()
                    loss_1.backward()
                    optimizer1.step()
                    optimizer2.zero_grad()
                    loss_2.backward()
                    optimizer2.step()

                    pbar.update(1)

                train_acc1 = float(train_correct) / float(train_total)
                train_acc2 = float(train_correct2) / float(train_total2)

                return train_acc1, train_acc2

            # Evaluate the Model
            def evaluate(test_loader, model1, model2):
                model1.eval()  # Change model to 'eval' mode.
                correct1 = 0
                total1 = 0
                for images, labels, _ in test_loader:
                    images = images = images.to(DEVICE)
                    logits1 = model1(images)
                    outputs1 = F.softmax(logits1, dim=1)
                    _, pred1 = torch.max(outputs1.data, 1)
                    total1 += labels.size(0)
                    correct1 += (pred1.cpu() == labels).sum()

                model2.eval()  # Change model to 'eval' mode
                correct2 = 0
                total2 = 0
                for images, labels, _ in test_loader:
                    images = images = images.to(DEVICE)
                    logits2 = model2(images)
                    outputs2 = F.softmax(logits2, dim=1)
                    _, pred2 = torch.max(outputs2.data, 1)
                    total2 += labels.size(0)
                    correct2 += (pred2.cpu() == labels).sum()

                acc1 = 100 * float(correct1) / float(total1)
                acc2 = 100 * float(correct2) / float(total2)
                return acc1, acc2

            model1, optimizer1, criterion1 = get_model()
            model2, optimizer2, criterion2 = get_model()

            # training
            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(total=N_EPOCHS * len(train_loader_cot))
            results[dataset_name][corruption_type][run] = {}
            for epoch in range(N_EPOCHS):
                # train models
                model1.train()
                model2.train()
                train_acc1, train_acc2 = train_epoch(
                    train_loader_cot,
                    epoch,
                    model1,
                    optimizer1,
                    model2,
                    optimizer2,
                    pbar,
                )
                # evaluate models
                test_acc1, test_acc2 = evaluate(test_loader_cot, model1, model2)

                # performance on the test
                tqdm.tqdm._instances.clear()
                model1.eval()
                with torch.no_grad():
                    sums, total = 0, 0
                    predictions = []
                    targets = []
                    for batch in test_loader:
                        (
                            X,
                            y,
                        ) = batch
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        outputs = torch.softmax(model1(X), 1)
                        predictions.append(outputs.cpu())
                        targets.append(y.cpu())
                    predictions = torch.cat(predictions)
                    targets = torch.cat(targets)

                print(
                    "accuracy on the test set after robust learning",
                    accuracy_topk(predictions, targets, topk=(1,)),
                )

                results[dataset_name][corruption_type][run][epoch] = {
                    k: v.item()
                    for k, v in zip(
                        ["test_top1acc"],
                        accuracy_topk(predictions, targets, topk=(1,)),
                    )
                }

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)

            pbar.close()
