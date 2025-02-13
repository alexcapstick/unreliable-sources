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
from torchvision.datasets.utils import download_url
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.utils.data as torchdata

sys.path.append("../../")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--results_name", type=str, default="results_idpa.json")
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

exp_seed = args.seed


EMBEDDING_DIM = 256  # embedding dimension
D_HIDDEN = (
    256  # dimension of the feedforward network model in ``nn.TransformerEncoder``
)
N_LAYERS = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
N_HEADS = 4  # number of heads in ``nn.MultiheadAttention``
DROPOUT = 0.1  # dropout probability
MAX_LENGTH = 128  # the max length of the sequence


## IDPA hyperparameters
ETA_LR = 5e-2
ETA_INIT = 0.01


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


class InstanceDependentDataset(torchdata.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.psi = [0] * len(dataset)
        return

    def update_psi(self, idx, value):
        self.psi[idx] = value

    def __getitem__(self, index):
        point = self.dataset[index]

        if len(point) == 3:
            x, y, s = point
        else:
            x, y = point

        return x, y, index, self.psi[index]

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

            ## model initialization for different datasets

            train_loader_idd = torchdata.DataLoader(
                InstanceDependentDataset(train_ds),
                shuffle=True,
                batch_size=BATCH_SIZE,
            )

            test_loader = torchdata.DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=BATCH_SIZE,
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

            # IDD Helper functions

            eye = torch.eye(len(dataset.dataset.y_unique)).to(DEVICE)

            # + Class-conditional probability
            def predict_truelabel(etas, targets, pnl):
                # re-estimated posterior of true label given eta and posterior of noisy labels
                part1 = eye[targets.long()] * (1 - etas).view(-1, 1)
                part2 = (pnl * etas).view(-1, 1)
                return (part1 + part2).clamp(min=1e-5).log()

            def logs(msg):
                sys.stdout.write("\r" + msg)

            # class re-weighting
            def upsample(targets):
                return torch.ones_like(targets)

            model, optimizer, criterion = get_model()

            # training base model to get the psi values

            tqdm.tqdm._instances.clear()
            model.train()
            pbar = tqdm.tqdm(desc="Training", total=N_EPOCHS * len(train_loader_idd))
            for epoch in range(N_EPOCHS):
                pbar.postfix = f"Epoch: {epoch+1}/{N_EPOCHS}"
                for batch in train_loader_idd:
                    optimizer.zero_grad()

                    X, y, _, _ = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)

                    output = model(X)
                    loss = criterion(output, y)

                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
            pbar.close()

            # predicting labels that are used as posterior

            tqdm.tqdm._instances.clear()
            model.eval()
            with torch.no_grad():
                sums, total = 0, 0  #
                predictions = []
                targets = []
                for batch in tqdm.tqdm(train_loader_idd, desc="Predicting Labels"):
                    X, y, indies, _ = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    outputs = torch.softmax(model(X), 1)
                    psi = (outputs * eye[y]).sum(1).cpu().tolist()
                    for idx, p in zip(indies, psi):  #
                        train_loader_idd.dataset.update_psi(idx, p)  #
                    sums += sum(psi)  #
                    total += len(psi)  #
                    predictions.append(outputs.cpu())
                    targets.append(y.cpu())
                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

            print(
                "accuracy on the training set before robust learning",
                accuracy_topk(predictions, targets, topk=(1, 5)),
            )

            # performance on the test
            tqdm.tqdm._instances.clear()
            # predicting labels that are used as posterior
            model.eval()
            with torch.no_grad():
                sums, total = 0, 0  #
                predictions = []
                targets = []
                for batch in tqdm.tqdm(test_loader, desc="Predicting Test Labels"):
                    (
                        X,
                        y,
                        s,
                    ) = batch
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    outputs = torch.softmax(model(X), 1)
                    predictions.append(outputs.cpu())
                    targets.append(y.cpu())
                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

            print(
                "accuracy on the test set before robust learning",
                accuracy_topk(predictions, targets, topk=(1, 5)),
            )

            ETA = torch.zeros((len(train_loader_idd.dataset),)).to(DEVICE) + ETA_INIT

            # double training loop to get robust model

            tqdm.tqdm._instances.clear()
            # TRAIN
            model.train()
            pbar = tqdm.tqdm(
                desc="Training Robust Model",
                total=N_EPOCHS * len(train_loader_idd),
            )
            results[dataset_name][corruption_type][run] = {}
            for epoch in range(N_EPOCHS):
                model.train()
                time_start = time.time()
                correct, total = 0, 0
                eta_hist = torch.Tensor([0] * 10).to(
                    DEVICE
                )  ##### was commented out for some reason #####
                pbar.set_description(f"Training, Epoch: {epoch+1}/{N_EPOCHS}")

                for inputs, targets, indies, pnl in train_loader_idd:
                    inputs, targets, indies, pnl = (
                        inputs.to(DEVICE),
                        targets.to(DEVICE),
                        indies.to(DEVICE),
                        pnl.to(DEVICE).float(),
                    )

                    optimizer.zero_grad()
                    outputs = torch.log_softmax(model(inputs), 1)

                    # ALTERNATING OPTIMIZATION
                    # ---------------------------------------------------- #
                    # + Prediction                                         #
                    pyz_x = (
                        predict_truelabel(ETA[indies], targets, pnl) + outputs.detach()
                    ).exp()
                    pz_x = pyz_x / pyz_x.sum(1).view(-1, 1)  #
                    # + Optimization                                       #
                    # |- classifier                                        #
                    loss = -(upsample(targets) * (pz_x * outputs).sum(1)).mean()
                    loss.backward()  #
                    optimizer.step()  #
                    # |- confusing                                         #
                    # For the simplicty of the updating rule, we actually assume pnl is close
                    # to 1. Directly assuming pnl=1 can lead to similar results in practice.
                    if epoch != 0:  #
                        disparities = (
                            pz_x
                            * (
                                1
                                + (pnl * ETA[indies] - ETA[indies] - 1).view(-1, 1)
                                * eye[targets]
                            )
                        ).sum(1)
                        ETA[indies] += (
                            ETA_LR * disparities / ETA[indies].clamp(min=1e-5)
                        )
                        ETA[indies] = ETA[indies].clamp(min=0, max=1)  #
                    # ---------------------------------------------------- #

                    # ---------------------------------------------------- #
                    # + Classifier                                         #
                    predicts = outputs.detach().argmax(1)  #
                    correct += (predicts == targets).float().sum().item()  #
                    total += inputs.size(0)  #
                    # + Etas                                               #
                    etas = ETA[indies]  #
                    eta_hist += (
                        torch.eye(10)
                        .to(DEVICE)[(etas * 10).long().clamp(min=0, max=9)]
                        .sum(0)
                    )
                    # ---------------------------------------------------- #
                    pbar.update(1)

                # performance on the test
                tqdm.tqdm._instances.clear()
                # predicting labels that are used as posterior
                model.eval()
                with torch.no_grad():
                    sums, total = 0, 0  #
                    predictions = []
                    targets = []
                    for batch in test_loader:
                        (X, y, s) = batch
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        outputs = torch.softmax(model(X), 1)
                        predictions.append(outputs.cpu())
                        targets.append(y.cpu())
                    predictions = torch.cat(predictions)
                    targets = torch.cat(targets)

                print(
                    "accuracy on the test set after robust learning",
                    accuracy_topk(predictions, targets, topk=(1, 5)),
                )

                results[dataset_name][corruption_type][run][epoch] = {
                    k: v.item()
                    for k, v in zip(
                        ["test_top1acc", "test_top5acc"],
                        accuracy_topk(predictions, targets, topk=(1, 5)),
                    )
                }

            pbar.close()

            # save results to json

            with open(RESULTS_FILE, "w") as fp:
                json.dump(results, fp)
