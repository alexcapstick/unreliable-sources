import numpy as np
from pathlib import Path
import tqdm
import time
from catalyst.metrics import AccuracyMetric, AUCMetric, BinaryPrecisionRecallF1Metric
from torch.utils.tensorboard import SummaryWriter
import json
import typing
import pandas as pd
import torch
import torch.nn as nn
import argparse
import wfdb
import ast
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_sample_weight
from sklearn.metrics import (
    precision_recall_curve,
    auc,
)
from torchvision.datasets.utils import download_and_extract_archive

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
    default="./outputs/ecg_results/",
)
args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)

# --- experiment options
DATASET_NAMES = ["ptb_xl"]
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
WARMUP_ITERS = 0

# --- data options
N_CORRUPT_SOURCES = [2, 3, 4, 5, 6, 7, 8]
MIN_CORRUPTION_LEVEL = args.min_corruption_level
MAX_CORRUPTION_LEVEL = args.max_corruption_level
BATCH_SIZE = 64
N_EPOCHS = 40
LR = 0.001
DATA_SUBSET = False
SAMPLING_RATE = 100  # 500 # 100
CORRUPTION_STD = 0.2  # 0.1
AXIS = "both"  # "x" # "y" # "both"

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
        self.memory_dataset[index] = output
        return output

    def __len__(self):
        return len(self.dataset)


########################### models


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_channels: int,
        out_channels: int,
        out_dim: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
    ):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.out_dim = out_dim

        self.x1 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                padding="same",
            ),
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.x2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                stride=input_dim // out_dim,
            )
        )

        self.y1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
            )
        )

        self.xy1 = nn.Sequential(
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        return

    def _skip_connection(self, y):
        downsample = self.input_dim // self.out_dim
        if downsample > 1:
            same_pad = np.ceil(
                0.5
                * (
                    (y.size(-1) // self.out_dim) * (self.out_dim - 1)
                    - y.size(-1)
                    + downsample
                )
            )
            if same_pad < 0:
                same_pad = 0
            y = nn.functional.pad(y, (int(same_pad), int(same_pad)), "constant", 0)
            y = nn.MaxPool1d(
                kernel_size=downsample,
                stride=downsample,
            )(y)

        elif downsample == 1:
            pass
        else:
            raise ValueError("Size of input should always decrease.")
        y = self.y1(y)

        return y

    def forward(self, inputs):
        x, y = inputs

        # y
        y = self._skip_connection(y)

        # x
        x = self.x1(x)
        same_pad = np.ceil(
            0.5
            * (
                (x.size(-1) // self.out_dim) * (self.out_dim - 1)
                - x.size(-1)
                + self.kernel_size
            )
        )
        if same_pad < 0:
            same_pad = 0
        x = nn.functional.pad(x, (int(same_pad), int(same_pad)), "constant", 0)
        x = self.x2(x)

        # xy
        xy = x + y
        y = x
        xy = self.xy1(xy)

        return [xy, y]


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        input_channels: int = 64,
        n_output: int = 10,
        kernel_size: int = 16,
        dropout_rate: float = 0.2,
        criterion: nn.Module = nn.CrossEntropyLoss(reduction="none"),
    ):
        """
        Model with 4 :code:`ResBlock`s, in which
        the number of channels increases linearly
        and the output dimensions decreases
        exponentially. This model will
        require the input dimension to be of at least
        256 in size. This model is designed for sequences,
        and not images. The expected input is of the type::

            [n_batches, n_filters, sequence_length]


        Examples
        ---------

        .. code-block::

            >>> model = ResNet(
                    input_dim=4096,
                    input_channels=64,
                    kernel_size=16,
                    n_output=5,
                    dropout_rate=0.2,
                    )
            >>> model(
                    torch.rand(1,64,4096)
                    )
            tensor([[0.3307, 0.4782, 0.5759, 0.5214, 0.6116]], grad_fn=<SigmoidBackward0>)


        Arguments
        ---------

        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and
            the sequence length.
            Defaults to :code:`4096`.

        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.

        - n_output: int, optional:
            The number of output classes in
            the prediction.
            Defaults to :code:`10`.

        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence.
            Defaults to :code:`16`.

        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.

        """
        super(ResNet, self).__init__()

        self.x1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(
                num_features=input_channels,
                affine=False,
            ),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            ResBlock(
                input_dim=input_dim,  # 4096
                input_channels=input_channels,  # 64
                out_channels=2 * input_channels // 1,  # 128
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 4,  # 1024,
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 4,  # 1024
                input_channels=2 * input_channels // 1,  # 128
                out_channels=3 * input_channels // 1,  # 192
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 16,  # 256
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 16,  # 256
                input_channels=3 * input_channels // 1,  # 192
                out_channels=4 * input_channels // 1,  # 256
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 64,  # 64
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 64,  # 64
                input_channels=4 * input_channels // 1,  # 256
                out_channels=5 * input_channels // 1,  # 320
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 256,  # 16
                dropout_rate=dropout_rate,
            ),
        )

        self.x3 = nn.Flatten()
        self.x4 = nn.Sequential(
            nn.Linear(
                (input_dim // 256) * (5 * input_channels // 1),
                n_output,
            )
        )

        self.criterion = criterion

    def forward(self, x, y=None, return_loss=False):
        x = self.x1(x)
        x, _ = self.x2([x, x])
        x = self.x3(x)
        x = self.x4(x)

        if return_loss:
            assert y is not None
            return self.criterion(x, y), x

        return x


def ResNet1D_PTBXL(
    input_dim: int = 10 * SAMPLING_RATE,
    input_channels: int = 12,
    n_output: int = 2,
    kernel_size: int = 15,
    dropout_rate: float = 0.2,
    criterion: nn.Module = nn.CrossEntropyLoss(reduction="none"),
):
    return ResNet(
        input_dim=input_dim,
        input_channels=input_channels,
        n_output=n_output,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        criterion=criterion,
    )


########################### datasets


class PTB_XL(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str = "./",
        train: bool = True,
        sampling_rate: typing.Literal[100, 500] = 100,
        source_name: typing.Literal["nurse", "site", "device"] = "nurse",
        return_sources: bool = True,
        binary: bool = False,
        subset=False,
    ):
        """
        ECG Data, as described here: https://physionet.org/content/ptb-xl/1.0.2/.
        A positive class when :code:`binary=True`, indicates that
        the ECG Data is abnormal.



        Examples
        ---------

        .. code-block::

            >>> dataset = PTB_XL(
            ...     data_path='../../data/',
            ...     train=True,
            ...     source_name='nurse',
            ...     sampling_rate=500,
            ...     return_sources=False,
            ...     )



        Arguments
        ---------

        - data_path: str, optional:
            The path that the data is saved
            or will be saved.
            Defaults to :code:`'./'`.

        - train: bool, optional:
            Whether to load the training or testing set.
            Defaults to :code:`True`.

        - sampling_rate: typing.Literal[100, 500], optional:
            The sampling rate. This should be
            in :code:`[100, 500]`.
            Defaults to :code:`100`.

        - source_name: typing.Literal['nurse', 'site', 'device'], optional:
            Which of the three attributes should be
            interpretted as the data sources. This should
            be in  :code:`['nurse', 'site', 'device']`.
            This is ignored if :code:`return_sources=False`.
            Defaults to :code:`'nurse'`.

        - return_sources: bool, optional:
            Whether to return the sources alongside
            the data and targets. For example, with
            :code:`return_sources=True`, for every index
            this dataset will return :code:`data, target, source`.
            Defaults to :code:`True`.

        - binary: bool, optional:
            Whether to return classes based on whether the
            ecg is normal or not, and so a binary classification
            problem.
            Defaults to :code:`False`.

        - subset: bool, optional:
            If :code:`True`, only the first 1000 items
            of the training and test set will be returned.
            Defaults to :code:`False`.


        """

        assert sampling_rate in [
            100,
            500,
        ], "Please choose sampling_rate from [100, 500]"
        assert type(train) == bool, "Please use train = True or False"
        assert source_name in [
            "nurse",
            "site",
            "device",
        ], "Please choose source_name from ['nurse', 'site', 'device']"

        self.data_path = data_path
        self.download()
        self.data_path = os.path.join(
            self.data_path,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/",
        )

        self.train = train
        self.sampling_rate = sampling_rate
        self.source_name = source_name
        self.return_sources = return_sources
        self.binary = binary
        self.meta_data = pd.read_csv(self.data_path + "ptbxl_database.csv")
        self.meta_data["scp_codes"] = self.meta_data["scp_codes"].apply(
            lambda x: ast.literal_eval(x)
        )
        self.aggregate_diagnostic()  # create diagnostic columns
        self.feature_names = [
            "I",
            "II",
            "III",
            "aVL",
            "aVR",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.meta_data = self.meta_data[~self.meta_data[self.source_name].isna()]

        if self.train:
            self.meta_data = self.meta_data.query("strat_fold != 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        else:
            self.meta_data = self.meta_data.query("strat_fold == 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]

        if binary:
            self.targets = self.meta_data[["NORM", "CD", "HYP", "MI", "STTC"]].values
            self.targets = 1 - self.targets[:, 0]
        else:
            self.targets = self.meta_data[["NORM", "CD", "HYP", "MI", "STTC"]].values

        self.sources = self.meta_data[self.source_name].values

        return

    def _check_exists(self):
        folder = os.path.join(
            self.data_path,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2",
        )
        return os.path.exists(folder)

    def download(self):
        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url="https://physionet.org/static"
            "/published-projects/ptb-xl/"
            "ptb-xl-a-large-publicly-available"
            "-electrocardiography-dataset-1.0.2.zip",
            download_root=self.data_path,
            extract_root=self.data_path,
            filename="ptbxl.zip",
            remove_finished=True,
        )

        return

    @staticmethod
    def single_diagnostic(y_dict, agg_df):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_diagnostic(self):
        agg_df = pd.read_csv(self.data_path + "scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        self.meta_data["diagnostic_superclass"] = self.meta_data["scp_codes"].apply(
            self.single_diagnostic,
            agg_df=agg_df,
        )
        mlb = MultiLabelBinarizer()
        self.meta_data = self.meta_data.join(
            pd.DataFrame(
                mlb.fit_transform(self.meta_data.pop("diagnostic_superclass")),
                columns=mlb.classes_,
                index=self.meta_data.index,
            )
        )
        return

    def __getitem__(self, index):
        data = self.meta_data.iloc[index]

        if self.sampling_rate == 100:
            f = data["filename_lr"]
            x = wfdb.rdsamp(self.data_path + f)
        elif self.sampling_rate == 500:
            f = data["filename_hr"]
            x = wfdb.rdsamp(self.data_path + f)
        x = torch.tensor(x[0]).transpose(0, 1).float()
        y = torch.tensor(
            data[["NORM", "CD", "HYP", "MI", "STTC"]].values.astype(np.int64)
        )
        if self.binary:
            y = 1 - y[0]
        source = data[self.source_name]

        if self.return_sources:
            return x, y, source
        else:
            return x, y

    def __len__(self):
        return len(self.meta_data)


class ECGCorruptor(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        corrupt_sources: typing.Union[list, int, None] = None,
        noise_level: typing.Union[list, float, None] = None,
        seed: typing.Union[int, None] = None,
        axis: str = "both",
        x_noise_std: float = 0.1,
    ):
        """
        ECG Data corruptor. You may pass a noise level, sources to corrupt,
        and the seed for determining the random events. This
        class allows you to corrupt either the :code:`'x'`, :code:`'y'`, 
        or :code:`'both'`. This class is built specifically for use with
        PTB_XL (found in :code:`aml.data.datasets`).

        This function will work as expected on all devices.
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = ECGCorruptor(
            ...     dataset=dataset_train
            ...     corrupt_sources=[0,1,2,3], 
            ...     noise_level=0.5, 
            ...     )

        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset:
            The dataset to corrupt. When iterated over,
            the dataset should return :code:`x`, :code:`y`, 
            and :code:`source`.

        - corrupt_sources: typing.Union[list, int, None], optional:
            The sources to corrupt in the dataset. This can be a 
            list of sources, an integer of the source, or :code:`None`
            for no sources to be corrupted.
            Defaults to :code:`None`.

        - noise_level: typing.Union[list, int, None], optional:
            This is the level of noise to apply to the dataset. 
            It can be a list of noise levels, a single noise level to
            use for all sources, or :code:`None` for no noise.
            Defaults to :code:`None`.

        - seed: typing.Union[int, None], optional:
            This is the seed that is used to determine random events.
            Defaults to :code:`None`.

        - axis: str, optional:
            This is the axis to apply the corruption to. This
            should be either :code:`'x'`, :code:`'y'`, 
            or :code:`'both'`.
            
            - :code:`'x'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1`.
            
            - :code:`'y'`: \
            Swaps the binary label using the function :code:`1-y_true`.
            
            - :code:`'both'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1` \
            and swaps the binary label using the function :code:`1-y_true`.

            Defaults to :code:`'both'`.

        - x_noise_std: float, optional:
            This is the standard deviation of the noise that 
            is added to :code:`x` when it is corrupted.
            Defaults to :code:`0.1`.
        
        
        """

        assert axis in [
            "x",
            "y",
            "both",
        ], "Please ensure that the axis is from ['x', 'y', 'both']"

        self._dataset = dataset
        self._axis = axis
        self._x_noise_std = x_noise_std

        # setting the list of corrupt sources
        if corrupt_sources is None:
            self._corrupt_sources = []
        elif type(corrupt_sources) == int:
            self._corrupt_sources = [corrupt_sources]
        elif hasattr(corrupt_sources, "__iter__"):
            self._corrupt_sources = corrupt_sources
        else:
            raise TypeError(
                "Please ensure that corrupt_sources is an integer, iterable or None."
            )

        # setting the noise level
        if noise_level is None:
            self._noise_level = [0] * len(self._corrupt_sources)
        elif type(noise_level) == float:
            self._noise_level = [noise_level] * len(self._corrupt_sources)
        elif hasattr(noise_level, "__iter__"):
            if hasattr(noise_level, "__len__"):
                if hasattr(self._corrupt_sources, "__len__"):
                    assert len(noise_level) == len(self._corrupt_sources), (
                        "Please ensure that the noise level "
                        "is the same length as the corrupt sources. "
                        f"Expected {len(self._corrupt_sources)} noise levels, "
                        f"got {len(noise_level)}."
                    )
            self._noise_level = noise_level
        else:
            raise TypeError(
                "Please ensure that the noise level is a float, iterable or None"
            )
        self._noise_level = {
            cs: nl for cs, nl in zip(self._corrupt_sources, self._noise_level)
        }

        if seed is None:
            rng = np.random.default_rng(None)
            seed = rng.integers(low=1, high=1e9, size=1)[0]
        self.rng = np.random.default_rng(seed)

        self._corrupt_datapoints = {"x": {}, "y": {}}

        return

    def _corrupt_x(self, index, x, y, s):
        if index in self._corrupt_datapoints["x"]:
            x = self._corrupt_datapoints["x"][index]
        else:
            g_seed_mask, g_seed_values, class_seed = self.rng.integers(
                low=1, high=1e9, size=3
            )
            self.rng = np.random.default_rng(class_seed)
            g_values = torch.Generator(device=y.device).manual_seed(int(g_seed_values))
            g_mask = torch.Generator(device=y.device).manual_seed(int(g_seed_mask))
            mask = int(
                torch.rand(size=(), generator=g_mask, device=x.device)
                > 1 - self._noise_level[s]
            )
            values = torch.normal(
                mean=0,
                std=self._x_noise_std,
                generator=g_values,
                size=x.size(),
                device=x.device,
            )
            x = x + mask * values
            self._corrupt_datapoints["x"][index] = x
        return x, y, s

    def _corrupt_y(self, index, x, y, s):
        if index in self._corrupt_datapoints["y"]:
            y = self._corrupt_datapoints["y"][index]
        else:
            g_seed_mask, class_seed = self.rng.integers(low=1, high=1e9, size=2)
            self.rng = np.random.default_rng(class_seed)
            g_mask = torch.Generator().manual_seed(int(g_seed_mask))
            if torch.rand(size=(), generator=g_mask) > 1 - self._noise_level[s]:
                y = torch.tensor(1, dtype=y.dtype, device=y.device) - y

            self._corrupt_datapoints["y"][index] = y

        return x, y, s

    @property
    def corrupt_sources(self):
        return self._corrupt_sources

    def __getitem__(self, index):
        x, y, s = self._dataset[index]
        if s in self._noise_level:
            if self._axis == "x" or self._axis == "both":
                x, y, s = self._corrupt_x(index, x, y, s)
            if self._axis == "y" or self._axis == "both":
                x, y, s = self._corrupt_y(index, x, y, s)
        return x, y, s

    def __len__(
        self,
    ):
        return len(self._dataset)


# metrics


def auc_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label=None,
    sample_weight=None,
) -> float:
    """
    A function that calculates the area under
    the precision-recall curve
    between two arrays. This is modelled
    on the Scikit-Learn :code:`recall_score`,
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> auc_precision_recall_curve(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_proba=np.array([0,0,0,1,0]),
        ...     )
        0.85


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_proba: np.ndarray:
        The array of predicted score values. If :code:`y_pred`
        has shape :code:`(N,2)`, and :code:`y_true` has two unique
        values, then the probability of a positive class will
        be assumed to be :code:`y_proba[:,1]`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'`
        and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    Returns
    ---------

    - auc: float:
        The area under the precision-recall curve.

    """

    if len(y_proba.shape) == 2:
        if len(np.unique(y_true)) == 2:
            y_proba = y_proba[:, 1]

    y, x, _ = precision_recall_curve(
        y_true,
        y_proba,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )

    return auc(x, y)


class AUCBinaryPrecisionRecallMetric:
    def __init__(self):
        self.scores = []
        self.targets = []

    def update(self, scores, targets):
        self.scores.append(scores)
        self.targets.append(targets)

    def compute_key_value(self):
        scores = torch.cat(self.scores)
        targets = torch.cat(self.targets)
        auc = auc_precision_recall_curve(
            targets.cpu().detach().numpy(),
            scores.cpu().detach().numpy(),
        )
        return {"auc_pr": auc}


def train_batch(
    model,
    optimiser,
    x,
    y,
    sources,
    label_loss_weighting,
    device,
    writer=None,
):
    model.to(device)
    model.train()

    optimiser.zero_grad()

    label_loss, outputs = model(
        x,
        y,
        return_loss=True,
    )

    if label_loss_weighting is not None:
        label_loss = label_loss_weighting(
            losses=label_loss, sources=sources, writer=writer, writer_prefix="label"
        )

    loss = torch.mean(label_loss)
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
    scheduler=None,
    writer=None,
):
    model.to(device)

    model.train()

    train_loss = 0
    train_total = 0
    acc_meter = AccuracyMetric(topk=[1], compute_on_call=False)
    auc_meter = AUCMetric(compute_per_class_metrics=True, compute_on_call=False)
    bprf_meter = BinaryPrecisionRecallF1Metric(compute_on_call=False)
    auc_pr_meter = AUCBinaryPrecisionRecallMetric()

    lr = optimiser.param_groups[0]["lr"]
    pbar = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch_number+1}, LR {lr:.4f}")

    for batch_idx, (inputs, targets, sources) in enumerate(pbar):
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
            writer=writer,
        )

        if writer is not None:
            writer.add_scalar(
                "Train Loss",
                loss,
                epoch_number * len(train_loader) + batch_idx,
            )

        acc_meter.update(outputs, targets)
        auc_meter.update(outputs, targets)
        bprf_meter.update(outputs.argmax(1), targets)
        auc_pr_meter.update(outputs, targets)

        train_loss += loss * targets.size(0)
        train_total += targets.size(0)

        precision_recall_f1 = bprf_meter.compute_key_value()

        pbar.set_postfix(
            {
                "Loss": train_loss / train_total,
            }
        )

    if scheduler is not None:
        scheduler.step()

    metrics = {
        "Loss": train_loss / train_total,
        "Accuracy": acc_meter.compute_key_value()["accuracy01"],
        "AUC": auc_meter.compute_key_value()["auc/class_01"],
        "AUC PR": auc_pr_meter.compute_key_value()["auc_pr"],
        "Precision": precision_recall_f1["precision"],
        "Recall": precision_recall_f1["recall"],
        "F1": precision_recall_f1["f1"],
    }

    pbar.set_postfix(metrics)
    pbar.close()

    return train_loss / train_total, metrics


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    acc_meter = AccuracyMetric(topk=[1])
    auc_meter = AUCMetric(compute_per_class_metrics=True)
    bprf_meter = BinaryPrecisionRecallF1Metric()
    auc_pr_meter = AUCBinaryPrecisionRecallMetric()

    with torch.no_grad():
        test_loss = 0
        test_total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            label_loss, outputs = model(
                inputs,
                y=targets,
                return_loss=True,
            )

            loss = torch.mean(label_loss)
            acc_meter.update(outputs, targets)
            auc_meter.update(outputs, targets)
            bprf_meter.update(outputs.argmax(1), targets)
            auc_pr_meter.update(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)

        precision_recall_f1 = bprf_meter.compute_key_value()
        metrics = {
            "Loss": test_loss / test_total,
            "Accuracy": acc_meter.compute_key_value()["accuracy01"],
            "AUC": auc_meter.compute_key_value()["auc/class_01"],
            "AUC PR": auc_pr_meter.compute_key_value()["auc_pr"],
            "Precision": precision_recall_f1["precision"],
            "Recall": precision_recall_f1["recall"],
            "F1": precision_recall_f1["f1"],
        }

        print(
            "Test Loss",
            test_loss / test_total,
            "Test Acc",
            metrics["Accuracy"],
            "Test AUC",
            metrics["AUC"],
            "Test AUC PR",
            metrics["AUC PR"],
            "Test Precision",
            metrics["Precision"],
            "Test Recall",
            metrics["Recall"],
            "Test F1",
            metrics["F1"],
        )

    return test_loss / test_total, metrics


for dataset_name in DATASET_NAMES:
    for nc in N_CORRUPT_SOURCES:
        for run in RUNS:
            print(
                f"for dataset {dataset_name}, no. corrupt sources {nc}",
                f"and run {run} the following depression has been completed",
                results[dataset_name][nc][run].keys(),
            )


train_dataset = PTB_XL(
    data_path=DATA_DIR,
    train=True,
    source_name="nurse",
    sampling_rate=SAMPLING_RATE,
    return_sources=True,
    binary=True,
    subset=DATA_SUBSET,
)

test_dataset = PTB_XL(
    data_path=DATA_DIR,
    train=False,
    source_name="nurse",
    sampling_rate=SAMPLING_RATE,
    return_sources=False,
    binary=True,
    subset=DATA_SUBSET,
)

train_dataset = ToMemory(train_dataset)
test_dataset = ToMemory(test_dataset)


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

                unique_sources = np.unique(train_dataset.dataset.sources)
                corrupt_sources = np.random.default_rng(exp_seed).choice(
                    unique_sources, nc, replace=False
                )
                print("Corrupt Sources:", corrupt_sources)

                train_dataset_exp = ECGCorruptor(
                    dataset=train_dataset,
                    corrupt_sources=corrupt_sources,
                    noise_level=corruption_level,
                    seed=exp_seed,
                    axis=AXIS,
                    x_noise_std=CORRUPTION_STD,
                )

                model = ResNet1D_PTBXL()

                optimiser = torch.optim.Adam(
                    params=model.parameters(),
                    lr=LR,
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

                train_sources = np.array(train_dataset.dataset.sources)

                weights = compute_sample_weight(
                    class_weight="balanced",
                    y=train_sources,
                )

                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.from_numpy(weights),
                    num_samples=len(train_sources),
                    replacement=True,
                )

                train_loader = torch.utils.data.DataLoader(
                    train_dataset_exp,
                    batch_size=BATCH_SIZE,
                    sampler=sampler,
                )

                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                corrupt_sources = [s for s in train_dataset_exp.corrupt_sources]

                results_this_train = {}

                for epoch in range(N_EPOCHS):
                    train_loss, train_metrics = train_epoch(
                        model=model,
                        train_loader=train_loader,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        device=DEVICE,
                        epoch_number=epoch,
                        label_loss_weighting=label_loss_weighting,
                        writer=writer,
                    )
                    test_loss, test_metrics = test(model, test_loader, DEVICE)

                    if writer is not None:
                        writer.add_scalar("Test Loss", test_loss, epoch)
                        writer.add_scalar("Test Acc", test_metrics["Accuracy"], epoch)

                    results_this_train[epoch] = {
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }
                    results_this_train[epoch].update(
                        {f"train_{k}": v for k, v in train_metrics.items()}
                    )
                    results_this_train[epoch].update(
                        {f"test_{k}": v for k, v in test_metrics.items()}
                    )

                results_this_train["corrupt_sources"] = corrupt_sources

                results[dataset_name][nc][run][depression] = results_this_train

                writer.flush()
                writer.close()

                # save results to json

                with open(RESULTS_FILE, "w") as fp:
                    json.dump(results, fp)
