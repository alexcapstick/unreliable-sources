import torch
import torchvision
from torchvision import transforms
import numpy as np
import typing
import typing
import ast
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import os
from torchvision.datasets.utils import download_and_extract_archive
import joblib
import tqdm
from ..utils.utils import tqdm_style


try:
    import wfdb

    wfdb_import_error = False
except ImportError:
    wfdb_import_error = True


###### transformations


class FlattenImage(torch.nn.Module):
    def __init__(self):
        """
        Allows you to flatten an input to
        1D. This is useful in pytorch
        transforms when loading data.

        """
        super(FlattenImage, self).__init__()

    def forward(self, x):
        return x.reshape(-1)


def get_dataset_function(dataset_name):
    func_name = "get_{}".format(dataset_name)
    return globals()[func_name]


###### datasets


def get_mnist(path, return_targets=False):
    """
    Function to get the mnist data from pytorch with
    some transformations first.

    The returned MNIST data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``MNIST`.

    - return_targets: bool`, (optional):
        Whether to return the targets along with the
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_mnist: torch.utils.data.Dataset`

        - test_mnist: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_mnist_targets: torch.tensor`
            - test_mnist_targets: torch.tensor`

    """
    transform_images = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=0, std=1),
            FlattenImage(),
        ]
    )

    train_mnist = torchvision.datasets.MNIST(
        root=path, download=True, train=True, transform=transform_images
    )

    test_mnist = torchvision.datasets.MNIST(
        root=path, download=True, train=False, transform=transform_images
    )
    if return_targets:
        train_mnist_targets = torch.tensor(np.asarray(train_mnist.targets).astype(int))
        test_mnist_targets = torch.tensor(np.asarray(test_mnist.targets).astype(int))

        return train_mnist, test_mnist, train_mnist_targets, test_mnist_targets

    return train_mnist, test_mnist


def get_fmnist(path, return_targets=False):
    """
    Function to get the FMNIST data from pytorch with
    some transformations first.

    The returned FMNIST data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``FashionMNIST`.

    - return_targets: bool`, (optional):
        Whether to return the targets along with the
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_fmnist: torch.utils.data.Dataset`

        - test_fmnist: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_fmnist_targets: torch.tensor`
            - test_fmnist_targets: torch.tensor`

    """
    transform_images = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=0, std=1),
            FlattenImage(),
        ]
    )

    train_fmnist = torchvision.datasets.FashionMNIST(
        root=path, download=True, train=True, transform=transform_images
    )

    test_fmnist = torchvision.datasets.FashionMNIST(
        root=path, download=True, train=False, transform=transform_images
    )
    if return_targets:
        train_fmnist_targets = torch.tensor(
            np.asarray(train_fmnist.targets).astype(int)
        )
        test_fmnist_targets = torch.tensor(np.asarray(test_fmnist.targets).astype(int))

        return train_fmnist, test_fmnist, train_fmnist_targets, test_fmnist_targets

    return train_fmnist, test_fmnist


def get_cifar10(path, return_targets=False):
    """
    Function to get the CIFAR 10 data from pytorch with
    some transformations first.

    The returned CIFAR 10 data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``cifar-10-batches-py`.

    - return_targets: bool`, (optional):
        Whether to return the targets along with the
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_cifar: torch.utils.data.Dataset`

        - test_cifar: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_cifar_targets: torch.tensor`
            - test_cifar_targets: torch.tensor`

    """
    transform_images = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ]
    )

    train_cifar = torchvision.datasets.CIFAR10(
        root=path, download=True, train=True, transform=transform_images
    )

    test_cifar = torchvision.datasets.CIFAR10(
        root=path, download=True, train=False, transform=transform_images
    )
    if return_targets:
        train_cifar_targets = torch.tensor(np.asarray(train_cifar.targets).astype(int))
        test_cifar_targets = torch.tensor(np.asarray(test_cifar.targets).astype(int))

        return train_cifar, test_cifar, train_cifar_targets, test_cifar_targets

    return train_cifar, test_cifar


def get_cifar100(path, return_targets=False):
    """
    Function to get the CIFAR 100 data from pytorch with
    some transformations first.

    The returned CIFAR 100 data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``cifar-100-python`.

    - return_targets: bool`, (optional):
        Whether to return the targets along with the
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_cifar: torch.utils.data.Dataset`

        - test_cifar: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_cifar_targets: torch.tensor`
            - test_cifar_targets: torch.tensor`

    """
    transform_images = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ]
    )

    train_cifar = torchvision.datasets.CIFAR100(
        root=path, download=True, train=True, transform=transform_images
    )

    test_cifar = torchvision.datasets.CIFAR100(
        root=path, download=True, train=False, transform=transform_images
    )

    if return_targets:
        train_cifar_targets = torch.tensor(np.asarray(train_cifar.targets).astype(int))
        test_cifar_targets = torch.tensor(np.asarray(test_cifar.targets).astype(int))

        return train_cifar, test_cifar, train_cifar_targets, test_cifar_targets

    return train_cifar, test_cifar


class WrapperDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        functions_index: typing.Union[typing.List[int], int, None] = None,
        functions: typing.Union[
            typing.Callable, typing.List[typing.Callable]
        ] = lambda x: x,
    ):
        """
        This allows you to wrap a dataset with a set of 
        functions that will be applied to each returned 
        data point. You can apply a single function to all 
        outputs of a data point, or a different function
        to each of the different outputs.
        
        
        
        Examples
        ---------

        The following would multiply all of the first returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=0,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=None,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 3.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=[lambda x: x*2, lambda x: x*3]
            ...     )
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to be wrapped.
        
        - functions_index: typing.Union[typing.List[int], int, None], optional:
            The index of the functions to be applied to. 

            - If :code:`None`, then if the :code:`functions` is callable, it \
            will be applied to all outputs of the data points, \
            or if the :code:`functions` is a list, it will be applied to the corresponding \
            output of the data point.

            - If :code:`list` then the corresponding index will have the \
            :code:`functions` applied to them. If :code:`functions` is a list, \
            then it will be applied to the corresponding indicies given in :code:`functions_index` \
            of the data point. If :code:`functions` is callable, it will be applied to all of the \
            indicies in :code:`functions_index`
        
            - If :code:`int`, then the :code:`functions` must be callable, and \
            will be applied to the output of this index.
            
            Defaults to :code:`None`.
        
        - functions: _type_, optional:
            This is the function, or list of functions to apply to the
            corresponding indices in :code:`functions_index`. Please
            see the documentation for the :code:`functions_index` argument
            to understand the behaviour of different input types. 
            Defaults to :code:`lambda x:x`.
        
        
        """

        self._dataset = dataset
        if functions_index is None:
            if type(functions) == list:
                self.functions = {fi: f for fi, f in enumerate(functions)}
            elif callable(functions):
                self.functions = functions
            else:
                raise TypeError(
                    "If functions_index=None, please ensure "
                    "that functions is a list or a callable object."
                )

        elif type(functions_index) == list:
            if type(functions) == list:
                assert len(functions_index) == len(
                    functions
                ), "Please ensure that the functions_index is the same length as functions."
                self.functions = {fi: f for fi, f in zip(functions_index, functions)}
            elif callable(functions):
                self.functions = {fi: functions for fi in functions_index}
            else:
                raise TypeError(
                    "If type(functions_index)==list, please ensure "
                    "that functions is a list of the same length or a callable object."
                )

        elif type(functions_index) == int:
            if callable(functions):
                self.functions = {functions_index: functions}
            else:
                raise TypeError(
                    "If type(functions_index)==int, please ensure "
                    "the functions is a callable object."
                )

        else:
            raise TypeError(
                "Please ensure that functions_index is a list, int or None."
            )

        return

    def __getitem__(self, index):
        if type(self.functions) == dict:
            return [
                self.functions.get(nout, lambda x: x)(out)
                for nout, out in enumerate(self._dataset[index])
            ]
        elif callable(self.functions):
            return [self.functions(out) for out in self._dataset[index]]
        else:
            raise TypeError("The functions could not be applied.")

    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        now: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        """
        This dataset allows the user
        to wrap another dataset and
        load all of the outputs into memory,
        so that they are accessed from RAM
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.


        Examples
        ---------

        .. code-block::

            >>> dataset = MemoryDataset(dataset, now=True)


        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset to wrap and add to memory.

        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.

        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.

        - n_jobs: int, optional:
            The number of parallel operations when loading
            the data to memory.
            Defaults to :code:`1`.


        """

        self._dataset = dataset
        self._data_dict = {}
        if now:
            pbar = tqdm.tqdm(
                total=len(dataset),
                desc="Loading into memory",
                disable=not verbose,
                smoothing=0,
                **tqdm_style
            )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend="threading",
            )(joblib.delayed(add_to_dict)(index) for index in index_list)

            pbar.close()

        return

    def __getitem__(self, index):
        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output

    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError
