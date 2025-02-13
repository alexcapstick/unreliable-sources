import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import numpy as np
from itertools import product
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import os
import sys

sys.path.append("../../")
from experiment_code.utils.utils import update_dict
from experiment_code.data_utils.dataloader_loaders import get_train_data, get_test_data
import argparse


if __name__ == "__main__":

    ####### possible arguments #######
    parser = argparse.ArgumentParser(
        description="Train a chosen model with different algorithms"
    )
    parser.add_argument(
        "--dataset-name", help="The dataset to use", type=str, default="cifar10"
    )
    parser.add_argument("--model-name", help="The model Name", required=True)
    parser.add_argument(
        "--model-dir",
        help="The directory to the model saves",
        type=str,
        default="./outputs/models/",
    )
    parser.add_argument(
        "--test-method", help="The testing method", type=str, default="traditional"
    )
    parser.add_argument("--seed", help="random seed", type=int, default=None)
    parser.add_argument(
        "--data-dir",
        help="Directory for the data to be saved and loaded",
        type=str,
        default="../.././data/",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Whether to print information as the script runs",
        action="store_true",
    )
    parser.add_argument(
        "--config-file",
        help="The config file containing the model parameters and training methods",
        type=str,
        default="../../synthetic_config.yaml",
    )
    parser.add_argument(
        "--n-sources",
        help="The number of sources used in the training data",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-corrupt-sources",
        help="The number of corrupt sources used in the training data",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--source-size",
        help="The number of data points in each source batch",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr",
        help="The learning rate of the training",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--depression-strength",
        help="The depression strength of the training",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lap-n",
        help="The number of previous losses to use in the depression ranking",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--strictness",
        help="The strictness used when calculating which sources to "
        "to apply depression to. It is used in mean+strictness*std.",
        nargs="+",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--n-epochs",
        help="The number of epochs to train for",
        nargs="+",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    # hparams

    hparam_short_name = {
        "n_sources": "ns",
        "n_corrupt_sources": "ncs",
        "source_size": "ssize",
        "lr": "lr",
        "depression_strength": "ds",
        "lap_n": "lap_n",
    }

    hparam_list_list = []
    hparam_arg_names = [
        "n_sources",
        "n_corrupt_sources",
        "source_size",
        "lr",
        "depression_strength",
        "lap_n",
    ]
    for arg_name in hparam_arg_names:
        hparam_list = getattr(args, arg_name)
        if hparam_list is None:
            continue
        else:
            hparam_list_list.append([(arg_name, hparam) for hparam in hparam_list])
    hparam_runs = list(product(*hparam_list_list))
    hparam_runs
    hparams = hparam_runs[0]

    ####### Running with seed #######
    if args.verbose:
        print(" --------- Running with seed {} --------- ".format(args.seed))

    torch.manual_seed(args.seed)

    ####### configs from files #######
    model_config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)[
        args.model_name
    ]

    for hparam in hparams:
        param_name, param = hparam
        if param_name in ["n_sources", "n_corrupt_sources", "source_size"]:
            update_dict(model_config, {"train_params": {param_name: param}})

        elif param_name in ["lr", "depression_strength", "lap_n"]:
            for optim_name in model_config["model_params"]["train_optimizer"].keys():
                update_dict(
                    model_config,
                    {
                        "model_params": {
                            "train_optimizer": {optim_name: {param_name: param}}
                        }
                    },
                )

        else:
            raise NotImplementedError(
                "Please point the hparam to the correct model_config param."
            )

        if param_name in hparam_short_name:
            model_config["model_name"] += "-{}_{}".format(
                hparam_short_name[param_name], param
            )
        else:
            model_config["model_name"] += "-{}_{}".format(param_name, param)

    if args.verbose:
        print(" --------- Extracting the data --------- ")

    ####### collating training data #######
    train_loaders = get_train_data(args, model_config)
    test_config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)[
        "testing-procedures"
    ][args.test_method]
    test_loader, test_targets = get_test_data(args, test_config=test_config)
    train_dataset = {}
    source_list = []

    for data in train_loaders[0]:
        source = str(data[2][0].item())
        if args.dataset_name in ["cifar10", "cifar100"]:
            x = data[0].numpy().transpose(0, 2, 3, 1)
            y = data[1].numpy()
        else:
            x = data[0].numpy()
            y = data[1].numpy()

        if not source in train_dataset:
            train_dataset[source] = {"x": [x], "y": [y]}
            source_list.append(source)
        else:
            train_dataset[source]["x"].append(x)
            train_dataset[source]["y"].append(y)

    for source in source_list:
        train_dataset[source]["x"] = np.vstack(train_dataset[source]["x"])
        train_dataset[source]["y"] = np.hstack(train_dataset[source]["y"])
    test_dataset = {}

    for data in test_loader:
        inputs = data[0]
        targets = data[1].unsqueeze(1)
        for nd, (input, target) in enumerate(zip(inputs, targets)):
            input = input.unsqueeze(0)
            nd = nd % 10
            source = str(source_list[nd])
            if args.dataset_name in ["cifar10", "cifar100"]:
                x = input.numpy().transpose(0, 2, 3, 1)
                y = target.numpy()
            else:
                x = input.numpy()
                y = target.numpy()

            if not source in test_dataset:
                test_dataset[source] = {"x": [x], "y": [y]}

            else:
                test_dataset[source]["x"].append(x)
                test_dataset[source]["y"].append(y)

    for source in source_list:
        test_dataset[source]["x"] = np.vstack(test_dataset[source]["x"])
        test_dataset[source]["y"] = np.hstack(test_dataset[source]["y"])

    if args.dataset_name in ["cifar10", "cifar100"]:
        folder_data_save = "cifar10"
    else:
        folder_data_save = args.dataset_name

    if os.path.exists(
        os.path.join(f"./arfl-master/data/{folder_data_save}/data/", "data_cache.obj")
    ):
        os.remove(
            os.path.join(
                f"./arfl-master/data/{folder_data_save}/data/", "data_cache.obj"
            )
        )

    with open(
        os.path.join(f"./arfl-master/data/{folder_data_save}/data/", "data_cache.obj"),
        "wb",
    ) as f:
        pickle.dump(source_list, f)
        pickle.dump(train_dataset, f)
        pickle.dump(source_list, f)
        pickle.dump(test_dataset, f)
