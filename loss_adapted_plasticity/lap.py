import math
import numpy as np
import logging
import typing
import torch
import torch.nn as nn


def weighted_avg_and_std(
    values: np.array, weights: np.array
) -> typing.Tuple[np.array, np.array]:
    """
    Return the weighted average and standard deviation.

    Arguments
    ---------

    - values: np.array:
        The array containing the values to
        calculate the mean and std on.

    - weights: np.array:
        The weights used in the mean and std.


    Returns
    ---------

    - out: typing.Tuple[np.array, np.array]:
        The weighted mean and std.

    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


class DiscreteRankingSTD(object):
    def __init__(
        self,
        function=lambda x: x,
        discrete_amount: float = 0.005,
        warm_up: int = 0,
        leniency: float = 1.0,
    ):
        """
        This class calculates the level of depression
        to apply to gradient updates from a batch of
        source data.



        Arguments
        ---------

        - function: _type_, optional:
            This argument allows you to apply a function
            to the value before it is returned.
            Defaults to :code:`lambda x: x`.

        - discrete_amount: float, optional:
            The step size used when calculating the depression.
            Defaults to :code:`0.005`.

        - warm_up: int, optional:
            The number of calls to this function before
            depression will start. Until depression starts,
            this function will return 0 on each call.
            Defaults to :code:`0`.

        - leniency: float, optional:
            The number of standard deviations away from the
            mean loss a mean source loss has to be
            before depression is applied.
            Defaults to :code:`1.0`.


        """
        self.function = function
        self.discrete_amount = discrete_amount
        self.source_xn = np.asarray([])
        self.warm_up = warm_up
        self.leniency = leniency
        self.step = 0
        return

    def __call__(
        self, loss_array: np.ndarray, source_idx: int, *args, **kwargs
    ) -> float:
        """

        Arguments
        ---------

        - loss_array: np.ndarray:
            The loss values for the last n batches of each source.
            Where n is the history size.
            This should be of shape (n_sources, n_batches_prev_tracked).

        - source_idx: int:
            The index in the loss array of the source
            being updated.


        Returns
        --------

        - out: float:
            The depression value, d in the depression calculation:
            dep = 1-tanh(m*d)**2.
            This means, the larger the value, the more depression
            will be applied during training.

        """
        # increasing step and checking if the hold off time has passed.
        self.step += 1
        if self.step < self.warm_up:
            return 0

        logging.debug("Source Index {}".format(source_idx))

        # keeps track of the current depression applied to each source
        # these will be used as weights in the standard deviation and
        # mean calculations
        if len(loss_array) > len(self.source_xn):
            self.source_xn = np.hstack(
                [self.source_xn, np.zeros(len(loss_array) - len(self.source_xn))]
            )

        # mask is True where loss array source is not equal to the current source
        mask = np.ones(loss_array.shape[0], dtype=bool)
        mask[source_idx] = False

        # if the range in loss values is close to 0, return no depression
        if np.all(np.isclose(np.ptp(loss_array[mask]), 0)):
            return 0

        # mean loss of current source
        mean_source_loss = np.mean(loss_array[~mask])

        # weighted mean and standard deviation of the sources other
        # than the current source.
        weights = np.ones_like(loss_array) / ((self.source_xn + 1)[:, np.newaxis])
        (mean_not_source_loss, std_not_source_loss) = weighted_avg_and_std(
            loss_array[mask], weights=weights[mask]
        )

        # calculates whether to trust a source more or less
        logging.debug(
            "{} < {}".format(
                mean_source_loss,
                mean_not_source_loss + self.leniency * std_not_source_loss,
            )
        )
        if (
            mean_source_loss
            < mean_not_source_loss + self.leniency * std_not_source_loss
        ):
            movement = -1
        else:
            movement = 1
        logging.debug("movement {}".format(movement))
        logging.debug("source_xn {}".format(self.source_xn[source_idx]))
        # moving the current trust level depending on the movement calculated above
        self.source_xn[source_idx] += movement
        if self.source_xn[source_idx] < 0:
            self.source_xn[source_idx] = 0

        # calculating the depression value
        depression = self.function(self.discrete_amount * self.source_xn[source_idx])

        return depression


class SourceGradientWeighting(object):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        history_length: int = 10,
        depression_strength: float = 1.0,
        depression_function="discrete_ranking_std",
        depression_function_kwargs: dict = {},
        source_is_bool: bool = False,
        **opt_kwargs,
    ):
        """
        Depression won't be applied until at least :code:`history_length` loss values
        have been collected for at least two sources. This could be
        longer if a :code:`warm_up` parameter is used in the depression function.

        This class will wrap any optimiser and perform lap gradient depression
        before the values are passed to the underlying optimiser.

        This implementation requires each batch to contain a single source.


        Examples
        ---------

        The following wraps the Adam optimiser with the LAP functionality.

        .. code-block::

            >>> optimizer = SourceGradientWeighting(
            ...     torch.optim.Adam, params=model.parameters(), lr=0.01,
            ...     )

        Ensure that when using this optimiser, during the :code:`.step`
        method, you use the arguments :code:`loss` and :code:`source`.
        For example::

            >>> loss = loss.backward()
            >>> optimizer.step(loss, source)


        Arguments
        ---------

        - optimizer: torch.optim.Optimizer:
            The optimizer to wrap with the LAP algorithm.

        - history_length: int, optional:
            The number of previous loss values for each source
            to be used in the loss adapted plasticity
            calculations.
            Defaults to :code:`10`.

        - depression_strength: float:
            This float determines the strength of the depression
            applied to the gradients. It is the value of :code:`m` in
            :code:`dep = 1-tanh(m*d)**2`.
            Defaults to :code:`1`.

        - depression_function: function or string, optional:
            This is the function used to calculate the depression
            based on the loss array (with sources containing full
            loss history) and the source of the current batch.
            Ensure that the first two arguments of this function are
            loss_array and source_idx.
            If string, please ensure it is :code:`'discrete_ranking_std'`
            Defaults to :code:`'discrete_ranking_std'`.

        - depression_function_kwargs: dict, optional:
            Keyword arguments that will be used in :code:`depression_function`
            when initiating it, if it is specified by a string.
            Defaults to :code:`{}`.

        - source_is_bool: bool, optional:
            This tells the optimizer that the sources will be named True
            when the source is corrupted and False if the source is not.
            If the incoming source is corrupted, then the optimizer will not
            make a step.
            Defaults to :code:`False`.

        """

        if (not 0 <= history_length) and (type(history_length) == int):
            raise ValueError(
                "Invalid parameter for history_length: {}. "
                "Please use an integer larger than 0".format(history_length)
            )
        if not 0.0 <= depression_strength:
            raise ValueError(
                "Invalid depression stregnth: {}".format(depression_strength)
            )

        self.optimizer = optimizer(**opt_kwargs)

        # storing settings and creating the loss array
        self.history_length = history_length
        self.loss_array = -1 * np.ones((1, self.history_length))
        self.source_dict = {}
        self.n_sources = 0
        self.depression_strength = depression_strength
        self.depression_function_kwargs = depression_function_kwargs
        self.depression_function = (
            depression_function
            if not type(depression_function) == str
            else self._get_depression_function(depression_function)
        )
        self.source_step_dict = {}
        self.source_is_bool = source_is_bool

        return

    def _has_complete_history(self):
        # returns source indices in which there is a complete history of losses
        return np.argwhere(
            np.sum(self.loss_array != -1, axis=1) == self.history_length
        ).reshape(-1)

    def _get_depression_function(self, name):
        """
        Function to get the drepression function by name.
        """
        if name == "discrete_ranking_std":
            return DiscreteRankingSTD(**self.depression_function_kwargs)

        else:
            raise NotImplementedError(
                "{} is not a known depression function. Please "
                "pass the function instead of the name.".format(name)
            )

    @torch.no_grad()
    def step(
        self,
        loss: float,
        source: typing.Hashable,
        override_dep: typing.Union[bool, None] = None,
        writer=None,
        **kwargs,
    ):
        """
        Performs a single optimization step.

        Arguments
        ---------

        - loss: float:
            This is the loss value that is used in the depression calculations.

        - source: hashable:
            This is the source name that is used to
            store the loss values for the different sources.

        - override_dep: bool or None:
            If None, then whether to apply depression will be decided
            based on the logic of this class. If True, then depression will
            be applied. This might cause unexpected results if there is no depression value
            calculated based on whether there is enough data available in the
            .loss_array. In this case, not depression is applied.
            If False, then depression will not be applied.
            This is mostly useful as an option to turn off LAP.
            Defaults to :code:`None`.

        - writer: torch.utils.tensorboard.SummaryWriter:
            A tensorboard writer can be passed into this function to track metrics.
            Defaults to :code:`None`.

        """

        logging.debug("source, {}".format(source))
        logging.debug("loss, {}".format(loss))

        # if reliability of source is given, update only when
        # data is reliable
        if self.source_is_bool:
            if source:
                return None
            else:
                if not override_dep in [True, False]:
                    override_dep = False

        # building the loss array
        if not source in self.source_dict:
            # if new source, add row to the loss array
            self.source_dict[source] = self.n_sources
            self.n_sources += 1
            source_idx = self.source_dict[source]
            self.loss_array = np.concatenate(
                [self.loss_array, -1 * np.ones((1, self.history_length))], axis=0
            )
            self.loss_array[source_idx, -1] = loss
        else:
            # if already tracked source, move history along and add new loss value
            source_idx = self.source_dict[source]
            losses = self.loss_array[source_idx]
            losses[:-1] = losses[1:]
            losses[-1] = loss
            logging.debug("losses, {}".format(losses))
            logging.debug("loss array, {}".format(self.loss_array))

        # saves the number of times each source has been seen for summary writer
        if not source in self.source_step_dict:
            self.source_step_dict[source] = 0
        self.source_step_dict[source] += 1

        # finds sources that have a complete history of losses
        history_idx = self._has_complete_history()

        # if current source has full history and at least one other source does
        # then perform depression calculations
        if (len(history_idx) > 1) and (source_idx in history_idx):
            depressing = True
        else:
            depressing = False

        # calculate the depression value
        if depressing:
            depression = self.depression_function(
                loss_array=self.loss_array[history_idx],
                source_idx=np.argwhere(history_idx == source_idx).reshape(-1)[0],
            )
        logging.debug("depressing, {}".format(depressing))

        # depression boolean override from argument
        # if override is True and there is no depression value calculated
        # the then depression value is set to 0 (no depression)
        if not override_dep is None:
            if override_dep in [True, False]:
                if not depressing:
                    depression = 0.0
                depressing = override_dep
            else:
                raise TypeError(
                    "override_dep must be of boolean value, or None. Please see docs."
                )

        for group in self.optimizer.param_groups:
            params_with_grad = []

            # calculate the actual depression to be multiplied by the gradients
            if depressing:
                logging.debug("Depression, {}".format(depression))
                actual_depression = (
                    1
                    - torch.pow(
                        torch.tanh(torch.tensor(depression * self.depression_strength)),
                        2,
                    ).item()
                )
            else:
                actual_depression = 1

            # saves the depression value to the writer
            if not writer is None:
                writer.add_scalars(
                    "Actual Depression Value",
                    {"{}".format(source): actual_depression},
                    self.source_step_dict[source],
                )

            logging.debug("Actual Depression, {}".format(actual_depression))

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "This has not been designed for sparse "
                            "gradients and may not return expected results"
                        )

                    # ======= applying depression =======
                    p.grad.mul_(actual_depression)
                    # ===================================

            self.optimizer.step(**kwargs)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self.optimizer, name):
            return getattr(self.optimizer, name)
        else:
            raise AttributeError


class SourceLossWeighting(nn.Module):
    """
    This class calculates the loss weighting for each source
    based on the loss history of each source.
    It acts on the loss function.

    This is different from the LAP class, which acts on the gradients. This
    class weights the loss values.

    Examples
    ---------

    The following can be used to transform loss values as follows:


    .. code-block::

        >>> losses = loss_weighting(losses=losses, sources=sources)
        >>> loss = torch.mean(losses)
        >>> loss.backward()



    Arguments
    ---------
    - history_length: int, optional:
        The number of previous loss values for each source
        to be used in the loss adapted plasticity
        calculations.
        Defaults to :code:`10`.

    - warmup_iters: int, optional:
        The number of iterations before the loss weighting
        starts to be applied.
        Defaults to :code:`100`.

    - depression_strength: float, optional:
        This float determines the strength of the depression
        applied to the gradients. It is the value of :code:`m` in
        :code:`dep = 1-tanh(m*d)**2`.
        Defaults to :code:`1`.

    - discrete_amount: float, optional:
        The step size used when calculating the depression.
        Defaults to :code:`0.005`.

    - leniency: float, optional:
        The number of standard deviations away from the
        mean loss a mean source loss has to be
        before depression is applied.
        Defaults to :code:`1.0`.

    """

    def __init__(
        self,
        history_length: int = 50,
        warmup_iters: int = 100,
        depression_strength: float = 1,
        discrete_amount: float = 0.005,
        leniency: float = 1.0,
        device="cpu",
    ):
        super().__init__()

        # options

        self.history_length = nn.Parameter(
            torch.tensor(history_length), requires_grad=False
        )
        self.warmup_iters = nn.Parameter(
            torch.tensor(warmup_iters), requires_grad=False
        )
        self.depression_strength = nn.Parameter(
            torch.tensor(depression_strength), requires_grad=False
        )
        self.discrete_amount = nn.Parameter(
            torch.tensor(discrete_amount), requires_grad=False
        )
        self.leniency = nn.Parameter(torch.tensor(leniency), requires_grad=False)

        # tracked values
        self.step_count = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.n_sources = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.source_order = nn.ParameterDict({})
        self.loss_history = nn.Parameter(
            torch.full((1, self.history_length + 1), float("nan")), requires_grad=False
        )
        self.sum_of_values = nn.Parameter(
            torch.full((1,), float("nan")), requires_grad=False
        )
        self.squared_sum_of_values = nn.Parameter(
            torch.full((1,), float("nan")), requires_grad=False
        )
        self.source_unreliability = nn.Parameter(
            torch.full((1,), 0.0), requires_grad=False
        )

        self.to(device)

    @staticmethod
    def shift_array_values(inputs, shifts: torch.LongTensor):
        n_rows, n_cols = inputs.shape
        arange_original = (
            torch.arange(n_cols, device=shifts.device)
            .view((1, n_cols))
            .repeat((n_rows, 1))
        )
        arange_new = (arange_original - shifts.reshape(-1, 1)) % n_cols

        return inputs[torch.arange(n_rows).reshape(-1, 1), arange_new]

    @staticmethod
    def _has_complete_history(loss_history):
        """
        Returns True if the history for at least two sources is complete.
        """
        return ~torch.isnan(loss_history[:, 1:]).any(dim=1)

    def _update_source_unreliability(
        self,
        sources_update_idx,
        sum_of_values,
        squared_sum_of_values,
        N,
        source_unreliability,
    ):
        """
        Updates the unreliability of the source with the given index and
        array of loss history sums and squarred sums.
        """

        # make sure new tensors are on the same device
        device = sum_of_values.device

        # get the indices of the sources that are updated
        source_update_bool = torch.arange(
            sum_of_values.shape[0], device=device
        ) == sources_update_idx.reshape(-1, 1)
        source_update_bool = source_update_bool[source_update_bool.any(dim=1)]

        # expand the weights and the sum of values to the same shape
        weights_expanded = torch.reshape(
            (
                1
                - torch.pow(
                    torch.tanh(
                        self.discrete_amount
                        * self.depression_strength
                        * source_unreliability
                    ),
                    2,
                )
            ).expand(source_update_bool.shape[0], source_unreliability.shape[0])[
                ~source_update_bool
            ],
            (source_update_bool.shape[0], source_unreliability.shape[0] - 1),
        )
        sum_of_values_expanded = torch.reshape(
            (sum_of_values).expand(source_update_bool.shape[0], sum_of_values.shape[0])[
                ~source_update_bool
            ],
            (source_update_bool.shape[0], sum_of_values.shape[0] - 1),
        )
        squared_sum_of_values_expanded = torch.reshape(
            (squared_sum_of_values).expand(
                source_update_bool.shape[0], squared_sum_of_values.shape[0]
            )[~source_update_bool],
            (source_update_bool.shape[0], squared_sum_of_values.shape[0] - 1),
        )

        # calculate the mean and std
        mean_source_loss = sum_of_values[sources_update_idx] / N
        mean_not_source_loss = torch.sum(
            sum_of_values_expanded * weights_expanded, axis=1
        ) / (torch.sum(weights_expanded, axis=1) * N)
        std_not_source_loss = torch.sqrt(
            torch.sum(squared_sum_of_values_expanded * weights_expanded, axis=1)
            / (torch.sum(weights_expanded, axis=1) * N)
            - mean_not_source_loss**2
        )

        stable_source = ~torch.isclose(
            std_not_source_loss, torch.tensor(0, device=device).float()
        )

        movement = torch.zeros_like(stable_source).float()

        movement[stable_source] = torch.where(
            mean_source_loss[stable_source]
            < mean_not_source_loss[stable_source]
            + self.leniency * std_not_source_loss[stable_source],
            -1,
            +1,
        ).float()

        source_unreliability[sources_update_idx] += movement
        source_unreliability = torch.clamp(source_unreliability, min=0.0)

        return source_unreliability

    def get_source_unrelaibility(self):
        source_list = list(self.source_order.keys())
        # sorted source list
        source_list = sorted(source_list)
        source_idx = []
        for source in source_list:
            source_idx.append(self.source_order[source])

        return [self.source_unreliability[s_i].item() for s_i in source_idx]

    def get_source_order(self):
        source_list = list(self.source_order.keys())
        # sorted source list
        source_list = sorted(source_list)
        return source_list

    def forward(
        self,
        losses: torch.tensor,
        sources: torch.tensor,
        writer=None,
        writer_prefix: typing.Optional[str] = None,
    ):
        """
        Arguments
        ----------

        - losses: torch.Tensor of shape (batch_size,):
            The losses for each example in the batch.

        - sources: torch.Tensor of shape (batch_size,):
            The source for each example in the batch.

        - writer: torch.utils.tensorboard.SummaryWriter, optional:
            A tensorboard writer can be passed into this function to track metrics.
            Defaults to :code:`None`.

        - writer_prefix: str, optional:
            A prefix to add to the writer metrics.
            Defaults to :code:`None`.

        Returns
        --------

        - output: torch.Tensor of shape (batch_size,):
            The weighted losses for each example in the batch.


        """

        input_device = losses.device

        device = self.loss_history.device

        losses = losses.to(device)
        sources = sources.to(device)

        # self.loss_history = self.loss_history.to(device)
        # self.source_unreliability = self.source_unreliability.to(device)
        # self.sum_of_values = self.sum_of_values.to(device)
        # self.squared_sum_of_values = self.squared_sum_of_values.to(device)

        unique_sources = torch.unique(sources)
        to_sum = (sources == unique_sources.reshape(-1, 1)).float()
        mean_source_loss = to_sum @ losses / to_sum.sum(dim=1)

        # update the source_order dict
        for source in unique_sources:
            source = str(source.item())
            # adding new sources to the tracker, row to the loss history,
            # and a new entry to the source_unreliability tensor
            # and sum_of_values and squared_sum_of_values
            if source not in self.source_order:
                self.source_order[source] = self.n_sources.item()
                self.n_sources += 1
                self.loss_history = nn.Parameter(
                    torch.cat(
                        (
                            self.loss_history,
                            torch.full(
                                (1, self.history_length + 1),
                                float("nan"),
                                device=device,
                            ),
                        ),
                        dim=0,
                    ),
                    requires_grad=False,
                )
                self.source_unreliability = nn.Parameter(
                    torch.cat(
                        (self.source_unreliability, torch.zeros(1, device=device)),
                        dim=0,
                    ),
                    requires_grad=False,
                )
                self.sum_of_values = nn.Parameter(
                    torch.cat(
                        (
                            self.sum_of_values,
                            torch.full((1,), float("nan"), device=device),
                        ),
                        dim=0,
                    ),
                    requires_grad=False,
                )
                self.squared_sum_of_values = nn.Parameter(
                    torch.cat(
                        (
                            self.squared_sum_of_values,
                            torch.full((1,), float("nan"), device=device),
                        ),
                        dim=0,
                    ),
                    requires_grad=False,
                )

        source_idx = torch.tensor(
            [self.source_order[str(s.item())] for s in unique_sources]
        )

        # update the loss history
        shifts = torch.zeros(self.loss_history.shape[0]).long()
        shifts[source_idx] = -1

        # shift the loss history
        self.loss_history = nn.Parameter(
            self.shift_array_values(self.loss_history, shifts=shifts),
            requires_grad=False,
        )

        # update the loss history with the mean loss for each source given
        self.loss_history[source_idx, -1] = mean_source_loss

        # get the loss history for the sources with complete history
        complete_history = self._has_complete_history(self.loss_history)

        source_idx_to_update = torch.tensor(
            [
                self.source_order[str(s.item())]
                for s in unique_sources
                if complete_history[self.source_order[str(s.item())]]
            ]
        )

        depression_values = torch.zeros(
            self.loss_history.shape[0], device=device
        ).float()

        if (
            (sum(complete_history) >= 2)
            and (self.step_count >= self.warmup_iters)
            and len(source_idx_to_update) > 0
        ):
            loss_history_full = self.loss_history[complete_history]
            source_unreliability_full = self.source_unreliability[complete_history]

            sum_of_values = self.sum_of_values[complete_history]
            squared_sum_of_values = self.squared_sum_of_values[complete_history]

            # if the sum_of_values and squared_sum_of_values is nan
            # at a given index, set it to the sum of the loss history from [1:]
            # and the squared sum of the loss history from [1:]
            # if not nan, then take away the first value and add the last value
            nan_idx = torch.isnan(sum_of_values)
            sum_of_values = torch.where(
                nan_idx, loss_history_full[:, 1:].sum(axis=1), sum_of_values
            )
            squared_sum_of_values = torch.where(
                nan_idx,
                (loss_history_full[:, 1:] ** 2).sum(axis=1),
                squared_sum_of_values,
            )

            sources_complete_idx_to_update = (
                torch.cumsum(complete_history, dim=0)[source_idx_to_update] - 1
            )

            # but we do not want to update the sums if there was no
            # change in the loss history or if we just summed them
            # above
            sums_idx_to_update = sources_complete_idx_to_update[
                ~nan_idx[sources_complete_idx_to_update]
            ]

            sum_of_values[sums_idx_to_update] = (
                sum_of_values[sums_idx_to_update]
                - loss_history_full[sums_idx_to_update, 0]
                + loss_history_full[sums_idx_to_update, -1]
            )
            squared_sum_of_values[sums_idx_to_update] = (
                squared_sum_of_values[sums_idx_to_update]
                - loss_history_full[sums_idx_to_update, 0] ** 2
                + loss_history_full[sums_idx_to_update, -1] ** 2
            )

            source_unreliability_full = self._update_source_unreliability(
                sources_complete_idx_to_update,
                sum_of_values,
                squared_sum_of_values,
                self.history_length,
                source_unreliability_full,
            )

            source_unreliability_full_to_update = source_unreliability_full[
                sources_complete_idx_to_update
            ]

            self.sum_of_values[complete_history] = sum_of_values
            self.squared_sum_of_values[complete_history] = squared_sum_of_values

            self.source_unreliability[source_idx_to_update] = (
                source_unreliability_full_to_update
            )
            depression = self.discrete_amount * source_unreliability_full_to_update

            depression_values[source_idx_to_update] = depression

        multiplier = 1 - torch.pow(
            torch.tanh(depression_values * self.depression_strength), 2
        )

        multiplier_out = torch.zeros_like(losses)
        for s, s_idx in zip(unique_sources, source_idx):
            multiplier_out[sources == s] = multiplier[s_idx]

        # logging the unreliability of each source
        if writer is not None:
            for source in unique_sources:
                source = source.item()
                source_idx = self.source_order[str(source)]
                # write the reliability of the source to tensorboard
                writer.add_scalar(
                    (
                        f"unreliability/source_{source}"
                        if writer_prefix is None
                        else f"{writer_prefix}/unreliability/source_{source}"
                    ),
                    self.source_unreliability[source_idx].cpu().item(),
                    self.step_count.cpu().item(),
                )
                # write the loss of the source to tensorboard
                writer.add_scalar(
                    (
                        f"depression_multiplier/source_{source}"
                        if writer_prefix is None
                        else f"{writer_prefix}/depression_multiplier/source_{source}"
                    ),
                    multiplier[self.source_order[str(source)]].cpu().item(),
                    self.step_count.cpu().item(),
                )
                # write the loss of the source to tensorboard
                writer.add_scalar(
                    (
                        f"loss/source_{source}"
                        if writer_prefix is None
                        else f"{writer_prefix}/loss/source_{source}"
                    ),
                    torch.mean(losses[sources == source]).cpu().cpu().item(),
                    self.step_count.cpu().item(),
                )

        self.step_count += 1

        output = losses * multiplier_out

        self.sum_of_values = nn.Parameter(self.sum_of_values, requires_grad=False)
        self.squared_sum_of_values = nn.Parameter(
            self.squared_sum_of_values, requires_grad=False
        )
        self.source_unreliability = nn.Parameter(
            self.source_unreliability, requires_grad=False
        )
        self.loss_history = nn.Parameter(self.loss_history, requires_grad=False)

        return output.to(input_device)
