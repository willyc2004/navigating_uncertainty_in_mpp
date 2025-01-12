from __future__ import annotations

import torch
import torch.nn.functional as F
import math

# Typing
from torch import Tensor
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union
from tensordict import (
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.utils import NestedKey
from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)

# TorchRL
from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    ValueEstimators,
    distance_loss,
)

from torchrl.objectives.ppo import PPOLoss
from torchrl.objectives.sac import SACLoss, _delezify, compute_log_prob

# Custom
from environment.utils import compute_violation
from rl_algorithms.utils import compute_loss_feasibility, recursive_check_for_nans, check_for_nans


def loss_feasibility(td, action, aggregate_feasibility="sum"):
    lhs_A = td.get("lhs_A")
    rhs = td.get("rhs")
    mean_violation = compute_violation(action, lhs_A, rhs)

    # Get aggregation dimensions
    if aggregate_feasibility == "sum":
        sum_dims = [-x for x in range(1, mean_violation.dim())]
        return mean_violation.sum(dim=sum_dims).mean(), mean_violation
    elif aggregate_feasibility == "mean":
        return mean_violation.mean(), mean_violation


class FeasibilitySACLoss(SACLoss):
    """TorchRL implementation of the SAC loss.

    Presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/abs/1801.01290
    and "Soft Actor-Critic Algorithms and Applications" https://arxiv.org/abs/1812.05905

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model.
            This module typically outputs a ``"state_action_value"`` entry.
            If a single instance of `qvalue_network` is provided, it will be duplicated ``num_qvalue_nets``
            times. If a list of modules is passed, their
            parameters will be stacked unless they share the same identity (in which case
            the original parameter will be expanded).

            .. warning:: When a list of parameters if passed, it will __not__ be compared against the policy parameters
              and all the parameters will be considered as untied.

        value_network (TensorDictModule, optional): V(s) parametric model.
            This module typically outputs a ``"state_value"`` entry.

            .. note::
              If not provided, the second version of SAC is assumed, where
              only the Q-Value network is needed.

    Keyword Args:
        num_qvalue_nets (integer, optional): number of Q-Value networks used.
            Defaults to ``2``.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (float, optional): max value of alpha.
            Default is None (no maximum value).
        action_spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        delay_actor (bool, optional): Whether to separate the target actor
            networks from the actor networks used for data collection.
            Default is ``False``.
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
            Default is ``True``.
        delay_value (bool, optional): Whether to separate the target value
            networks from the value networks used for data collection.
            Default is ``True``.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Tensordict key where to write the
            priority (for prioritized replay buffer usage). Defaults to ``"td_error"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"advantage"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"sample_log_prob"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey = "sample_log_prob"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    value_network: TensorDictModule | None
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    value_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    target_value_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule | List[TensorDictModule],
        value_network: Optional[TensorDictModule] = None,
        *,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = True,
        delay_value: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
        in_keys: Optional[List[NestedKey]] = None,
        out_keys: Optional[List[NestedKey]] = None,
    ) -> None:
        self._in_keys = in_keys
        self._out_keys = out_keys
        if reduction is None:
            reduction = "mean"
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            value_network=value_network,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            action_spec=action_spec,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            delay_value=delay_value,
            gamma=gamma,
            priority_key=priority_key,
            separate_losses=separate_losses,
            reduction=reduction,
        )

    state_dict = _delezify(LossModule.state_dict)
    load_state_dict = _delezify(LossModule.load_state_dict)

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        print("\nactor_loss:")
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            a_reparm = dist.rsample()
        log_prob = compute_log_prob(dist, a_reparm, self.tensor_keys.log_prob)

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qnetworkN0(
            td_q,
            self._cached_detached_qvalue_params,  # should we clone?
        )
        min_q_logprob = (
            td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        )

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        return self._alpha * log_prob - min_q_logprob, {"log_prob": log_prob, "action": a_reparm}

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._version == 1:
            loss_qvalue, value_metadata = self._qvalue_v1_loss(tensordict)
            loss_value, _ = self._value_loss(tensordict)
        else:
            loss_qvalue, value_metadata = self._qvalue_v2_loss(tensordict)
            loss_value = None
        loss_actor, metadata_actor = self._actor_loss(tensordict)
        loss_alpha = self._alpha_loss(log_prob=metadata_actor["log_prob"])
        tensordict.set(self.tensor_keys.priority, value_metadata["td_error"])
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
        }
        if self._version == 1:
            out["loss_value"] = loss_value
        td_out = TensorDict(out, [])
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        action = metadata_actor["action"]
        feasibility_loss, mean_violation = loss_feasibility(tensordict, action)
        td_out.set("loss_feasibility", feasibility_loss)
        td_out.set("violation", mean_violation)
        return td_out

    def _qvalue_v1_loss(
            self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_params = self._cached_target_params_actor_value
        with set_exploration_type(self.deterministic_sampling_mode):
            target_value = self.value_estimator.value_estimate(
                tensordict, target_params=target_params
            ).squeeze(-1)

        # Q-nets must be trained independently: as such, we split the data in 2
        # if required and train each q-net on one half of the data.
        shape = tensordict.shape
        if shape[0] % self.num_qvalue_nets != 0:
            raise RuntimeError(
                f"Batch size={tensordict.shape} is incompatible "
                f"with num_qvqlue_nets={self.num_qvalue_nets}."
            )
        tensordict_chunks = tensordict.reshape(
            self.num_qvalue_nets, -1, *tensordict.shape[1:]
        )
        target_chunks = target_value.reshape(
            self.num_qvalue_nets, -1, *target_value.shape[1:]
        )

        # if vmap=True, it is assumed that the input tensordict must be cast to the param shape
        tensordict_chunks = self._vmap_qnetwork00(
            tensordict_chunks, self.qvalue_network_params
        )
        pred_val = tensordict_chunks.get(self.tensor_keys.state_action_value)
        pred_val = pred_val.squeeze(-1)
        loss_value = distance_loss(
            pred_val, target_chunks, loss_function=self.loss_function
        ).view(*shape)
        metadata = {"td_error": (pred_val - target_chunks).pow(2).flatten(0, 1)}

        return loss_value, metadata

    def _compute_target_v2(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(
                ExplorationType.RANDOM
            ), self.actor_network_params.to_module(self.actor_network):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = compute_log_prob(
                    next_dist, next_action, self.tensor_keys.log_prob
                )

            # get q-values
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if (
                state_action_value.shape[-len(next_sample_log_prob.shape) :]
                != next_sample_log_prob.shape
            ):
                next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
            next_state_value = state_action_value - self._alpha * next_sample_log_prob
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _qvalue_v2_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # we pass the alpha value to the tensordict. Since it's a scalar, we must erase the batch-size first.
        print("\nqvalue_v2_loss (no_grad):")
        target_value = self._compute_target_v2(tensordict)
        print("target_value.requires_grad", target_value.requires_grad)

        print("\nprediction:")
        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys, strict=False),
            self.qvalue_network_params,
        )
        pred_val = tensordict_expand.get(self.tensor_keys.state_action_value).squeeze(
            -1
        )
        print("pred_val.requires_grad", pred_val.requires_grad,)
        td_error = abs(pred_val - target_value)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).sum(0)
        print("\nloss_qval.requires_grad", loss_qval.requires_grad)
        metadata = {"td_error": td_error.detach().max(0)[0]}
        return loss_qval, metadata

    def _value_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys, strict=False).detach()
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)
        pred_val = td_copy.get(self.tensor_keys.value).squeeze(-1)
        with self.target_actor_network_params.to_module(self.actor_network):
            action_dist = self.actor_network.get_dist(td_copy)  # resample an action
        action = action_dist.rsample()

        td_copy.set(self.tensor_keys.action, action, inplace=False)

        td_copy = self._vmap_qnetworkN0(
            td_copy,
            self.target_qvalue_network_params,
        )

        min_qval = (
            td_copy.get(self.tensor_keys.state_action_value).squeeze(-1).min(0)[0]
        )

        log_p = compute_log_prob(action_dist, action, self.tensor_keys.log_prob)

        if log_p.shape != min_qval.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {min_qval.shape} and {log_p.shape}"
            )
        target_val = min_qval - self._alpha * log_p

        loss_value = distance_loss(
            pred_val, target_val, loss_function=self.loss_function
        )
        return loss_value, {}

    def _alpha_loss(self, log_prob: Tensor) -> Tensor:
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return alpha_loss

    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha


class CustomSACLoss(SACLoss):
    def __init__(self, actor, critics, target_critics, value_network, alpha_init=0.1, gamma=0.99, target_entropy=0.1):
        super().__init__(actor_network=actor, qvalue_network=critics, value_network=value_network,)
        self.target_critics = target_critics
        self.log_alpha = torch.nn.Parameter(torch.tensor(alpha_init).log())
        self.gamma = gamma

    def _qvalue_loss(self, tensordict):
        # Compute target Q value
        with torch.no_grad():
            next_dist = self.actor_network.get_dist(tensordict["next"])
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).unsqueeze(-1)

            tensordict["next"].set("action", next_action)
            target_q_value0 = self.target_critics[0](tensordict["next"])
            target_q_value1 = self.target_critics[1](tensordict["next"])
            target_q_min = torch.min(target_q_value0["state_action_value"], target_q_value1["state_action_value"])
            target_q = target_q_min - self.alpha * next_log_prob

            reward = tensordict["next"].get("reward")
            done = tensordict.get("done").float()
            target = reward + self.gamma * (1 - done) * target_q

        # Critic loss
        q_value0 = self.qvalue_network[0](tensordict)
        q_value1 = self.qvalue_network[1](tensordict)
        qvalue0_loss = distance_loss(q_value0["state_action_value"], target, loss_function="smooth_l1")
        qvalue1_loss = distance_loss(q_value1["state_action_value"], target, loss_function="smooth_l1")
        qvalue_loss = (qvalue0_loss + qvalue1_loss).mean()
        return qvalue_loss, {}

    def _actor_loss(self, tensordict):
        # Sample action and compute log probabilities
        dist = self.actor_network.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)

        # Critic evaluation
        tensordict.set("action", action)
        with torch.no_grad():
            q_value0 = self.qvalue_network[0](tensordict)
            q_value1 = self.qvalue_network[1](tensordict)
            q_min = torch.min(q_value0["state_action_value"], q_value1["state_action_value"])
        # Actor loss
        actor_loss = (self.alpha * log_prob - q_min).mean()
        return actor_loss, {"log_prob": log_prob, "action": action}

    def _alpha_loss(self, tensordict):
        # Recompute log_prob
        dist = self.actor_network.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, tensordict):
        loss_actor, metadata_actor = self._actor_loss(tensordict)
        loss_qvalue, _ = self._qvalue_loss(tensordict)
        loss_qvalue = self._alpha_loss(tensordict)
        action = metadata_actor["action"]
        feasibility_loss, mean_violation = loss_feasibility(tensordict, action)
        entropy = -metadata_actor["log_prob"]

        # Output
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_qvalue,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
            "loss_feasibility": feasibility_loss,
            "violation": mean_violation,
        }
        td_out = TensorDict(out, [])
        return td_out

        td_out.set("loss_feasibility", feasibility_loss)
        td_out.set("violation", mean_violation)

        return {"loss_actor": actor_loss, "loss_qvalue": qvalue_loss, "alpha_loss": alpha_loss}


class FeasibilityClipPPOLoss(PPOLoss):
    """Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``. Set ``critic_coef`` to ``None`` to exclude the value
            loss from the forward outputs.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction with respect to the input tensordict value estimate and use it to
            calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
            helping stabilize training and preventing large updates. However, it will have no impact if the value
            estimate was done by the current version of the value estimator. If instead ``True`` is provided, the
            ``clip_epsilon`` parameter will be used as the clipping threshold. If not provided or ``False``, no
            clipping will be performed. Defaults to ``False``.

    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: bool | float | None = None,
        **kwargs,
    ):
        # Define clipping of the value loss
        if isinstance(clip_value, bool):
            clip_value = clip_epsilon if clip_value else None

        super(FeasibilityClipPPOLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            **kwargs,
        )
        for p in self.parameters():
            device = p.device
            break
        else:
            device = None
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon, device=device))

    @property
    def _clip_bounds(self):
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "clip_fraction"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain}, batch_size=[])
        td_out.set("clip_fraction", clip_fraction)

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        # Feasibility loss based on policy mean
        loc = dist.loc if hasattr(dist, 'loc') else dist.base_dist.loc
        feasibility_loss, mean_violation = loss_feasibility(tensordict, loc)
        td_out.set("loss_feasibility", feasibility_loss)
        td_out.set("mean_violation", mean_violation)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out