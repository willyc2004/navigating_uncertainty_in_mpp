from __future__ import annotations

import torch
from tensordict import (
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
import torch.nn.functional as F

from torchrl.objectives.utils import (
    _reduce,
)
from torchrl.objectives.ppo import PPOLoss

# Custom
from environment.utils import compute_violation
from rl_algorithms.utils import compute_loss_feasibility, recursive_check_for_nans, check_for_nans

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
        feasibility_coef: float = 1.0,
        aggregate_feasibility: str = "sum",
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
        self.register_buffer("feasibility_coef", torch.tensor(feasibility_coef, device=device))
        self.aggregate_feasibility = aggregate_feasibility

    @property
    def _clip_bounds(self):
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    def loss_feasibility(self, td, dist):
        loc = dist.loc if hasattr(dist, 'loc') else dist.base_dist.loc
        lhs_A = td.get("lhs_A")
        rhs = td.get("rhs")
        mean_violation = compute_violation(loc, lhs_A, rhs)

        # Get aggregation dimensions
        if self.aggregate_feasibility == "sum":
            sum_dims = [-x for x in range(1, mean_violation.dim())]
            return self.feasibility_coef * mean_violation.sum(dim=sum_dims).mean(), mean_violation
        elif self.aggregate_feasibility == "mean":
            return self.feasibility_coef * mean_violation.mean(), mean_violation

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
        if self.feasibility_coef is not None:
            feasibility_loss, mean_violation = self.loss_feasibility(tensordict, dist)
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

def optimize_sac_loss(subdata, policy, critics, actor_optim, critic_optim, **kwargs):
    # todo: make in similar format as loss_module

    ## Hyperparameters
    gamma = kwargs["algorithm"]["gamma"]
    tau = kwargs["algorithm"]["tau"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]

    ## Unpack critics
    critic1 = critics["critic1"]
    critic2 = critics["critic2"]
    target_critic1 = critics["target_critic1"]
    target_critic2 = critics["target_critic2"]

    ## Loss optimization
    loss_out = {}
    # Critic loss calculation
    with torch.no_grad():
        # Next action
        next_policy_out = policy(subdata)
        next_action = next_policy_out["action"]
        next_log_prob = next_policy_out["sample_log_prob"].unsqueeze(-1)
        next_log_prob = torch.clamp(next_log_prob, -20, 2)  # Clip log_prob to avoid NaNs

        # Target value
        target_q1 = target_critic1(next_policy_out["observation"], next_action)
        target_q2 = target_critic2(next_policy_out["observation"], next_action)
        target_q_min = torch.min(target_q1, target_q2) - entropy_lambda * next_log_prob
        target_value = subdata["next", "reward"] + (1 - subdata["done"].float()) * gamma * target_q_min
        check_for_nans(target_value, "target_value")

    # Current value
    current_q1 = critic1(subdata["observation"], subdata["action"])
    current_q2 = critic2(subdata["observation"], subdata["action"])

    # Update critic
    loss_out["loss_critic"] = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
    check_for_nans(loss_out["loss_critic"], "loss_critic")
    critic_optim.zero_grad()
    loss_out["loss_critic"].backward()
    loss_out["gn_critic1"] = torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_grad_norm)
    loss_out["gn_critic2"] = torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_grad_norm)
    check_for_nans(loss_out["loss_critic"], "loss_critic")
    check_for_nans(loss_out["gn_critic1"], "gn_critic1")
    check_for_nans(loss_out["gn_critic2"], "gn_critic2")
    critic_optim.step()

    # Soft update target critics
    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Compute Actor Loss
    policy_out = policy(subdata)
    policy_out["sample_log_prob"] = torch.clamp(policy_out["sample_log_prob"], -20, 2)  # Clip log_prob to avoid NaNs
    recursive_check_for_nans(policy_out)

    # Feasibility loss
    loss_out["loss_feasibility"], loss_out["mean_violation"] = compute_loss_feasibility(policy_out, policy_out["action"], feasibility_lambda, "sum")
    check_for_nans(loss_out["loss_feasibility"], "loss_feasibility")
    q1 = critic1(policy_out["observation"], policy_out["action"])
    q2 = critic2(policy_out["observation"], policy_out["action"])
    q_min = torch.min(q1, q2)
    check_for_nans(q_min, "q_min")

    # Update actor
    loss_out["loss_actor"] = (entropy_lambda * policy_out["sample_log_prob"].unsqueeze(-1) - q_min).mean() + loss_out["loss_feasibility"]
    actor_optim.zero_grad()
    loss_out["loss_actor"].backward()
    check_for_nans(loss_out["loss_actor"], "loss_actor")
    loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    check_for_nans(loss_out["gn_actor"], "gn_actor")
    actor_optim.step()
    return loss_out, policy_out