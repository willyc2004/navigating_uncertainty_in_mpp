from typing import Any, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    ListStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def make_replay_buffer(buffer_size, batch_size, device="cuda"):
    if device == "cpu":
        storage = LazyMemmapStorage(buffer_size, device="cpu")
        prefetch = 3
    else:
        storage = ListStorage(buffer_size)
        prefetch = 3
    return TensorDictReplayBuffer(
        storage=storage,
        batch_size=batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
        pin_memory=False,
        prefetch=prefetch,
    )


class Projection_StepwisePPO(RL4COLitModule):

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        clip_range: float = 0.2,  # epsilon of PPO
        update_timestep: int = 1,
        buffer_size: int = 100_000,
        ppo_epochs: int = 2,  # inner epoch, K
        batch_size: int = 256, # batch size
        mini_batch_size: Union[int, float] = 0.25,  # mini batch size,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        feasibility_lambda: float = 1.0,  # lambda of feasibility loss
        demand_lambda: float = 1.0,  # lambda of demand violations
        stability_lambda: float = 1.0,  # lambda of stability violations
        projection_lambda: float = 1.0,  # lambda of projection loss
        normalize_adv: bool = True,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        buffer_storage_device: str = "gpu",
        metrics: dict = {
            "train": ["loss", "surrogate_loss", "value_loss", "entropy",
                      "feasibility_loss", "projection_loss",
                      "reward", "costs", "adv", "value_pred", "ratio",
                      # "action", "logprobs", "logprobs_old",
                      # "violations", "total_loaded"
                      ],
        },
        **kwargs,
    ):
        self.projection_layer = kwargs.pop("projection_type", None) # pop before passing to super
        self.projection_kwargs = kwargs.pop("projection_kwargs", None) # pop before passing to super
        super().__init__(env, policy, metrics=metrics, **kwargs)
        self.automatic_optimization = False  # PPO uses custom optimization routine
        self.policy_old = copy.deepcopy(self.policy)
        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            critic = create_critic_from_actor(policy, **critic_kwargs)
        self.critic = critic

        if isinstance(mini_batch_size, float) and (
            mini_batch_size <= 0 or mini_batch_size > 1
        ):
            default_mini_batch_fraction = 0.25
            log.warning(
                f"mini_batch_size must be an integer or a float in the range (0, 1], got {mini_batch_size}. Setting mini_batch_size to {default_mini_batch_fraction}."
            )
            mini_batch_size = default_mini_batch_fraction

        if isinstance(mini_batch_size, int) and (mini_batch_size <= 0):
            default_mini_batch_size = 128
            log.warning(
                f"mini_batch_size must be an integer or a float in the range (0, 1], got {mini_batch_size}. Setting mini_batch_size to {default_mini_batch_size}."
            )
            mini_batch_size = default_mini_batch_size

        self.rb = make_replay_buffer(buffer_size, mini_batch_size, buffer_storage_device)
        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "feasibility_lambda": feasibility_lambda,
            "projection_lambda": projection_lambda,
            "normalize_adv": normalize_adv,
            "max_grad_norm": max_grad_norm,
            "update_timestep": update_timestep,
        }
        self.lambda_violations = torch.tensor([demand_lambda] + [stability_lambda] * env.n_stability,
                                              device='cuda', dtype=torch.float32)

    def configure_optimizers(self):
        parameters = list(self.policy.parameters()) + list(self.critic.parameters())
        return super().configure_optimizers(parameters)

    def on_train_epoch_end(self):
        """
        ToDo: Add support for other schedulers.
        """

        sch = self.lr_schedulers()

        # If the selected scheduler is a MultiStepLR scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.MultiStepLR):
            sch.step()

    def update(self, device):
        outs = []
        for _ in range(self.ppo_cfg["ppo_epochs"]):  # PPO inner epoch, K
            for sub_td in self.rb:
                # Get the previous log probabilities and rewards
                previous_logprobs = sub_td["logprobs"]
                previous_reward = sub_td["reward"].view(-1, 1)

                # Evaluate the policy and critic
                out_td = self.policy.evaluate(sub_td, env=self.env, action=sub_td["action"],)
                value_pred = self.critic(sub_td)  # [batch, 1]

                # Unpack output
                logprobs, entropy, action = out_td["logprobs"], out_td["entropy"], out_td["action"]
                mean_logits, std_logits, proj_mean_logits = out_td["mean_logits"], out_td["std_logits"], out_td["proj_mean_logits"]
                lhs_A, rhs, agg_costs = out_td["lhs_A"], out_td["rhs"], out_td["agg_costs"]

                # Compute the ratio of probabilities of new and old actions
                ratios = torch.exp(logprobs - previous_logprobs)
                advantages = previous_reward - value_pred.detach()  # [batch, 1]

                # Normalize advantage
                if self.ppo_cfg["normalize_adv"]:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Compute the surrogate loss
                surrogate_loss = -torch.min(
                    ratios * advantages,
                    torch.clamp(
                        ratios,
                        1 - self.ppo_cfg["clip_range"],
                        1 + self.ppo_cfg["clip_range"],
                    )
                    * advantages,
                ).mean()

                # compute value function loss
                value_loss = F.huber_loss(value_pred, previous_reward)

                # Custom loss terms are computed with the mean logit, as it is part of the computation graph.
                # Note that actions are sampled from policy distribution derived from logits, i.e. detached from
                # the computation graph. Instead, we use the mean to compute distances;
                # - Feasibility loss computes distance between the feasible set and the project policy distribution
                # - Projection loss computes the distance between the raw and projected policy distributions
                lhs = (lhs_A * proj_mean_logits.unsqueeze(-2)).sum(dim=-1)
                violation = torch.clamp(lhs - rhs, min=0)
                feasibility_loss = F.mse_loss(self.lambda_violations * violation, torch.zeros_like(violation),
                                              reduction="mean")
                projection_loss = F.mse_loss(mean_logits, proj_mean_logits, reduction="mean")

                # compute total loss:
                #   max surrogate loss, entropy,
                #   min value loss, feasibility, and projection loss
                loss = (
                        surrogate_loss
                        + self.ppo_cfg["vf_lambda"] * value_loss
                        - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                        + self.ppo_cfg["feasibility_lambda"] * feasibility_loss
                        + self.ppo_cfg["projection_lambda"] * projection_loss
                )

                # perform manual optimization following the Lightning routine
                # https://lightning.ai/docs/pytorch/stable/common/optimization.html
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.ppo_cfg["max_grad_norm"] is not None:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                        gradient_clip_algorithm="norm",
                    )
                opt.step()

                metrics = {
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": self.ppo_cfg["vf_lambda"] * value_loss,
                    "entropy": -self.ppo_cfg["entropy_lambda"] * entropy.mean(),
                    "feasibility_loss": self.ppo_cfg["feasibility_lambda"] * feasibility_loss,
                    "projection_loss": self.ppo_cfg["projection_lambda"] * projection_loss,
                    # extra logging
                    "reward": previous_reward.mean(),
                    "ratio": ratios.mean(),
                    "adv": ratios.mean(),
                    "value_pred": value_pred.mean(),
                    "action": action.mean(),
                    "logprobs": logprobs.sum(),
                    "logprobs_old": sub_td["logprobs"].sum(),
                    "violations": violation.mean(),
                    "costs": agg_costs.mean(),
                    # "total_loaded": total_loaded.mean(),
                }
                outs.append(metrics)

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        outs = {k: torch.stack([dic[k] for dic in outs], dim=0) for k in outs[0]}
        return outs

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        next_td = self.env.reset(batch)
        device = next_td.device

        if phase == "train":
            while not next_td["done"].all():
                with torch.no_grad():
                    td = self.policy_old.act(next_td, self.env, phase=phase,)

                # get reward from the environment # todo: check if step-wise or aggregate reward
                next_td = self.env.step(td.clone())["next"]
                reward = self.env.get_reward(next_td, td["state"]["utilization"],)
                td.set("reward", reward)
                self.rb.extend(td)

                if batch_idx % self.ppo_cfg["update_timestep"] == 0:
                    out = self.update(device)
                    self.rb.empty()
        else:
            out = self.policy.generate(next_td, env=self.env, phase = phase)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}