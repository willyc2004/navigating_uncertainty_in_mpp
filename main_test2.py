import os
import yaml
import torch
import tqdm

from collections import defaultdict
from typing import Optional
from dotmap import DotMap
from tensordict.nn import TensorDictModule
from torch import nn

# TorchRL
from torchrl.envs.utils import check_env_specs

# Custom Environment
from environment.env_ import MasterPlanningEnv

def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs:DotMap, device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    return MasterPlanningEnv(**env_kwargs).to(device)  # Custom environment

class SimplePolicy(nn.Module):
    def __init__(self, hidden_dim, act_dim, device: torch.device = torch.device("cuda"), dtype=torch.float32):
        super().__init__()
        self.net = nn.Sequential(
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(act_dim),
    ).to(device).to(dtype)

    def forward(self, x):
        return self.net(x)

def main(config: Optional[DotMap] = None):
    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64  # or some higher value

    # Set random seed and device
    torch.set_num_threads(1)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        torch.backends.cudnn.deterministic = True

    ## Environment initialization
    env_kwargs = config.env
    env = make_env(env_kwargs)
    env.set_seed(seed)
    # env = ParallelEnv(4, make_env)     # todo: fix parallel env
    check_env_specs(env)  # this must pass for ParallelEnv to work

    ## Model initialization
    # todo: more complex model
    net = SimplePolicy(hidden_dim=128, act_dim=env.action_spec.shape[0], device=device, dtype=env.float_type)
    policy = TensorDictModule(
        net,
        in_keys=["observation", ],
        out_keys=["action"],
    )
    optim = torch.optim.Adam(policy.parameters(), lr=1e-5)

    ## Hyperparameters
    batch_size = 64
    total_train_steps = 1_000_000

    ## Training
    # torch.autograd.set_detect_anomaly(True)
    pbar = tqdm.tqdm(range(total_train_steps // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_train_steps)
    logs = defaultdict(list)
    for _ in pbar:
        init_td = env.reset(env.generator(batch_size=batch_size))
        rollout = env.rollout(72, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        traj_violation = rollout["next", "violation"].sum(dim=(-1,-2)).mean()
        loss = -traj_return + 0.05 * traj_violation
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, "
            f"last total_revenue: {rollout[..., -1]['next', 'state', 'total_revenue'].mean(): 4.4f}, "
            f"last total_cost: {rollout[..., -1]['next', 'state', 'total_cost'].mean(): 4.4f}, "
            f"last total_loaded: {rollout[..., -1]['next', 'state', 'total_loaded'].mean(): 4.4f}, "
            f"last total_violation: {rollout[..., -1]['next', 'state', 'total_violation'].sum(dim=-1).mean(): 4.4f}, "
            f"gradient norm: {gn: 4.4}, "
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

    def plot():
        from matplotlib import pyplot as plt

        with plt.ion():
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(logs["return"])
            plt.title("returns")
            plt.xlabel("iteration")
            plt.subplot(1, 2, 2)
            plt.plot(logs["last_reward"])
            plt.title("last reward")
            plt.xlabel("iteration")
            plt.show()

    plot()


if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    main(config)