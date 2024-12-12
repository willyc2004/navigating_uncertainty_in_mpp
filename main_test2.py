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
from torchrl.modules import ProbabilisticActor

# RL4CO
from rl4co.models.zoo.am.encoder import AttentionModelEncoder

# Custom
from environment.env_ import MasterPlanningEnv
from environment.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from models.encoder import MLPEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache


# Helper functions
def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs:DotMap, device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    return MasterPlanningEnv(**env_kwargs).to(device)  # Custom environment

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for ReLU
    if isinstance(m, torch.nn.MultiheadAttention):
        torch.nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.01)  # Small normal init for attention weights
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)

# Classes
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

class AutoEncoder(nn.Module):
    """Create autoencoder model from initial embedding, encoder, context embedding, decoder models passed as arguments."""
    def __init__(self, encoder, decoder, init_embed, context_embed, dynamic_embed):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_embed = init_embed
        self.context_embed = context_embed
        self.dynamic_embed = dynamic_embed

    def forward(self, obs):
        # Initial embedding
        init_embed = self.init_embed(obs)
        # Encode initial embedding to hidden state
        hidden, _ = self.encoder(init_embed)
        # Context embedding
        context_embed = self.context_embed(obs, hidden)
        # Decode context embedding to output
        dec_out = self.decoder(context_embed)
        return dec_out



## Training
def train(batch_size, train_data_size, policy, env, model, optim):
    # torch.autograd.set_detect_anomaly(True)
    pbar = tqdm.tqdm(range(train_data_size // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, train_data_size)
    logs = defaultdict(list)
    for _ in pbar:
        init_td = env.reset(env.generator(batch_size=batch_size))
        rollout = env.rollout(72, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        traj_violation = rollout["next", "violation"].sum(dim=(-1,-2)).mean()
        loss = -traj_return + 0.05 * traj_violation
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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

def main(config: Optional[DotMap] = None):
    # todo: clean-up and refactor all hyperparameters etc.
    # Environment kwargs
    env_kwargs = config.env

    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64  # or some higher value

    # Set random seed and device
    torch.set_num_threads(1)
    seed = env_kwargs.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        torch.backends.cudnn.deterministic = True

    ## Environment initialization
    env = make_env(env_kwargs)
    env.set_seed(seed)
    # env = ParallelEnv(4, make_env)     # todo: fix parallel env
    check_env_specs(env)  # this must pass for ParallelEnv to work

    ## Model initialization
    # Embedding dimensions
    init_dim = config.model.init_dim
    embed_dim = config.model.embed_dim
    sequence_dim = env.K * env.T
    obs_dim = env.observation_spec["observation"].shape[0]
    action_dim = env.action_spec.shape[0]
    # Embedding initialization
    init_embed = MPPInitEmbedding(embed_dim, action_dim, env)
    context_embed = MPPContextEmbedding(action_dim, embed_dim, env, config.model.demand_aggregation)
    dynamic_embed = StaticEmbedding(obs_dim, embed_dim)

    # Model initialization
    hidden_dim = config.model.feedforward_hidden
    encoder_layers = config.model.num_encoder_layers
    decoder_layers = config.model.num_decoder_layers
    num_heads = config.model.num_heads
    dropout_rate = config.model.dropout_rate
    decoder_args = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_hidden_layers": decoder_layers,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "action_size": action_dim,
        "state_size": obs_dim,
        "total_steps": sequence_dim,
        "init_embedding": init_embed,
        "context_embedding": context_embed,
        "dynamic_embedding": dynamic_embed,
        "normalization": config.model.normalization,
    }
    encoder_args = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "init_embedding": init_embed,
        "env_name": env.name,
        "num_layers": encoder_layers,
        "feedforward_hidden": hidden_dim,
        "normalization": config.model.normalization,
    }

    # Setup model
    if config.model.encoder_type == "attention":
        encoder = AttentionModelEncoder(**encoder_args,)
    else:
        encoder = MLPEncoder(**encoder_args)
    if config.model.decoder_type == "attention":
        decoder = AttentionDecoderWithCache(**decoder_args)
    elif config.model.decoder_type == "mlp":
        decoder = MLPDecoderWithCache(**decoder_args)
    else:
        raise ValueError(f"Decoder type {config.model.decoder_type} not recognized.")


    # todo: more complex model
    model = AutoEncoder(encoder, decoder, init_embed, context_embed, dynamic_embed).to(device)
    # model = SimplePolicy(hidden_dim=hidden_dim, act_dim=action_dim, device=device, dtype=env.float_type).apply(init_weights)

    # Wrap in a TensorDictModule
    td_module = TensorDictModule(
        model,
        in_keys=["observation",],  # Input tensor key in TensorDict
        out_keys=["loc"]  # Output tensor key in TensorDict
    )
    # ProbabilisticActor (for stochastic policies)
    policy = ProbabilisticActor(
        module=td_module,
        in_keys=["loc"],
        distribution_class=torch.distributions.Normal,  # Gaussian policy
        distribution_kwargs={"scale": 1.0}
    )




    # # AM Model initialization
    # model_params = {
    #     "decoder": decoder,
    #     "encoder": encoder,
    #     # "init_embedding": init_embed,
    #     # "context_embedding": context_embed,
    #     # "dynamic_embedding": dynamic_embed,
    #     "projection_type": config.am_ppo.projection_type,
    #     "projection_kwargs": config.am_ppo.projection_kwargs,
    #     "select_obs_td":["obs", "done", "timestep", "action_mask", "lhs_A", "rhs", "clip_min", "clip_max",
    #                           "reward",
    #                           ("state", "utilization"), ("state", "target_long_crane"), ("state", "total_loaded"),
    #                           ("state", "total_revenue"), ("state", "total_cost"), ("state", "total_rc"),
    #                           "realized_demand",
    #                           ],
    #     **config.model
    # }


    ## Hyperparameters
    batch_size = config.model.batch_size
    train_data_size = config.am_ppo.train_data_size
    lr = config.am_ppo.lr

    # Optimizer
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    # Train the model
    train(batch_size, train_data_size, policy, env, model, optim)

if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    main(config)