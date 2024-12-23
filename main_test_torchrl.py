## Imports
import os
import copy

# Datatypes
import yaml
from dotmap import DotMap
from typing import Optional
from tensordict.nn import TensorDictModule

# Machine learning
import torch
import wandb

# TorchRL
from torchrl.envs.utils import check_env_specs
from torchrl.modules import TruncatedNormal, ValueOperator

# Custom:
# Training
from models.utils import make_env, adapt_env_kwargs
from models.train import train
# Models
from models.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from models.common import Autoencoder
from models.encoder import MLPEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache
from models.critic import CriticNetwork
from models.projection import ProjectionFactory
from models.projection_prob_actor import ProjectionProbabilisticActor
# RL4CO model
from rl4co.models.zoo.am.encoder import AttentionModelEncoder


# Main function
def main(config: Optional[DotMap] = None):
    # todo: clean-up and refactor all hyperparameters etc.
    # Environment kwargs
    env_kwargs = config.env

    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch._dynamo.config.cache_size_limit = 64  # or some higher value
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Environment initialization
    env = make_env(env_kwargs)
    seed = env_kwargs.seed
    env.set_seed(seed)
    # env = ParallelEnv(4, make_env)     # todo: fix parallel env
    check_env_specs(env)  # this must pass for ParallelEnv to work

    ## Model initialization
    # Embedding dimensions
    embed_dim = config.model.embed_dim
    obs_dim = env.observation_spec["observation"].shape[0]
    action_dim = env.action_spec.shape[0]
    sequence_dim = env.K * env.T if env.action_spec.shape[0] == env.B*env.D else env.P-1
    # Embedding initialization
    init_embed = MPPInitEmbedding(obs_dim, action_dim, embed_dim, sequence_dim, env)
    context_embed = MPPContextEmbedding(obs_dim, action_dim, embed_dim, sequence_dim, env, config.model.demand_aggregation)
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
        "temperature": config.model.temperature,
        "scale_max":config.model.scale_max,
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
    if config.algorithm.type == "sac":
        # Define two Q-networks for the critics
        critic1 = ValueOperator(
            CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=decoder_layers,
                          context_embedding=context_embed, normalization=config.model.normalization,
                          dropout_rate=dropout_rate, temperature=config.model.critic_temperature, customized=True,
                          use_q_value=True, action_dim=action_dim).to(device),
            in_keys=["observation", "action"],  # Input tensor key in TensorDict
            out_keys=["state_action_value"],
        )
        critic2 = copy.deepcopy(critic1)  # Second critic network
        critic = [critic1, critic2]
    else:
        # Get critic
        critic = TensorDictModule(
            CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=decoder_layers,
                          context_embedding=context_embed, normalization=config.model.normalization,
                          dropout_rate=dropout_rate, temperature=config.model.critic_temperature,
                          customized=True).to(device),
            in_keys=["observation",],  # Input tensor key in TensorDict
            out_keys=["state_value"],  # ["state_action_value"]  # Output tensor key in TensorDict
        )

    # Get ProbabilisticActor (for stochastic policies)
    actor = TensorDictModule(
        Autoencoder(encoder, decoder).to(device),
        in_keys=["observation",],  # Input tensor key in TensorDict
        out_keys=["loc","scale"]  # Output tensor key in TensorDict
    )
    if config.training.projection_type in ["linear_violation", "linear_program", "convex_program"]:
        config.training.projection_kwargs["n_action"] = action_dim
        config.training.projection_kwargs["n_constraints"] = env.n_constraints
        projection_layer = ProjectionFactory.create_class(config.training.projection_type, config.training.projection_kwargs)
    else:
        projection_layer = None
    policy = ProjectionProbabilisticActor(
        module=actor,
        in_keys=["loc", "scale"],
        distribution_class=TruncatedNormal,
        distribution_kwargs={"low": env.action_spec.low[0], "high": env.action_spec.high[0]},
        # distribution_kwargs={"upscale":1.0},
        return_log_prob=True,
        projection_layer=projection_layer,
        # action_rescale_min=env.action_spec.low[0],
        # action_rescale_max=env.action_spec.high[0],
    )

    ## Main loop
    # Train the model
    if config.model.phase == "train":
        train(policy, critic, **config)
    # Test the model
    elif config.model.phase == "test":
            # todo: extract trained hyper-parameters
            # todo: check if two models can be re-loaded
            # todo: use validation function, add visualizations per episode
        raise NotImplementedError("Testing not implemented yet.")

if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config_torchrl.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    # Call your main() function
    try:
        wandb.init(config=config,)
        model = main(config)
    except Exception as e:
        # Log the error to WandB
        wandb.log({"error": str(e)})

        # Optionally, use WandB alert for critical errors
        wandb.alert(
            title="Training Error",
            text=f"An error occurred during training: {e}",
            level="error"  # 'info' or 'warning' levels can be used as needed
        )

        # Print the error for local console logging as well
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()