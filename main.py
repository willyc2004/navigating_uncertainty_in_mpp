"""
Deep Reinforcement Learning for Master Planning Problem (MPP) in Container Vessel Stowage Planning
Author: Jaike van Twiller
Date: 2024/07/11

This script demonstrates how to train a model for the Master Planning Problem (MPP) in Container Stowage Planning. The
MPP is a combinatorial optimization problem, which aims to find a global stowage plan on a container vessel that
maximizes the total profit during a fixed-schedule multi-port voyage. The voyage demand is generated and transformed
into a state-dependent dataset.

In our approach, we use the Attention Model Proximal Policy Optimization algorithm (AM-PPO) is used to learn a policy.
This architecture consists of an attention-based encoder-decoder model, which is trained using the single-step PPO
algorithm.
"""

# todo: features to add:
#  - allow for multiple GPUs
#  - create a parallel environment with RL4COEnvBase inheritance
#  - jit compile the model; resolve graph breaks in the model
#  - resolve the issue with multiple workers

## Import libraries and modules
import yaml
from dotmap import DotMap
import time
import wandb
from tensordict import TensorDict

# PyTorch, Lightning
import torch
import torch.profiler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ObservationNorm

# RL4CO
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.trainer import RL4COTrainer
from rl4co.models.zoo import AMPPO
# Customized RL4CO modules
from models.projection_ppo import Projection_SinglePPO
from models.projection_stepwise_ppo import Projection_StepwisePPO
from models.projection_n_step_ppo import Projection_Nstep_PPO
from models.constructive import ConstructivePolicy
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
AutoregressivePolicy.__bases__ = (ConstructivePolicy,) # Adapt base class

# Custom modules
from environment.env import MasterPlanningEnv
from environment.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from environment.data import StateDependentDataset, custom_collate_fn
from environment.results import rollout_results
from environment.trial import trial
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache
from models.am_policy import AttentionModelPolicy4PPO
from models.critic import CriticNetwork

# Helper functions
def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs, device):
    """Setup custom environment"""
    return MasterPlanningEnv(**env_kwargs).to(device).half()

def check_env_specs(env):
    """Verifies that the environment's specifications (action and observation spaces) are valid."""
    try:
        action_spec = env.action_spec
        observation_spec = env.observation_spec
        print("Action space shape:", action_spec.shape)
        print("Observation space shape:", observation_spec.shape)
        return True
    except AttributeError as e:
        print(f"Error: {e}")
        print("Please make sure your environment defines valid action_spec and observation_spec properties.")
        return False

# Main function
def main(config=None):
    """Main function to train model"""
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
    env = make_env(env_kwargs, device)

    if check_env_specs(env):
        print("Environment specifications seem valid!")
    else:
        print("Environment specification check failed.")

    # Run a trial of the environment
    batch_size = config.model.batch_size
    td = env.reset(batch_size=batch_size, )
    # trial(env, td, device, num_rollouts=30, EDA=True) # uncomment to trial

    ## Model initialization
    # Embedding dimensions
    init_dim = config.model.init_dim
    embed_dim = config.model.embed_dim
    sequence_dim = env.K * env.T
    obs_dim = env.observation_spec["observation"].shape[0]
    action_dim = env.action_spec.shape[0]

    # Initialize positional encoding to generalize
    max_P = 6
    time_step_lookup = torch.zeros((max_P, max_P, env.K), dtype=torch.long, device=device)
    timestep = 0
    for pol in range(max_P-1):
        for pod in range(pol+1, max_P):
            for class_k in range(env.K):
                time_step_lookup[pol, pod, class_k] = timestep
                timestep += 1


    # Embedding initialization
    init_embed = MPPInitEmbedding(embed_dim, action_dim, env)
    context_embed = MPPContextEmbedding(action_dim, embed_dim, env, device,)
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
    encoder = AttentionModelEncoder(**encoder_args,)
    if config.model.decoder_type == "attention":
        decoder = AttentionDecoderWithCache(**decoder_args)
    elif config.model.decoder_type == "mlp":
        decoder = MLPDecoderWithCache(**decoder_args)
    else:
        raise ValueError(f"Decoder type {config.model.decoder_type} not recognized.")

    # AM Model initialization
    model_params = {
        "decoder": decoder,
        "encoder": encoder,
        "init_embedding": init_embed,
        "context_embedding": context_embed,
        "dynamic_embedding": dynamic_embed,
        "projection_type": config.am_ppo.projection_type,
        "projection_kwargs": config.am_ppo.projection_kwargs,
        **config.model
    }
    # AM PPO initialization
    projection_kwargs = config.am_ppo.pop("projection_kwargs",)
    projection_kwargs.update({"n_actions": env.action_spec.shape[0], "n_constraints": env.n_constraints})
    policy = AttentionModelPolicy4PPO(**model_params) # AttentionModelPolicy(**model_params),
    policy.apply(init_he_weights)
    critic = CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim,num_layers = decoder_layers, context_embedding=context_embed)
    critic.apply(init_he_weights)

    am_ppo_params = {
        "env": env,
        "policy":policy,
        "critic": critic,
        "lr_scheduler": LinearLR,
        "lr_scheduler_kwargs": {"end_factor": config.model.lr_end_factor,},
        "projection_kwargs": projection_kwargs,
        **config.ppo,
        **config.am_ppo,
    }
    AMPPO.__bases__ = (Projection_Nstep_PPO,) # Adapt base class
    model = AMPPO(**am_ppo_params).to(device)
    # print(model)
    # breakpoint()

    ## Training configuration
    date_time_str = time.strftime("%Y/%m/%d/%H-%M-%S")
    print(f"Training started at {date_time_str}")

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{date_time_str}/",  # save to checkpoints/
        filename=f"latest_policy",
        every_n_train_steps=500,
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/reward",  # monitor validation reward; episodic profit in this case
        mode="max", # maximize validation reward
    )
    rich_model_summary = RichModelSummary(max_depth=3)
    early_stopping = EarlyStopping(monitor="val/reward", patience=3, mode="max", verbose=False, check_finite=True)
    callbacks = [checkpoint_callback, rich_model_summary, early_stopping]
    # callbacks = None # uncomment this line if you don't want callbacks

    # Initialize logger
    if config.model.logger == "wandb":
        logger = WandbLogger(project="mpp", name="mpp_under_uncertainty", log_model='all', config=config)
    else:
        logger = None

    # Initialize datasets and dataloaders
    train_dataset = StateDependentDataset(env=env, td=None, batch_size=config.model.batch_size,
                                          total_samples=config.am_ppo.train_data_size)
    val_dataset = StateDependentDataset(env=env, td=None, batch_size=config.model.batch_size,
                                        total_samples=config.am_ppo.val_data_size)
    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0, collate_fn=custom_collate_fn,
                                  pin_memory=False,) # shuffle=False (for iterable dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0, collate_fn=custom_collate_fn,
                                pin_memory=False,) # shuffle=False (for iterable dataset)
    # Main trainer configuration
    trainer = RL4COTrainer(
        max_epochs=1, # full training epochs
        val_check_interval=0.1, # validate every epoch
        accelerator="gpu",
        devices=1, #torch.cuda.device_count(),
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed', # '16-mixed' or '32'
        matmul_precision="medium",
        # profiler=AdvancedProfiler(),
        # strategy="ddp_find_unused_parameters_true", # only for multiple gpus
    )

    ## Main training loop
    # The model gathers sequences of actions, to get reward of a rollout
    if config.model.phase == "train":
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # Load the checkpoint model for testing
    elif config.model.phase == "test":
        date_stamp = f"2024/10/30/00-53-46"
        checkpoint_path = f"checkpoints/{date_stamp}"
        ckpt_name = "/last.ckpt" #"/epoch_epoch=00-val_loss=0.00.ckpt"
        checkpoint = torch.load(checkpoint_path + ckpt_name,)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        # Initialize policy
        policy = model.policy
        env_kwargs["float_type"] = torch.float32
        test_env = make_env(env_kwargs, device)  # Re-initialize the environment

        # Run multiple iterations to measure inference time
        num_runs = 20
        outs = []
        times = []

        for i in range(num_runs):
            # Set a new seed for each run
            seed = i
            torch.manual_seed(seed)
            td_init = test_env.reset(batch_size=batch_size, td=td)

            # Sync GPU before starting timer if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run inference
            out = policy(td_init.clone(), test_env, phase="test", decode_type="continuous_sampling",
                         return_actions=True, return_td=True, return_feasibility=True,
                         projection_type=am_ppo_params["projection_type"], projection_kwargs=projection_kwargs)

            # Sync GPU again after inference if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Calculate and record inference time for each run
            outs.append(out)
            times.append(end_time - start_time)
        times = torch.tensor(times)
        rollout_results(test_env, outs, td, batch_size, checkpoint_path,
                        am_ppo_params["projection_type"], config["env"]["utilization_rate_initial_demand"], times)
    return model

def init_he_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for ReLU
    if isinstance(m, torch.nn.MultiheadAttention):
        torch.nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.01)  # Small normal init for attention weights

if __name__ == "__main__":
    # Load static configuration from the YAML file
    with open('config.yaml', 'r') as file:
    # with open('test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    # Call your main() function
    try:
        wandb.init()
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

    # uncomment to profile the code
    # import torch.autograd.profiler as profiler
    #
    # # Wrap the part of your code you want to profile
    # with profiler.profile(use_cuda=True) as prof:
    #     model = main()
    #
    # # Print the profiler results
    # print(prof.key_averages().table(sort_by="cuda_time_total"))