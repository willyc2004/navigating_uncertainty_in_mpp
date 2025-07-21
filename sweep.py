import yaml
import wandb
from dotmap import DotMap
from main import main, adapt_env_kwargs
import argparse

if __name__ == "__main__":
    # add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", nargs="?", default=None, const=None,
                        help="Provide a sweep name to resume an existing sweep, or leave empty to create a new sweep.")
    # path
    parser.add_argument("--path", type=str, default="results/trained_models/navigating_uncertainty",
                        help="Path to the directory containing the config.yaml and sweep_config.yaml files.")
    parser.add_argument("--folder", type=str, default="sac-pd",
                        help="Folder to save the sweep configuration and results.")
    args = parser.parse_args()

    def train():
        try:
            # Load static configuration from the YAML file
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = DotMap(config)
                config = adapt_env_kwargs(config)
                n_constraints = config.training.projection_kwargs.n_constraints
                config.testing.folder = args.folder
                config.testing.path = args.path

            # Initialize W&B
            wandb.init(config=config)
            sweep_config = wandb.config

            # # Model hyperparameters
            # # config['model']['num_heads'] = sweep_config.num_heads
            # # config['model']['dropout_rate'] = sweep_config.dropout_rate
            # # config['model']['normalization'] = sweep_config.normalization
            # config['model']['hidden_dim'] = sweep_config.hidden_dim
            # config['model']['embed_dim'] = sweep_config.embed_dim
            # config['model']['num_encoder_layers'] = sweep_config.num_encoder_layers
            # config['model']['num_decoder_layers'] = sweep_config.num_decoder_layers
            # config['model']['batch_size'] = sweep_config.batch_size
            # config['model']['scale_max'] = sweep_config.scale_max
            # config['model']['temperature'] = sweep_config.temperature
            #
            # # # PPO hyperparameters
            # # config['algorithm']['ppo_epochs'] = sweep_config.ppo_epochs
            # # config['algorithm']['mini_batch_size'] = sweep_config.mini_batch_size
            # # config['algorithm']['entropy_lambda'] = sweep_config.entropy_lambda
            #
            # # # AM-PPO hyperparameters
            config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
            # config['training']['lr'] = sweep_config.lr
            # config['training']['pd_lr'] = sweep_config.pd_lr
            # config['training']['projection_kwargs']['alpha'] = sweep_config.alpha
            # config['training']['projection_kwargs']['delta'] = sweep_config.delta
            # config['training']['projection_kwargs']['max_iter'] = sweep_config.max_iter
            # config['training']['projection_kwargs']['scale'] = sweep_config.scale

            # Algorithm hyperparameters
            for i in range(n_constraints):
                config['algorithm'][f'lagrangian_multiplier_{i}'] = sweep_config[f'lagrangian_multiplier_{i}']
                # Error handling for missing lagrangian multipliers
                if f'lagrangian_multiplier_{i}' not in sweep_config:
                    raise ValueError(f"Missing lagrangian_multiplier_{i} in sweep configuration")

            # Call your main() function
            model = main(config)

            # # Optionally log some results, metrics, or intermediate values here
            # wandb.log({"training_loss": 0.1})  # Example logging
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

    # Load the sweep configuration from YAML
    with open('sweep_config.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep with W&B
    if args.sweep:
        sweep_id = args.sweep
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project="mpp_ppo")

    # Start the sweep agent, which runs the 'train' function with different hyperparameters
    wandb.agent(sweep_id, function=train, project="mpp_ppo", entity="stowage_planning_research")