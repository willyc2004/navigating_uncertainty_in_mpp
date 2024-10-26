import yaml
import wandb
from dotmap import DotMap
from main import main, adapt_env_kwargs

if __name__ == "__main__":
    def train():
        try:
            # Initialize W&B
            wandb.init()
            sweep_config = wandb.config

            # Load static configuration from the YAML file
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = DotMap(config)
                config = adapt_env_kwargs(config)

            # Model hyperparameters
            config['model']['feedforward_hidden'] = sweep_config.feedforward_hidden
            config['model']['num_encoder_layers'] = sweep_config.num_encoder_layers
            config['model']['num_decoder_layers'] = sweep_config.num_decoder_layers
            config['model']['num_heads'] = sweep_config.num_heads
            # config['model']['embed_dim'] = sweep_config.embed_dim
            # config['model']['dropout_rate'] = sweep_config.dropout_rate
            # config['model']['lr_end_factor'] = sweep_config.lr_end_factor

            # PPO hyperparameters
            config['ppo']['ppo_epochs'] = sweep_config.ppo_epochs
            config['ppo']['mini_batch_size'] = sweep_config.mini_batch_size
            # config['ppo']['kl_threshold'] = sweep_config.kl_threshold
            # config['ppo']['clip_range'] = sweep_config.clip_range
            # config['ppo']['gamma'] = sweep_config.gamma
            # config['ppo']['gae_lambda'] = sweep_config.gae_lambda

            # # AM-PPO hyperparameters
            config['am_ppo']['lr'] = sweep_config.lr
            # config['am_ppo']['projection_kwargs']['iters'] = sweep_config.projection_iters
            # config['am_ppo']['train_data_size'] = sweep_config.train_data_size
            # config['am_ppo']['val_data_size'] = sweep_config.val_data_size

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
    sweep_id = wandb.sweep(sweep=sweep_config, project="mpp_ppo")
    # sweep_id = "e08klhqn"  # Use this line to manually set the sweep ID

    # Start the sweep agent, which runs the 'train' function with different hyperparameters
    wandb.agent(sweep_id, function=train)