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

    def train():
        try:
            # Load static configuration from the YAML file
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = DotMap(config)
                config = adapt_env_kwargs(config)

            # Initialize W&B
            wandb.init(config=config)
            sweep_config = wandb.config
            config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
            config['algorithm'][f'lagrangian_multiplier_0'] = sweep_config.lagrangian_multiplier_0
            config['algorithm'][f'lagrangian_multiplier_1'] = sweep_config.lagrangian_multiplier_1
            config['algorithm'][f'lagrangian_multiplier_2'] = sweep_config.lagrangian_multiplier_2
            config['algorithm'][f'lagrangian_multiplier_3'] = sweep_config.lagrangian_multiplier_3
            config['algorithm'][f'lagrangian_multiplier_4'] = sweep_config.lagrangian_multiplier_4
            config['algorithm'][f'lagrangian_multiplier_5'] = sweep_config.lagrangian_multiplier_5
            config['algorithm'][f'lagrangian_multiplier_6'] = sweep_config.lagrangian_multiplier_6
            config['algorithm'][f'lagrangian_multiplier_7'] = sweep_config.lagrangian_multiplier_7
            config['algorithm'][f'lagrangian_multiplier_8'] = sweep_config.lagrangian_multiplier_8
            config['algorithm'][f'lagrangian_multiplier_9'] = sweep_config.lagrangian_multiplier_9
            config['algorithm'][f'lagrangian_multiplier_10'] = sweep_config.lagrangian_multiplier_10
            config['algorithm'][f'lagrangian_multiplier_11'] = sweep_config.lagrangian_multiplier_11
            config['algorithm'][f'lagrangian_multiplier_12'] = sweep_config.lagrangian_multiplier_12
            config['algorithm'][f'lagrangian_multiplier_13'] = sweep_config.lagrangian_multiplier_13
            config['algorithm'][f'lagrangian_multiplier_14'] = sweep_config.lagrangian_multiplier_14
            config['algorithm'][f'lagrangian_multiplier_15'] = sweep_config.lagrangian_multiplier_15
            config['algorithm'][f'lagrangian_multiplier_16'] = sweep_config.lagrangian_multiplier_16
            config['algorithm'][f'lagrangian_multiplier_17'] = sweep_config.lagrangian_multiplier_17
            config['algorithm'][f'lagrangian_multiplier_18'] = sweep_config.lagrangian_multiplier_18
            config['algorithm'][f'lagrangian_multiplier_19'] = sweep_config.lagrangian_multiplier_19
            config['algorithm'][f'lagrangian_multiplier_20'] = sweep_config.lagrangian_multiplier_20
            config['algorithm'][f'lagrangian_multiplier_21'] = sweep_config.lagrangian_multiplier_21
            config['algorithm'][f'lagrangian_multiplier_22'] = sweep_config.lagrangian_multiplier_22
            config['algorithm'][f'lagrangian_multiplier_23'] = sweep_config.lagrangian_multiplier_23
            config['algorithm'][f'lagrangian_multiplier_24'] = sweep_config.lagrangian_multiplier_24


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
    with open('feas_tune_config.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep with W&B
    if parser.parse_args().sweep:
        sweep_id = parser.parse_args().sweep
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project="mpp_ppo")

    # Start the sweep agent, which runs the 'train' function with different hyperparameters
    wandb.agent(sweep_id, function=train, project="mpp_ppo", entity="stowage_planning_research")