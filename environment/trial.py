import time
import torch
from models.decoding import rollout_mpp
from environment.data import StateDependentDataset, custom_collate_fn
from torch.utils.data import DataLoader
from tensordict import TensorDict

def trial(env, td, device, num_rollouts=3):
    """Run a trial of the environment"""

    # Run profiling on the environment
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    #         record_shapes=True,
    #         profile_memory=True
    # ) as prof:
    #     for idx in range(num_rollouts):
    #         td = env.reset(batch_size=td.batch_size, td=td)
    #         done = td["done"][0]
    #         while not done:
    #             td = random_action_policy(td)
    #             next_td = env.step(td)["next"]
    #             prof.step()  # Update the profiler
    #             td = next_td
    #             done = td["done"][0]

    # Test dataset
    batch_size = [1024]
    total_samples = 4_000_000
    dataset = StateDependentDataset(env, td, total_samples, batch_size)
    print(f"Number of batches: {len(dataset)}")

    # Test the environment
    def random_action_policy(td):
        """Helper function to select a random action from available actions"""
        batch_size = td.batch_size
        action = torch.distributions.Uniform(env.action_spec.low, env.action_spec.high).sample(batch_size) / 10
        td.set("action", action.to(torch.float16))
        return td

    # Rollout the environment with random actions
    run_results = {}

    runtimes_rollout = torch.zeros(num_rollouts, dtype=torch.float16, device=device)
    for idx in range(num_rollouts):
        with torch.cuda.amp.autocast():
            start_time = time.time()
            reward, td, actions, results = rollout_mpp(env, env.reset(batch_size=td.batch_size, td=td), random_action_policy)
            runtime = time.time() - start_time
            runtimes_rollout[idx] = runtime

            # Get
            run_results[idx] = results


    # Print statistics
    print("#"*50)
    print(f"Runtime statistics for {num_rollouts} rollouts")
    print("-"*50)
    print(f"Mean runtime: {torch.mean(runtimes_rollout):.3f} s")
    print(f"Std runtime: {torch.std(runtimes_rollout):.3f} s")
    print(f"Max runtime: {torch.max(runtimes_rollout):.3f} s")
    print(f"Min runtime: {torch.min(runtimes_rollout):.3f} s")
    print("*"*50)
    # todo: add correlation between state representation, reward, revenue, cost, infeasibility

    # Aggregate all data for each feature across runs and steps
    from collections import defaultdict

    def aggregate_feature_data(runs):
        """
        Aggregates tensors for each feature across all runs and steps.
        """
        feature_data = defaultdict(list)
        for run, steps in runs.items():
            for step in steps:
                for feature, tensor in step.items():
                    feature_data[feature].append(tensor)
        return feature_data

    def reshape_features(feature_data):
        """
        Concatenates tensors for each feature across runs and steps, preserving original feature dimensions.
        Checks for NaNs in each reshaped feature.
        """
        reshaped_features = {}
        features_with_nans = set()
        for feature, tensors in feature_data.items():
            # Concatenate along the sample dimension without flattening
            concatenated = torch.cat(tensors, dim=0)
            reshaped_features[feature] = concatenated

            # Check if NaNs exist in the reshaped feature tensor
            if torch.isnan(concatenated).any() or torch.isinf(concatenated).any():
                features_with_nans.add(feature)

        # Print or return features with NaNs for inspection
        if features_with_nans:
            print("Features containing NaN/Inf values:", features_with_nans)
        else:
            print("No Nan/Inf values found in any features.")

        return reshaped_features

    def calculate_multi_dim_correlation(features_dict):
        """
        Calculate a pairwise correlation matrix for multi-dimensional features.
        Reduces features to a single scalar per sample.
        """
        feature_names = list(features_dict.keys())
        num_features = len(feature_names)
        correlation_matrix = torch.zeros((num_features, num_features))

        for i, feature_i in enumerate(feature_names):
            for j, feature_j in enumerate(feature_names):
                tensor_i = features_dict[feature_i]
                tensor_j = features_dict[feature_j]

                # Reduce each multi-dimensional tensor to a single scalar per sample
                if tensor_i.dim() > 1:
                    tensor_i = tensor_i.flatten(start_dim=1).mean(dim=1)
                if tensor_j.dim() > 1:
                    tensor_j = tensor_j.flatten(start_dim=1).mean(dim=1)

                # Center the data
                centered_i = tensor_i - tensor_i.mean()
                centered_j = tensor_j - tensor_j.mean()

                # Calculate norms
                norm_i = torch.norm(centered_i)
                norm_j = torch.norm(centered_j)

                # Avoid division by zero
                if norm_i == 0 or norm_j == 0:
                    correlation_matrix[i, j] = float('nan')
                else:
                    # Frobenius-based correlation
                    frobenius_inner_product = (centered_i * centered_j).sum()
                    correlation_matrix[i, j] = frobenius_inner_product / (norm_i * norm_j)

        return correlation_matrix, feature_names

    def find_all_nan_columns(correlation_matrix, feature_names):
        """
        Identifies and prints feature names with a column of all NaN values in the correlation matrix.
        """
        nan_columns = []

        for i, feature in enumerate(feature_names):
            # Check if the entire column i is NaN
            if torch.isnan(correlation_matrix[:, i]).all():
                nan_columns.append(feature)

        if nan_columns:
            print("Features with a column of all NaN values in the correlation matrix:", nan_columns)
        else:
            print("No columns with all NaN values found in the correlation matrix.")

    def print_correlation_matrix_neatly(correlation_matrix, feature_names):
        """
        Prints the correlation matrix in a neat, tabular format using pandas DataFrame.
        """
        import pandas as pd
        # Convert the torch correlation matrix to a pandas DataFrame
        corr_df = pd.DataFrame(correlation_matrix.numpy(), index=feature_names, columns=feature_names)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print("Correlation Matrix:")
        print(corr_df.round(4))  # Round values for better readability

    # Aggregate and reshape feature data
    feature_data = aggregate_feature_data(run_results)
    reshaped_features = reshape_features(feature_data)

    # Calculate multi-dimensional correlation matrix
    correlation_matrix, feature_names = calculate_multi_dim_correlation(reshaped_features)
    find_all_nan_columns(correlation_matrix, feature_names)
    # Export to excel
    import pandas as pd
    corr_matrix = correlation_matrix.detach().cpu().numpy()
    df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    df.to_excel("correlation_matrix.xlsx")

    # print_correlation_matrix_neatly(correlation_matrix, feature_names)

    # Give summary statistics for each feature
    for feature, tensor in reshaped_features.items():
        print("-" * 50)
        print(f"Feature: {feature}")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean():.3f}")
        print(f"Std: {tensor.std():.3f}")
        print(f"Min: {tensor.min():.3f}")
        print(f"Max: {tensor.max():.3f}")




