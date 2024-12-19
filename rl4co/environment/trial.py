import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
import pandas as pd
from rl4co.rl4co.decoding import rollout_mpp


# Aggregate all data for each feature across runs and steps
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
    # Convert the torch correlation matrix to a pandas DataFrame
    corr_df = pd.DataFrame(correlation_matrix.numpy(), index=feature_names, columns=feature_names)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print("Correlation Matrix:")
    print(corr_df.round(4))  # Round values for better readability


def trial(env, td, device, num_rollouts=3, EDA=False, profiling=False):
    """Run a trial of the environment"""
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
    if profiling:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
                     record_shapes=True) as prof:
            for idx in range(num_rollouts):
                with record_function("Rollout Iteration"):
                    start_time = time.time()
                    reward, td, actions, results = rollout_mpp(
                        env,
                        env.reset(batch_size=td.batch_size, td=td),
                        random_action_policy,
                        EDA
                    )
                    runtime = time.time() - start_time
                    runtimes_rollout[idx] = runtime
                    run_results[idx] = results

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    else:
        for idx in range(num_rollouts):
            start_time = time.time()
            reward, td, actions, results = rollout_mpp(env, env.reset(batch_size=td.batch_size, td=td), random_action_policy, EDA)
            runtime = time.time() - start_time
            runtimes_rollout[idx] = runtime
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

    # Exploratory Data Analysis
    if EDA:
        feature_data = aggregate_feature_data(run_results)
        reshaped_features = reshape_features(feature_data)

        # Calculate multi-dimensional correlation matrix
        correlation_matrix, feature_names = calculate_multi_dim_correlation(reshaped_features)
        find_all_nan_columns(correlation_matrix, feature_names)
        # Export the correlation matrix to csv
        corr_matrix = correlation_matrix.detach().cpu().numpy()
        corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
        corr_df.to_csv("EDA/correlation_matrix.csv")

        # Initialize an empty list to collect all summary statistics
        summary_list = []

        # Loop through each feature, compute summary statistics, and add to the list
        for feature, tensor in reshaped_features.items():
            summary = {
                "feature": feature,
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item()
            }
            # Append the summary dictionary to the list
            summary_list.append(summary)

        # Convert the list of summaries to a DataFrame
        all_summaries = pd.DataFrame(summary_list)

        # Save all summary statistics to a single CSV file
        all_summaries.to_csv("EDA/all_features_summary.csv", index=False)