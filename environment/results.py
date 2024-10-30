import matplotlib.pyplot as plt
import torch
import numpy as np

def rollout_results(env, out, td, batch_size, checkpoint_path, test_projection, utilization_rate_initial_demand):
    """Analyze the results of a rollout"""
    # Get metrics and reward
    metrics = env._get_metrics_n_step(td, out["utilization"], actions=out["actions"])
    total_revenue = env._compute_total_revenue(metrics["actions"], metrics["realized_demand"])
    total_costs = env._compute_total_costs(metrics["total_overstowage"], metrics["total_excess_crane_moves"])
    reward = total_revenue - total_costs

    # Get violations
    lhs_A, rhs = out["lhs_A"], out["rhs"]
    lhs = (lhs_A * out["actions"].unsqueeze(-2)).sum(dim=-1)
    violations = torch.clamp(lhs - rhs, min=0)
    # todo: analyze relative violations!
    # relative_violations = ((violations / rhs).nan_to_num(nan=0)).sum(dim=1).mean(dim=-1) # [batch_size, 5]
    # print(relative_violations)

    # Demand analysis
    demand = out["td"]["realized_demand"].view(*batch_size, -1)  # [batch_size, K*T]
    loaded = out["actions"].sum(dim=-1)  # [batch_size, K*T]
    backorders = demand - loaded
    demand_teu = (out["td"]["realized_demand"] * env.teus.view(1, -1, 1)).view(*batch_size, -1)  # [batch_size, K*T]
    loaded_teu = (out["actions"].sum(dim=-1).view(*batch_size, env.K, env.T) * env.teus.view(1, -1, 1)).view(
        *batch_size, -1)  # [batch_size, K*T]
    backorders_teu = demand_teu - loaded_teu
    # Efficiency analysis
    hatch_overstowage = metrics["total_overstowage"].view(*batch_size, -1)  # [batch_size,]
    long_crane_excess = metrics["total_excess_crane_moves"].view(*batch_size, -1)  # [batch_size,]

    # Revenue and cost analysis
    max_revenue = (env.revenues[:-1].view(1, env.K * env.T) * demand).sum(dim=-1)

    # Safe text results to file
    with open(checkpoint_path + f"/results_{test_projection}_rate_{utilization_rate_initial_demand}.txt", "w") as f:
        f.write("Overall analysis:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Reward: E[x]: {reward.mean()}, Std[x]: {reward.std()}\n")
        # f.write(f"Relative Violations: E[x]: {relative_violations.sum(dim=-1).mean()}, "
        #         f"Std[x]: {relative_violations.sum(dim=-1).std()}\n")
        # todo: add speed
        f.write("*" * 100 + "\n")
        f.write("Demand analysis:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Demand (#): E[x]: {demand.sum(dim=-1).mean()}, Std[x]: {demand.sum(dim=-1).std()}\n")
        f.write(f"Total Loaded (#): E[x]: {loaded.sum(dim=-1).mean()}, Std[x]: {loaded.sum(dim=-1).std()}\n")
        f.write(f"Backorders (#): E[x]: {backorders.sum(dim=-1).mean()}, Std[x]: {backorders.sum(dim=-1).std()}\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Demand (TEU): E[x]: {demand_teu.sum(dim=-1).mean()}, Std[x]: {demand_teu.sum(dim=-1).std()}\n")
        f.write(f"Total Loaded (TEU): E[x]: {loaded_teu.sum(dim=-1).mean()}, Std[x]: {loaded_teu.sum(dim=-1).std()}\n")
        f.write(f"Backorders (TEU): E[x]: {backorders_teu.sum(dim=-1).mean()}, Std[x]: {backorders_teu.sum(dim=-1).std()}\n")
        f.write("*" * 100 + "\n")
        f.write("Efficiency analysis:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Hatch Overstowage: E[x]: {hatch_overstowage.mean()}, "
                f"Std[x]: {hatch_overstowage.std()}\n")
        f.write(f"Total Long Crane Excess: E[x]: {long_crane_excess.mean()}, "
                f"Std[x]: {long_crane_excess.std()}\n")
        f.write("*" * 100 + "\n")
        f.write("Revenue and cost analysis:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Revenue (~$): E[x]: {total_revenue.mean()}, Std[x]: {total_revenue.std()}\n")
        f.write(f"Total Cost (~$): E[x]: {total_costs.mean()}, Std[x]: {total_costs.std()}\n")
        f.write("-" * 100 + "\n")
        f.write(f"Potential Revenue Load All (~$): E[x]: {max_revenue.mean()}, Std[x]: {max_revenue.std()}\n")
        # f.write(f"Min Cost (~$): E[x]: ?, Std[x]: ?\n")  # as low as possible, but not necessarily zero due to hatch constraints
    f.close()
    with open(checkpoint_path + f"/results_{test_projection}_rate_{utilization_rate_initial_demand}.txt", "r") as f:
        print(f.read())

    # Feasibility analysis in plots
    # Prepare data for visualization
    violations = violations.permute(2, 1, 0).detach().cpu().numpy() # [n_constraints, K*T, batch_size]
    teu_utilization = (out["utilization"] * env.teus.view(1, 1, 1, 1, -1, 1)).sum(dim=(2,3,4,5))
    remaining_capacity = (env.total_capacity - teu_utilization).permute(1,0).detach().cpu().numpy()  # [batch_size, K*T]
    save_path = checkpoint_path + f"/feasibility_analysis_{test_projection}_rate_{utilization_rate_initial_demand}.png"
    visualize_feasibility(demand_violations=violations[0], violations=violations[1:],
                          remaining_capacity=remaining_capacity, env=env, path=save_path)
    # Analyze action distribution at each time step
    actions = out["actions"].view(*batch_size, env.K*env.T, env.B, env.D)  # [batch_size, K, T, n_actions]
    e_x_actions = actions.mean(dim=0)  # [T, n_actions]
    std_x_actions = actions.std(dim=0)  # [T, n_actions]
    cv_actions = std_x_actions / e_x_actions  # [T, n_actions]
    # handle nan or inf values
    cv_actions[cv_actions == float('inf')] = 0
    cv_actions[cv_actions != cv_actions] = 0

    # Plot E[X] and CV over T with n_actions as subplots
    save_path = checkpoint_path + f"/e_x_d_0_action_analysis_{test_projection}_rate_{utilization_rate_initial_demand}.png"
    visualize_cv_actions(e_x_actions[...,0].detach().cpu().numpy(), env=env, title="E[X] Above Hatch", path=save_path)
    save_path = checkpoint_path + f"/e_x_d_1_action_analysis_{test_projection}_rate_{utilization_rate_initial_demand}.png"
    visualize_cv_actions(e_x_actions[...,1].detach().cpu().numpy(), env=env, title="E[X] Below Hatch", path=save_path)
    save_path = checkpoint_path + f"/cv_d_0_action_analysis_{test_projection}_rate_{utilization_rate_initial_demand}.png"
    visualize_cv_actions(cv_actions[...,0].detach().cpu().numpy(), env=env, title="CV Above Hatch", path=save_path)
    save_path = checkpoint_path + f"/cv_d_1_action_analysis_{test_projection}_rate_{utilization_rate_initial_demand}.png"
    visualize_cv_actions(cv_actions[...,1].detach().cpu().numpy(), env=env, title="CV Below Hatch", path=save_path)

def visualize_cv_actions(cv_actions, env, title="", path=None):
    """Visualize the coefficient of variation of actions taken by the agent"""
    # Create subplots: X rows, 1 column
    time = cv_actions.shape[0]
    bays = env.B
    fig, axs = plt.subplots(env.B, 1, figsize=(10, 12))

    # Set larger font sizes for labels, titles, and legends
    label_fontsize = 14
    title_fontsize = 16
    tick_fontsize = 12
    legend_fontsize = 12

    # Visualize actions if provided
    for b in range(bays):
        axs[b].plot(range(time), cv_actions[:, b,], label="CV Actions")
        # add every 4th value to the plot
        for i, txt in enumerate(cv_actions[:, b,]):
            if i % 4 == 0:
                axs[b].annotate(f"{txt:.2f}", (i, txt), textcoords="offset points", xytext=(0, 10), ha='center')

        axs[b].set_ylabel("Actions", fontsize=label_fontsize)
        axs[b].set_xlabel("Episodic time steps", fontsize=label_fontsize)
        axs[b].set_title(f"Actions Bay {b}", fontsize=title_fontsize)
        axs[b].tick_params(axis='both', labelsize=tick_fontsize)

    # Set all ylimits to be the same; take highest value
    max_cv = cv_actions.max()
    for ax in axs:
        ax.set_ylim(0, max_cv)
        ax.label_outer()

    # Title, layout, and save
    fig.suptitle(f"{title}", fontsize=title_fontsize)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

def visualize_feasibility(violations=None, remaining_capacity=None, demand_violations=None, env=None, path=None):
    """Visualize stability and capacity feasibility violations in subplots with larger text"""

    # Create subplots: 3 rows, 1 column
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Set larger font sizes for labels, titles, and legends
    title_fontsize = 16
    label_fontsize = 12
    tick_fontsize = 12
    legend_fontsize = 12

    # Visualize remaining capacity if provided
    if remaining_capacity is not None:
        mean = remaining_capacity.mean(axis=-1)
        std = remaining_capacity.std(axis=-1)
        axs[0].plot(range(len(mean)), mean, label="Remaining Capacity")
        axs[0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        axs[0].set_ylabel("TEU", fontsize=label_fontsize)
        axs[0].set_xlabel("Episodic time steps", fontsize=label_fontsize)
        axs[0].set_title("Capacity Constraint", fontsize=title_fontsize)
        axs[0].tick_params(axis='both', labelsize=tick_fontsize)
        axs[0].legend(fontsize=legend_fontsize, loc='upper left')


    # Visualize demand violations if provided
    if demand_violations is not None:
        mean = demand_violations.mean(axis=-1)
        std = demand_violations.std(axis=-1)
        axs[1].plot(range(len(mean)), mean, label="Demand violation")
        axs[1].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        axs[1].set_ylabel("Containers", fontsize=label_fontsize)
        axs[1].set_xlabel("Episodic time steps", fontsize=label_fontsize)
        axs[1].set_title("Demand Constraint", fontsize=title_fontsize)
        axs[1].tick_params(axis='both', labelsize=tick_fontsize)

    # Visualize violations if provided
    # todo:  allow multiple lines to clearly see the difference
    # todo: improve y-label
    if violations is not None:
        labels = ["LCG UB", "LCG LB", "VCG UB", "VCG LB"]
        for idx, constraint in enumerate(violations):
            mean = constraint.mean(axis=-1)
            std = constraint.std(axis=-1)
            axs[2].plot(range(len(mean)), mean, label=labels[idx])
            axs[2].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        axs[2].set_ylabel("Bound violation", fontsize=label_fontsize)
        axs[2].set_xlabel("Episodic time steps", fontsize=label_fontsize)
        axs[2].set_title("Stability Constraints", fontsize=title_fontsize)
        axs[2].tick_params(axis='both', labelsize=tick_fontsize)
        axs[2].legend(fontsize=legend_fontsize, loc='upper center')

    # Indicate steps that vessels depart from port
    if env is not None:
        depart_from_port = np.cumsum([x * env.K for x in range(env.P - 1, 0, -1)]) - 1
        for i in depart_from_port:
            for ax in axs:  # Add departure lines to all subplots
                ax.axvline(x=i, color='y', linestyle='--', alpha=0.5, label='Departure from port')

    # Adjust the layout
    plt.tight_layout()

    # Create a unique legend for the departure line
    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if unique:
        axs[0].legend(*zip(*unique), fontsize=legend_fontsize)

    # Save the plot if path is provided
    if path is not None:
        plt.savefig(path)

    # Show the plot
    plt.show()

