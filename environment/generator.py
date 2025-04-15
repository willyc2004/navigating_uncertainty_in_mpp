# Imports
import torch as th
from typing import Tuple, Optional
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator
from environment.utils import get_transport_idx, get_load_transport
import matplotlib.pyplot as plt

# Classes
class MPP_Generator(Generator):
    """
    Demand generator for the master planning problem (MPP) in container stowage planning.

    This generator simulates cargo demands across multiple transport legs
    and allows configurations for uncertainty, customer and cargo classes, and different sampling strategies.
    """
    def __init__(self, device="cuda", **kwargs):
        """
        Initialize the MPP_Generator.

        Args:
            device (str): Device to run computations on ('cuda' or 'cpu').
            **kwargs: Configuration parameters for the generator:
                - seed (int): Random seed.
                - ports (int): Number of ports in the voyage.
                - bays (int): Number of bays in the vessel.
                - decks (int): Number of decks (0=deck, 1=hold).
                - customer_classes (int): Number of customer contract classes.
                - cargo_classes (int): Number of cargo types per customer.
                - weight_classes (int): Number of weight classes.
                - capacity (list[int] or torch.Tensor): Per-bay/deck capacity.
                - utilization_rate_initial_demand (float): Capacity utilization target.
                - spot_percentage (float): Ratio of spot contract cargo.
                - iid_demand (bool): Use IID sampling if True, otherwise GBM.
                - cv_demand (float): Coefficient of variation for demand.
                - demand_uncertainty (bool): Enable demand uncertainty.
                - generalization (bool): Enable uniform distribution sampling.
                - perturbation (float): Perturbation scale for stochastic effects.
        """
        super().__init__(**kwargs)
        # Input simulation
        self.device = th.device(device)
        self.seed = kwargs.get("seed")
        self.rng = th.Generator(device=self.device).manual_seed(self.seed)

        # Configuration
        self.P = kwargs.get("ports")
        self.B = kwargs.get("bays")
        self.D = kwargs.get("decks")
        self.CC = kwargs.get("customer_classes")
        self.K = kwargs.get("cargo_classes") * self.CC
        self.W = kwargs.get("weight_classes")
        c = kwargs.get("capacity")
        self.c = th.full((self.B, self.D,), c[0]) if len(c) == 1 else c
        self.total_capacity = th.sum(self.c)

        # Derived values
        self.T = int((self.P ** 2 - self.P) / 2)
        self.teus = th.arange(1, self.K // (self.CC * self.W) + 1, dtype=th.float16, device=self.device)\
            .repeat_interleave(self.W).repeat(self.CC)

        # Demand control
        self.utilization_rate_initial_demand = kwargs.get("utilization_rate_initial_demand", 1.2)
        self.spot_percentage = kwargs.get("spot_percentage", 0.3)
        self.spot_lc_percentage = th.cat([
            th.full((self.K // 2,), 2 * (1 - self.spot_percentage), device=self.device),
            th.full((self.K // 2,), 2 * self.spot_percentage, device=self.device)
        ])
        self.iid_demand = kwargs.get("iid_demand", True)
        self.cv_demand = kwargs.get("cv_demand", 0.5)
        self.demand_uncertainty = kwargs.get("demand_uncertainty", False)
        self.generalization = kwargs.get("generalization", False)
        self.perturbation = kwargs.get("perturbation", 0.1)

        # Demand variability
        self.cv = th.empty((self.K,), device=self.device, dtype=th.float16)
        self.cv[:self.K // 2] = 0.5
        self.cv[self.K // 2:] = 0.3

        # Precomputations
        self.train_max_demand = 0
        self.wave = self._create_wave(self.P - 1)
        self.transport_idx = get_transport_idx(self.P, device=self.device)
        self.num_loads = self._get_num_loads_in_voyage(self.transport_idx, self.P)
        self.num_discharges = self.num_loads.flip(0)
        POL = th.arange(self.P, device=self.device).unsqueeze(1).unsqueeze(1)
        self.num_ac = self._get_num_AC_in_voyage(self.transport_idx, POL)
        self.num_ob = self._get_num_OB_in_voyage(self.transport_idx, POL)
        self.tr_wave = th.repeat_interleave(self.wave, self.num_loads)
        self.tr_loads = th.repeat_interleave(self.num_loads, self.num_loads)
        self.tr_discharges = th.repeat_interleave(self.num_discharges, self.num_loads)
        self.tr_ac = th.repeat_interleave(self.num_ac, self.num_loads)
        self.tr_ob = th.repeat_interleave(self.num_ob, self.num_loads)

    def __call__(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, rng:Optional=None) -> TensorDict:
        """Generate demand for the MPP."""
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size, td)

    ## Generate demand
    def _gbm_lognormal_distribution(self, t:th.Tensor, s0:th.Tensor, mu=th.tensor(0.0000), sigma=th.tensor(0.01),) \
            -> Tuple[th.Tensor, th.Tensor, th.distributions.LogNormal]:
        """
        Obtain a log-normal distribution that corresponds to the GBM process for each element in s0.

        Parameters:
        s0 (tensor): Initial values of the GBM process
        mu (float): Drift rate.
        sigma (float): Volatility.
        t (float): Time duration.

        Returns:
        tensor: Array of log-normal samples corresponding to the GBM process.
        """
        # Calculate parameters for the log-normal distribution
        mu_log = th.log(s0) + (mu - 0.5 * sigma ** 2) * t[0]
        sigma_log = sigma * th.sqrt(t[0])

        # Convert to mu and sigma to regular scale
        mean = th.exp(mu_log + 0.5 * sigma_log ** 2)
        variance = (th.exp(sigma_log ** 2) - 1) * th.exp(2 * mu_log + sigma_log ** 2)
        std_dev = th.sqrt(variance)

        # Generate log-normal samples for each initial value
        log_dist = th.distributions.LogNormal(loc=mu_log, scale=sigma_log)
        return mean, std_dev, log_dist

    def _iid_normal_distribution(self, e_x: th.Tensor, std_x: th.Tensor,) -> Tuple[th.Tensor, th.Tensor, th.distributions.Normal]:
        """Get normal distribution for demand"""
        return e_x, std_x, th.distributions.Normal(loc=e_x, scale=std_x)

    def _create_std_x(self, e_x:th.Tensor, cv:float=0.5) -> th.Tensor:
        """Create std_x from coefficient of variation: cv < 0.1 is low, 0.3<x<0.5 is moderate, >0.5 is high"""
        return e_x * cv

    def _generalization_uniform_distribution(self, mu:th.Tensor, sigma:th.Tensor) -> th.distributions.Uniform:
        """Provided the mu and sigma of a Gaussian distribution, we can create an equivalent uniform distribution.
        Let's equate mu and sigma^2 to a,b parameters of the uniform distribution:
        - mu = (a + b) / 2
        - sigma^2 = (b - a)^2 / 12

        Using some algebra, we can obtain the following:
        - a = mu - sqrt(12 sigma**2)/2
        - b = mu + sqrt(12 sigma**2)/2

        Now, we get uniform distribution bounds [a,b] for generalization."""
        a = mu - th.sqrt(12 * sigma ** 2)/2
        b = mu + th.sqrt(12 * sigma ** 2)/2
        dist = th.distributions.Uniform(a, b)
        return dist

    def _generate(self, batch_size:Tuple[int, ...], td:Optional=None,) -> TensorDict:
        """Generate demand matrix for voyage with GBM process"""
        # Get initial demand if not provided
        if td is None or td.is_empty():
            bound = self._initialize_demand_bound_on_capacity(batch_size)
            e_x_init_demand, _ = self._generate_initial_moments(batch_size, bound, self.cv)
            batch_updates = th.zeros(batch_size, device=self.device).view(*batch_size, 1)
            self.train_max_demand = self._get_ub_demand_normalization(e_x_init_demand)
        else:
            e_x_init_demand = td["observation", "init_expected_demand"].view(-1, self.T, self.K)
            batch_updates = td["observation", "batch_updates"].clone() + 1

        # Get moments and distribution
        if not self.iid_demand:
            e_x, std_x, dist = self._gbm_lognormal_distribution(batch_updates, e_x_init_demand,)
        else:
            std_x = self._create_std_x(e_x_init_demand, self.cv_demand)
            e_x, std_x, dist = self._iid_normal_distribution(e_x_init_demand, std_x,)

        if self.generalization:
            dist = self._generalization_uniform_distribution(e_x, std_x)

        # Sample demand
        demand = th.clamp(dist.sample(), min=1)

        # Return demand matrix
        return TensorDict({"observation":
                               {"realized_demand": demand.view(*batch_size, self.T*self.K),
                                "expected_demand": e_x.view(*batch_size, self.T*self.K),
                                "std_demand":std_x.view(*batch_size, self.T*self.K),
                                "init_expected_demand": e_x_init_demand.view(*batch_size, self.T*self.K),
                                "batch_updates":batch_updates.clone(),}}, batch_size=batch_size, device=self.device,)

    ## Initial demand
    def _initialize_demand_bound_on_capacity(self, batch_size:Tuple[int, ...],) -> th.Tensor:
        """Get initial demand bound based on capacity

        Bound without wave:
        f(x) = (2*utilization_rate * th.sum(self.c)) / (self.K *  num_load (1-num_ac/num_ob))

        This computes the bound for each (K,T) pair assuming full capacity utilization. In reality, however,
        the utilization rate is only approaching 80-90% of the total capacity at the middle of the voyage. Hence,
        we introduce a wave parameter to increase the bound until the middle of the voyage and decrease it afterward.
        Note: 2*utilization_rate is used to account for the fact this is an upper bound of uniform distribution.

        Bound with wave:
        g(x) = wave * (2*utilization_rate * th.sum(self.c)) / (self.K *  num_load (1-num_ac/num_ob))

        Note that if wave is added, then capacity used in previous steps is underestimated with wave < 1. Similarly,
        if wave >1 then capacity used in previous steps is overestimated. Nonetheless, this bound used for an initial
        E[X] and V[X], hence this is acceptable if we tune wave parameters to obtain sufficient levels of utilization.
        """
        # Get transport bound with wave
        utilization_bound = self.tr_wave * 2 * self.utilization_rate_initial_demand * self.total_capacity
        num_cargo = (self.K * self.tr_loads / (1 - self.tr_ac / self.tr_ob))
        utilization_bound /= num_cargo
        # Get bound with spot, lc using self.spot_lc_percentage
        return th.ger(utilization_bound, self.spot_lc_percentage) / self.teus.view(1, -1,)

    def _random_perturbation(self, input:th.Tensor, perturb_factor=0.1) -> th.Tensor:
        """Apply a random perturbation to bound while keeping it close to the original value."""
        perturbation = 1 + (th.rand_like(input) * 2 - 1) * perturb_factor  # U(1-α, 1+α)
        return th.clamp(input * perturbation, min=1.0)  # Ensure positive output

    def _generate_initial_moments(self, batch_size:Tuple[int, ...], bound:th.Tensor,
                                  cv:th.Tensor, eps:float=1e-2) -> Tuple[th.Tensor, th.Tensor]:
        """Generate initial E[X] and V[X] for spot and longterm contracts.
        - E[X] = Uniform sample * bound
        - V[X] = (E[X] * cv)^2"""
        # shape of bound: [batch_size, T, K]
        if len(bound.shape) == 2:
            bound = bound.unsqueeze(0)

        # Sample uniformly from 0 to bound (inclusive) using torch.rand
        expected = th.rand(*batch_size, self.T, self.K, dtype=bound.dtype, device=self.device,
                           generator=self.rng) * bound
        variance = (expected * cv.view(1, 1, self.K,)) ** 2
        return th.where(expected < eps, eps, expected), variance

    # Support functions
    def _create_wave(self, length:int, param:float=0.3, ) -> th.Tensor:
        """Create a wave function for the bound"""
        mid_index = length // 2
        increasing_values = 1 + param * th.cos(th.linspace(th.pi, th.pi / 2, steps=mid_index + 1,
                                                           device=self.device, dtype=th.float32, ))
        decreasing_values = 1 + param * th.cos(th.linspace(th.pi / 2, th.pi, steps=length - mid_index,
                                                           device=self.device, dtype=th.float32, ))
        return th.cat([increasing_values[:-1], decreasing_values, th.zeros((1,), device=self.device, dtype=th.float32)])

    def _get_num_loads_in_voyage(self, transport_idx:th.Tensor, P:int, ) -> th.Tensor:
        """Get number of transports loaded per POL"""
        # Create a boolean mask for load pairs using broadcasting and advanced indexing
        load_mask = th.zeros((P, P), dtype=th.bool, device=self.device, )
        load_mask[transport_idx[:, 0], transport_idx[:, 1]] = True
        # Count loads for each port
        return load_mask.sum(dim=1)

    def _get_num_AC_in_voyage(self, transport_idx:th.Tensor, POL:int, ) -> th.Tensor:
        """Get number of transport in arrival condition per POL"""
        mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

    def _get_num_OB_in_voyage(self, transport_idx:th.Tensor, POL:int, ) -> th.Tensor:
        """Get number of transports in onbord per POL"""
        mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

    def _get_ub_demand_normalization(self, bound:th.Tensor, eps:float=1e-2) -> th.Tensor:
        """Get upper bound for demand normalization"""
        return (bound + 4 * (bound / 2 * 0.5)).max()

class UniformMPP_Generator(MPP_Generator):
    """Class for generating demand for stowage plans using uniform distribution."""

    def __call__(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, rng:Optional=None) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size, td)

    def _generate(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, ) -> TensorDict:
        """Generate demand matrix for voyage with uniform distribution"""
        # Get initial demand if not provided
        if td is None or td.is_empty():
            # Get initial demand bound based on capacity
            bound = self._initialize_demand_bound_on_capacity(batch_size)
            self.train_max_demand = self._get_ub_demand_normalization(bound/2)
            if batch_size != []: bound = bound.unsqueeze(0).expand(*batch_size, -1, -1) # Expand to batch size

            # Get initial demand based on random perturbed bound
            bound = self._random_perturbation(bound, self.perturbation)
            demand, _ = self._generate_initial_moments(batch_size, bound, self.cv)
            # Get moments from uniform distribution
            e_x = (bound * 0.5).expand_as(demand)
            init_e_x = e_x.clone()
            std_x = bound / th.sqrt(th.tensor(12, device=self.device)).expand_as(demand)
            batch_updates = th.zeros(batch_size, device=self.device).view(*batch_size, 1)
        else:
            demand = td["observation", "realized_demand"].view(-1, self.T, self.K)
            e_x = td["observation", "expected_demand"].view(-1, self.T, self.K)
            init_e_x = td["observation", "init_expected_demand"].view(-1, self.T, self.K)
            std_x = td["observation", "std_demand"].view(-1, self.T, self.K)
            batch_updates = td["observation", "batch_updates"].clone() + 1

        if not self.iid_demand:
            e_x, std_x, _ = self._gbm_lognormal_distribution(batch_updates, init_e_x,)

        # Return demand matrix
        return TensorDict({"observation":
                               {"realized_demand": demand.view(*batch_size, self.T * self.K),
                                "expected_demand": e_x.view(*batch_size, self.T * self.K),
                                "init_expected_demand": init_e_x.view(*batch_size, self.T * self.K),
                                "std_demand": std_x.view(*batch_size, self.T * self.K),
                                "batch_updates": batch_updates.clone(), }}, batch_size=batch_size,
                          device=self.device, )

def plot_demand_history(demand_history:th.Tensor, updates:int,
                        y_label:str="Containers", title:str="Container Demand History", summarize:bool=False):
    """Plot demand history"""
    plt.figure()
    demand_history = demand_history.detach().cpu().numpy()
    if summarize:
        # Plot standard deviation
        plt.fill_between(range(updates),
                         demand_history.sum(axis=(-1, -2)).mean(axis=(1,)) -
                         demand_history.sum(axis=(-1, -2)).std(axis=(1,)),
                         demand_history.sum(axis=(-1, -2)).mean(axis=(1,)) +
                         demand_history.sum(axis=(-1, -2)).std(axis=(1,)), alpha=0.3, label="Mean +/- Std")
        # Add maximum and minimum
        plt.fill_between(range(updates),
                         demand_history.sum(axis=(-1, -2)).max(axis=1),
                         demand_history.sum(axis=(-1, -2)).min(axis=1), alpha=0.3, label="Max-Min Range", color="grey")
    else:
        # Plot all demand rollouts histories
        for i in range(demand_history.size(1)):
            plt.plot(demand_history[:, i].sum(axis=(-1,-2)), alpha=0.3)
    # Plot mean total demand
    plt.plot(demand_history.sum(axis=(-1, -2)).mean(axis=(1,)), label="Mean")

    # Add labels
    plt.ylim(0, demand_history.sum(axis=(-1, -2)).max() + 20)
    plt.xlabel("Batch Updates")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()

# visualize demand of UniformMPP_Generator
if __name__ == "__main__":
    # Parameters
    import os
    from main import load_config
    from rl_algorithms.utils import make_env
    from rl_algorithms.test import compute_summary_stats

    file_path = "/mnt/c/Users/jaiv/PycharmProjects/DRL_master_planning_problem"
    config = load_config(f'{file_path}/config.yaml')
    env = make_env(config.env, device='cpu')
    batch_size = 30

    # Create generator
    generator = env.generator

    # Generate demand
    td = generator(batch_size)
    # demand_history = th.empty((batch_size, env.T, env.K), device="cpu")
    td = generator(batch_size, td)
    demand = td["observation", "realized_demand"].view(batch_size, env.T, env.K)

    # EDA of all types
    demand_dict = {}
    for i in range(env.T):
        for j in range(env.K):
            demand_dict[f"transport_{i}_type_{j}"] = demand[:, i, j]
    summary_stats = compute_summary_stats(demand_dict)
    # print(summary_stats)

    # Histogram of aggregated demand
    demand_port = th.zeros((batch_size, env.P-1), device="cpu")
    teu_port = th.zeros((batch_size, env.P-1), device="cpu")
    for i in range(env.P-1):
        for j in range(i+1, env.P):
            condition = (env.transport_idx[:, 0] == i) & (env.transport_idx[:, 1] == j)
            index = th.where(condition)[0]  # get indices where the condition is true
            demand_port[:, i] += demand[:, index, :].sum(dim=(-1,)).squeeze()
            teu_port[:, i] += (env.teus * demand[:, index, :]).sum(dim=(-1,)).squeeze()

    # Give mean, std, max, min of demand at each port
    print("Mean of demand at each port: ", demand_port.mean(dim=0))
    print("Std of demand at each port: ", demand_port.std(dim=0))
    print("Max of demand at each port: ", demand_port.max(dim=0))
    print("Min of demand at each port: ", demand_port.min(dim=0))

    # (Batch, port) shape;
    # Plot boxplot at each port
    demand_port = demand_port.detach().cpu().numpy()
    plt.figure()
    plt.boxplot(demand_port)
    plt.xlabel("Port")
    plt.ylabel("Containers")
    plt.ylim(0, demand_port.max() + 1000)
    plt.title("Container Demand at Each Port")
    plt.show()
    # Plot boxplot of TEU at each port
    teu_port = teu_port.detach().cpu().numpy()
    plt.figure()
    plt.boxplot(teu_port)
    plt.xlabel("Port")
    plt.ylabel("TEU")
    plt.ylim(0, teu_port.max() + 1000)
    plt.title("TEU Demand at Each Port")
    plt.show()
