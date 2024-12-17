# Imports
import torch as th
from typing import Tuple, Optional
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator
from environment.utils import get_transport_idx, get_load_transport
import matplotlib.pyplot as plt

# Classes
class MPP_Generator(Generator):
    """Class for generating demand for stowage plans"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Input simulation
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.seed = kwargs.get("seed")

        # Input env
        self.P = kwargs.get("ports")  # Number of ports
        self.B = kwargs.get("bays")  # Number of bays
        self.D = kwargs.get("decks")  # Number of decks (on-deck=0, hold=1)
        self.T = int((self.P ** 2 - self.P) / 2) # Number of (POL,POD) transport groups
        self.CC = kwargs.get("customer_classes")  # Number of customer contracts
        self.K = kwargs.get("cargo_classes") * self.CC  # Number of cargo types
        self.W = kwargs.get("weight_classes")  # Number of weight classes
        c = kwargs.get("capacity")
        self.c = th.full((self.B, self.D,), c[0]) if len(c) == 1 else c  # Container capacity per bay/deck
        self.total_capacity = th.sum(self.c)
        self.teus = th.arange(1, self.K // (self.CC * self.W) + 1, dtype=th.float16, device=self.device)\
            .repeat_interleave(self.W).repeat(self.CC)

        # Precompute
        self.utilization_rate_initial_demand = kwargs.get("utilization_rate_initial_demand", 1.2)
        self.spot_percentage = kwargs.get("spot_percentage", 0.3)
        self.spot_lc_percentage = th.cat([
                     th.full((self.K//2,), 2*(1 - self.spot_percentage), device=self.device, dtype=th.float32),
                     th.full((self.K//2,), 2*(self.spot_percentage), device=self.device, dtype=th.float32)],  dim=0)
        self.iid_demand = kwargs.get("iid_demand", True)
        self.cv_demand = kwargs.get("cv_demand", 0.5)
        self.demand_uncertainty = kwargs.get("demand_uncertainty", False)

        # Get cv vector
        self.cv = th.empty((self.K,), device=self.device, dtype=th.float16,)
        self.cv[:self.K//2] = 0.5
        self.cv[self.K//2:] = 0.3

        # Precompute wave and transport
        self.wave = self._create_wave(self.P-1,)
        self.transport_idx = get_transport_idx(self.P, device=self.device)
        self.num_loads = self._get_num_loads_in_voyage(self.transport_idx, self.P)
        self.num_discharges = self.num_loads.flip(0)
        POL = th.arange(self.P, device=self.device,).unsqueeze(1).unsqueeze(1) # Shape: [P, 1, 1]
        self.num_ac = self._get_num_AC_in_voyage(self.transport_idx, POL,)
        self.num_ob = self._get_num_OB_in_voyage(self.transport_idx, POL,)
        # Transform in shape [T]
        self.tr_wave = th.repeat_interleave(self.wave, self.num_loads)
        self.tr_loads = th.repeat_interleave(self.num_loads, self.num_loads)
        self.tr_discharges = th.repeat_interleave(self.num_discharges, self.num_loads)
        self.tr_ac = th.repeat_interleave(self.num_ac, self.num_loads)
        self.tr_ob = th.repeat_interleave(self.num_ob, self.num_loads)

    def __call__(self, batch_size, td: Optional[TensorDict] = None, rng:Optional=None) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size, td)

    ## Generate demand
    def _gbm_lognormal_distribution(self, t:th.Tensor, s0:th.Tensor, mu=th.tensor(0.00), sigma=th.tensor(0.01),):
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

    def _iid_normal_distribution(self, e_x, std_x,):
        """Get normal distribution for demand"""
        return e_x, std_x, th.distributions.Normal(loc=e_x, scale=std_x)

    def _create_std_x(self, e_x, cv=0.5):
        """Create std_x from coefficient of variation: cv < 0.1 is low, 0.3<x<0.5 is moderate, >0.5 is high"""
        return e_x * cv

    def _generate(self, batch_size, td:Optional=None,) -> TensorDict:
        """Generate demand matrix for voyage with GBM process"""
        # Get initial demand if not provided
        if td is None or td.is_empty():
            e_x_init_demand, _ = self._initial_contract_demand(batch_size)
            batch_updates = th.zeros(batch_size, device=self.device).view(*batch_size, 1)
        else:
            e_x_init_demand = td["init_expected_demand"].view(-1, self.T, self.K)
            batch_updates = td["batch_updates"].clone()

        # Get moments and distribution
        if not self.iid_demand:
            e_x, std_x, dist = self._gbm_lognormal_distribution(batch_updates + 1, e_x_init_demand,)
        else:
            std_x = self._create_std_x(e_x_init_demand, self.cv_demand)
            e_x, std_x, dist = self._iid_normal_distribution(e_x_init_demand, std_x,)

        # Sample demand
        demand = th.clamp(dist.sample(), min=1)

        # # Manually generate random numbers using the generator
        # random_tensor = th.rand(e_x.shape, generator=self.rng, device=self.device)
        # # Transform using the inverse CDF (ppf)
        # demand = dist.icdf(random_tensor)
        # demand = th.clamp(demand, min=1)

        # Observed demand: only transports of POL=0
        load_tr = get_load_transport(self.transport_idx, th.zeros((1,), device=self.device,))
        observed_demand = th.zeros_like(demand)
        if self.demand_uncertainty:
            observed_demand[:, load_tr, :] = demand[:, load_tr, :]
        else:
            observed_demand = demand

        # Return demand matrix
        return TensorDict({"realized_demand": demand.view(*batch_size, self.T*self.K),
                           "observed_demand": observed_demand.view(*batch_size, self.T*self.K),
                           "expected_demand": e_x.view(*batch_size, self.T*self.K),
                           "std_demand":std_x.view(*batch_size, self.T*self.K),
                           "init_expected_demand": e_x_init_demand.view(*batch_size, self.T*self.K),
                           "batch_updates":batch_updates + 1,}, batch_size=batch_size, device=self.device,)

    ## Initial demand
    def _initial_contract_demand(self, batch_size,) -> Tuple[th.Tensor,th.Tensor]:
        """Get initial E[X], V[X] for spot and longterm contracts; cv: <0.1 is low, 0.3<x<0.5 is moderate, >0.5 is high
        :param utilization_rate: Average utilization of vessel capacity; 2*0.9 to get 90% utilization in uniform distribution
        :param spot_percentage: Percentage of spot demand; longterm demand is 1-spot_percentage
        :param cv_spot: Coefficient of variation for spot demand
        :param cv_longterm: Coefficient of variation for longterm demand
        :return: E[X], V[X]

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
        bound = th.ger(utilization_bound, self.spot_lc_percentage) / self.teus.view(1, -1,)
        # Get expected demand and variance
        expected_demand, variance = self._generate_initial_moments(batch_size, bound, self.cv)
        return expected_demand, variance

    def _generate_initial_moments(self, batch_size, bound, cv, eps=1e-2) -> Tuple[th.Tensor, th.Tensor]:
        """Generate initial E[X] and V[X] for spot and longterm contracts.
        - E[X] = Uniform sample * bound
        - V[X] = (E[X] * cv)^2"""
        # Sample uniformly from 0 to bound (inclusive) using torch.rand
        expected = th.rand(*batch_size, self.T, self.K, dtype=bound.dtype, device=self.device,) * bound.unsqueeze(0)
        variance = (expected * cv.view(1, 1, self.K,)) ** 2
        return th.where(expected < eps, eps, expected), variance

    # Support functions
    def _create_wave(self, length, param=0.3, ) -> th.Tensor:
        """Create a wave function for the bound"""
        mid_index = length // 2
        increasing_values = 1 + param * th.cos(th.linspace(th.pi, th.pi / 2, steps=mid_index + 1,
                                                           device=self.device, dtype=th.float32, ))
        decreasing_values = 1 + param * th.cos(th.linspace(th.pi / 2, th.pi, steps=length - mid_index,
                                                           device=self.device, dtype=th.float32, ))
        return th.cat([increasing_values[:-1], decreasing_values, th.zeros((1,), device=self.device, dtype=th.float32)])

    def _get_num_loads_in_voyage(self, transport_idx, P, ):
        """Get number of transports loaded per POL"""
        # Create a boolean mask for load pairs using broadcasting and advanced indexing
        load_mask = th.zeros((P, P), dtype=th.bool, device=self.device, )
        load_mask[transport_idx[:, 0], transport_idx[:, 1]] = True
        # Count loads for each port
        return load_mask.sum(dim=1)

    def _get_num_AC_in_voyage(self, transport_idx, POL, ):
        """Get number of transport in arrival condition per POL"""
        mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

    def _get_num_OB_in_voyage(self, transport_idx, POL, ):
        """Get number of transports in onbord per POL"""
        mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

def plot_demand_history(demand_history, updates,
                        y_label="Containers", title="Container Demand History",
                        summarize=False):
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

# Simulate demand
if __name__ == "__main__":
    # Parameters
    batch_size = [1024]
    ports = 4
    bays = 10
    decks = 2
    cargo_classes = 6
    customer_classes = 2
    weight_classes = 3
    capacity = [50]
    seed = 42
    iid_demand = True
    cv_demand = 0.5

    # Create generator
    generator = MPP_Generator(ports=ports, bays=bays, decks=decks, cargo_classes=cargo_classes,
                                        weight_classes=weight_classes,customer_classes=customer_classes,
                                       capacity=capacity, seed=seed, iid_demand=iid_demand, cv_demand=cv_demand,)
    td = generator(batch_size)

    # Test demand generation
    demand_history = []
    updates = 4000
    e_x_init_demand, _ = generator._initial_contract_demand(batch_size)

    for i in range(updates):
        if not iid_demand:
            e_x, std_x, dist = generator._gbm_lognormal_distribution(th.tensor([i], device=generator.device),
                                                                     e_x_init_demand,)
        else:
            std_x = generator._create_std_x(e_x_init_demand, cv_demand)
            e_x, std_x, dist = generator._iid_normal_distribution(e_x_init_demand, std_x,)

        demand = th.clamp(dist.sample(), min=0)
        demand_history.append(demand)
    demand_history = th.stack(demand_history)
    teus = generator.teus.view(1, -1, 1)

    # Plot demand history
    print("#"*50)
    print("Analyze demand history")
    print("*"*50)
    print("Time 0:")
    print("*"*50)
    print("Init E[x] (#):", e_x_init_demand.mean(dim=0))
    print("E[x]_t=0 (#):", demand_history[0].mean(dim=0))
    print("Std[x]_t=0 (#):", demand_history[0].std(dim=0))
    print("CV[x]_t=0 (#):", demand_history[0].std(dim=0) / demand_history[0].mean(dim=0))
    print("-"*50)
    print("Init E[Sum(x)] (#):", e_x_init_demand.sum(dim=(-1, -2)).mean(dim=0))
    print("E[Sum(x)]_t=0 (#):", demand_history[0].sum(dim=(-1, -2)).mean(dim=0))
    print("Std[Sum(x)]_t=0 (#):", demand_history[0].sum(dim=(-1, -2)).std(dim=0))
    print("-"*50)
    select_20 = (teus == 1).view(1, -1, 1)
    print("Init 20 ft (#):", (e_x_init_demand * select_20).sum(dim=(-1, -2)).mean(dim=0))
    print("E[20 ft]_t=0 (#):", (demand_history[0] * select_20).sum(dim=(-1, -2)).mean(dim=0))
    print("Std[20 ft]_t=0 (#):", (demand_history[0] * select_20).sum(dim=(-1, -2)).std(dim=0))
    print("-"*50)
    select_40 = (teus == 2).view(1, -1, 1)
    print("Init 40 ft (#):", (e_x_init_demand * select_40).sum(dim=(-1, -2)).mean(dim=0))
    print("E[40 ft]_t=0 (#):", (demand_history[0] * select_40).sum(dim=(-1, -2)).mean(dim=0))
    print("Std[40 ft]_t=0 (#):", (demand_history[0] * select_40).sum(dim=(-1, -2)).std(dim=0))
    print("-"*50)
    print("Init E[Sum(x)] (TEU):", (e_x_init_demand * teus).sum(dim=(-1, -2)).mean(dim=0))
    print("E[Sum(x)]_t=0 (TEU):", (demand_history[0] * teus).sum(dim=(-1, -2)).mean(dim=0))
    print("Std[Sum(x)]_t=0 (TEU):", (demand_history[0] * teus).sum(dim=(-1, -2)).std(dim=0))
    print("*" * 50)
    print("Time -1:")
    print("*" * 50)
    print("E[x]_t=-1 (#):", demand_history[-1].mean(dim=0))
    print("Std[x]_t=-1 (#):", demand_history[-1].std(dim=0))
    print("CV[x]_t=-1 (#):", demand_history[-1].std(dim=0) / demand_history[-1].mean(dim=0))
    print("-"*50)
    print("E[Sum(x)]_t=-1) (#):", demand_history[-1].sum(dim=(-1, -2)).mean(dim=0))
    print("Std[Sum(x)]_t=-1 (#):", demand_history[-1].sum(dim=(-1, -2)).std(dim=0))
    print("-"*50)
    print("E[Sum(x)]_t=-1 (TEU):", (demand_history[-1] * teus).sum(dim=(-1, -2)).mean(dim=0))
    print("Std[Sum(x)]_t=-1 (TEU):", (demand_history[-1] * teus).sum(dim=(-1, -2)).std(dim=0))

    # Plot containers
    plot_demand_history(demand_history, updates,
                        y_label="Containers", title="Container Demand History", summarize=True)
    teu_demand_history = demand_history * teus.unsqueeze(0)
    plot_demand_history(teu_demand_history, updates,
                        y_label="TEU", title="TEU Demand History", summarize=True)