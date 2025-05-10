"""
Module for road environment
"""

import pulp
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from compute_road_reward import compute_reward, compute_ue_travel_time

# ---------- Resource-Allocation MILP (Big-M Linearization) ----------
def solve_team_allocation(A, sigma_bar, C, R, M=1e6):
    """
    Solve min theta s.t. team-allocation constraints (Model B11–B18).
    A : list[int]        # selected link indices
    sigma_bar : list     # average repair times per link
    C : int             # max teams per link assume the same for all links
    R : int              # total available teams
    Returns theta^* or None if infeasible.
    """
    prob = pulp.LpProblem("alloc", pulp.LpMinimize)
    theta = pulp.LpVariable("theta", lowBound=0)
    v = {(l, phi): pulp.LpVariable(f"v_{l}_{phi}", cat="Binary")
         for l in A for phi in range(1, C + 1)}
    vpr = {(l, phi): pulp.LpVariable(f"vpr_{l}_{phi}", lowBound=0)
           for l in A for phi in range(1, C + 1)}
    prob += theta  # B11
    # B16 & B17
    for l in A:
        prob += sum(v[l, phi] for phi in range(1, C + 1)) == 1
    prob += sum(phi * v[l, phi] for l in A for phi in range(1, C + 1)) <= R
    # B12
    for l in A:
        prob += sum(phi * vpr[l, phi] for phi in range(1, C + 1)) >= sigma_bar[l]
    # B13–B15
    for l in A:
        for phi in range(1, C + 1):
            prob += vpr[l, phi] <= M * v[l, phi]
            prob += vpr[l, phi] <= theta + M * (1 - v[l, phi])
            prob += vpr[l, phi] >= theta - M * (1 - v[l, phi])
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] == 'Optimal':
        return pulp.value(theta)
    return None


# ---------- Gymnasium Environment ----------
class RoadEnv(gym.Env):
    """
    Gym environment for road RL-OP model

    Attributes
    ----------
    links_df : pd.DataFrame
        DataFrame with columns ['from','to','capacity','t0'].
    od_matrix : np.ndarray
        OD demand matrix shape (N_zones, N_zones).
    sigma_bar : dict
        Average repair times per link.
    C : int
        Max teams per link assume the same for all links.
    R_max : int
        Total available teams.
    max_time : float
        Total allowed horizon (hours).
    num_dis_links : int
        Number of random links to be disrupted at the initial state.
    link_status : np.ndarray
        1 if link is currently disrupted, 0 if link is not.
    current_time : float
        Elapsed real env time (hours) so far.
    num_links : int
        Number of links in the network.
    """

    def __init__(self, links_df, od_matrix, sigma_bar, C, R_max, max_time, num_dis_links):
        super().__init__()
        self.links_df, self.od_matrix = links_df, od_matrix
        self.sigma_bar, self.C, self.R_max = sigma_bar, C, R_max
        self.max_time = max_time
        self.num_dis_links = num_dis_links
        self.link_status = np.zeros(len(self.links_df), dtype=int)
        self.current_time = 0.0
        self.num_links = len(self.links_df)

        # actions are all feasible subsets of links that can be selected for restoration in that state
        # 1 if the link is selected for repair, 0 o.w.
        self.action_space = spaces.MultiBinary(self.num_links)

        # observations are the conditions (good/bad) of links
        # 1 if link is currently disrupted, 0 if link is not
        self.observation_space = spaces.MultiBinary(self.num_links)

        # self.reset()

    def _get_obs(self):
        return self.link_status

    def _get_info(self):
        return {'current_real_time': self.current_time}

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Randomly disrupts `num_dis_links` links, resets time to zero, and returns
        an observation of link statuses and an info dict with current_time.
        """
        # 1. Initialize RNG and state via Gymnasium API
        super().reset(seed=seed)  # ensures np_random is seeded properly 

        # 2. Draw `num_dis_links` unique indices from 0..num_links-1 by numpy.random.Generator.choice
        disrupted = self.np_random.choice(self.num_links, size=self.num_dis_links, replace=False)

        # 3. Reset link_status and mark sampled indices as disrupted
        self.link_status[:] = 0
        self.link_status[disrupted] = 1  # mark selected links

        # 4. Reset simulation clock
        self.current_time = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: tuple):
        """
        Execute one time‐step within the environment.

        Parameters
        ----------
        action : tuple
            A binary vector of length self.num_links indicating which links to attempt restoring.

        Returns
        -------
        observation : np.ndarray
            The updated link_status array.
        reward : float
            The computed reward for this action.
        terminated : bool
            True if the episode has ended (time horizon reached or all links restored).
        truncated : bool
            False (we do not use truncation here).
        info : dict
            Auxiliary information, e.g., current_time.
        """
        # 1. Decode which links to repair this step
        action_links = [i for i, a in enumerate(action) if a == 1]

        # 2. Feasibility: solve resource allocation to get θ*
        theta_opt = solve_team_allocation(
            A=action_links,
            sigma_bar=self.sigma_bar,
            C=self.C,
            R=self.R_max
        )
        if theta_opt is None:
            # Infeasible allocation → heavy penalty, end episode
            raise ValueError("Infeasible action!")
            # return self._get_obs(), -1e3, False, False, {"current_time": self.current_time}
            # return self._get_obs(), -1e3, True, False, {"current_time": self.current_time}

        # 3. Compute reward
        reward, new_time = compute_reward(
            current_time=self.current_time,
            max_time=self.max_time,
            state_links=self.link_status,
            action_links=action_links,
            links_df=self.links_df,
            od_matrix=self.od_matrix,
            compute_ue_travel_time=compute_ue_travel_time
        )

        # 4. Record the real time for the next stage/timestep
        self.current_time = new_time

        # 5. Update link_status: mark all action_links as repaired (0)
        for l in action_links:
            self.link_status[l] = 0

        # 6. Check termination: time horizon or full restoration
        # Terminal: all links restored
        terminated = (self.link_status.sum() == 0)
        # Truncation: time horizon reached
        truncated = (self.current_time >= self.max_time)

        # 7. Return according to Gymnasium API
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Real Time: {self.current_time:.1f}h, "
              f"Remaining disrupted links: {int(self.link_status.sum())}")

    def feasible_actions(self, state):
        """
        Given a binary state vector of length L (1=disrupted),
        return all feasible actions (as 0/1 np.ndarray of length L) that:
        - choose a subset A of disrupted links with |A| <= R_max
        - solve_team_allocation(A) is feasible (returns non-None)

        Parameters
        ----------
        state : np.ndarray
            Binary array of length L indicating disrupted links.

        Returns
        -------
        List[np.ndarray]
            Each entry is a binary action vector of length L.
        """
        disrupted = [i for i, d in enumerate(state) if d == 1]
        feas_actions = []

        # Only consider subset sizes up to R_max
        max_k = min(len(disrupted), self.R_max)
        for k in range(1, max_k + 1):
            # Generate all combinations of size k
            for subset in itertools.combinations(disrupted, k):
                # Check resource-allocation feasibility
                theta_opt = solve_team_allocation(
                    A=list(subset),
                    sigma_bar=self.sigma_bar,
                    C=self.C,
                    R=self.R_max
                )
                if theta_opt is not None:
                    # Build a binary action vector
                    action = np.zeros(self.num_links, dtype=int)
                    action[list(subset)] = 1
                    feas_actions.append(action)
        return feas_actions