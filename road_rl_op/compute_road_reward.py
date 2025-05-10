# -*- coding: utf-8 -*-
"""
Module for road reward function
"""

import numpy as np
import pandas as pd
import heapq
import random

# ---------- Golden-Section Search ----------
def golden_section_search(f, a, b, tol=1e-5):
    """Find minimum lambda of unimodal objective f on [0, 1] via golden-section search."""
    invphi = (np.sqrt(5) - 1) / 2 # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2 # 1/phi^2
    h = b - a
    if h <= tol:
        return (a + b) / 2

    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc, yd = f(c), f(d)
    for _ in range(n):
        if yc < yd:
            b, d, yd = d, c, yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a, c, yc = c, d, yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    return (a + b) / 2

# ---------- All-or-Nothing Assignment ----------
def all_or_nothing(n_nodes, links, costs, od_matrix, disabled):
    """
        Compute auxiliary flows by assigning OD demand to the shortest paths.

        Args:
            n_nodes: number of nodes
            links: list of (u,v,capacity,t0)
            costs: current cost per link
            od_matrix: NxN numpy array of demands
            disabled: set of link indices to exclude

        Returns:
            flows: flow\demand value on each link
    """
    # Build adjacency: for each u, list of (v, cost, link_idx)
    adj = {i: [] for i in range(1, n_nodes+1)}
    for idx, (u, v, cap, t0) in enumerate(links):
        if idx in disabled:
            continue
        adj[u].append((v, costs[idx], idx))
    flows = np.zeros(len(links))
    # For each origin, run Dijkstra
    for i in range(1, n_nodes+1):
        # min-heap: (dist, node, prev_link)
        dist = {node: np.inf for node in adj}
        prev = {}
        dist[i] = 0
        heap = [(0, i)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            for v, w, idx in adj[u]:
                dv = du + w
                if dv < dist[v]:
                    dist[v] = dv
                    prev[v] = (u, idx)
                    heapq.heappush(heap, (dv, v))
        # Assign demands
        for j in range(1, n_nodes+1):
            d = od_matrix[i-1, j-1]
            if d <= 0 or (j not in prev and i != j):
                continue
            # backtrack path
            node = j
            while node != i:
                u, idx = prev[node]
                flows[idx] += d
                node = u
    return flows

# ---------- User Equilibrium via Frank–Wolfe ----------
def compute_ue_travel_time(n_nodes, links_df, od_matrix, disabled_links,
                           alpha=0.15, beta=4, max_iter=100, tol=1e-4):
    """
        Returns total travel time under UE.

        Args:
            n_nodes: number of nodes in the current network
            links_df: Pandas DataFrame for the network links with columns ['from','to','capacity','t0']
            od_matrix: OD demand with numpy array shape (n_nodes,n_nodes)
            disabled_links: list of link indices to disable (links disrupted or still in repair)

        Corner Case 1: some origin/destination nodes are unreachable, then the total travel time could be infinity

        Notice: link indices are consistent with the order of links_df
    """
    links = list(zip(
        links_df['from'].astype(int), # init_node
        links_df['to'].astype(int), # term_node
        links_df['capacity'].astype(float),
        links_df['t0'].astype(float) # free_flow_time
    ))
    K = links_df['capacity'].values
    t0 = links_df['t0'].values
    x = all_or_nothing(n_nodes, links, t0, od_matrix, set(disabled_links))
    for it in range(max_iter):
        costs = t0 * (1 + alpha * (x / K)**beta)
        y = all_or_nothing(n_nodes, links, costs, od_matrix, set(disabled_links))
        d = y - x
        def obj(lmbda):
            xt = x + lmbda * d
            return np.sum(t0 * xt * (1 + alpha * (xt / K)**beta))
        lam = golden_section_search(obj, 0, 1)
        x_new = x + lam * d
        gap = np.sum((y - x) * costs)
        if gap / np.sum(y * costs) < tol:
            x = x_new
            break
        x = x_new
    return np.sum(x * t0 * (1 + alpha * (x / K)**beta))

# ---------- Emergency Repair-Time Generator ----------
def generate_repair_times(link_lengths, avg_hours=24, variation=0.2):
    """
    Generate emergency repair times for each link.
    Parameters
    ----------
    link_lengths : array-like of float
        Proxy lengths (e.g., free-flow times) of the disrupted links.
    avg_hours : float
        Base average repair time (hours) per unit length (mile, assuming free flow speed 60mph).
    variation : float
        Fractional uniform variation around the mean.
    Returns
    -------
    times : np.ndarray
        Sampled repair times (hours).
    """
    mean_times = avg_hours * np.array(link_lengths)
    low = mean_times * (1 - variation)
    high = mean_times * (1 + variation)
    return np.random.uniform(low, high)

def compute_reward(
    current_time, max_time, state_links, action_links,
    links_df, od_matrix, compute_ue_travel_time
):
    """
    Compute the MDP reward for repairing `action_links` at `current_time`.

    Parameters
    ----------
    current_time : float
        Elapsed real env time (hours) so far.
    max_time : float
        Total allowed horizon (hours), e.g., 30*24 hrs.
    state_links : list of binary
        1 if link in that index is currently disrupted.
    action_links : list of int
        Indices of links chosen for repair.
    links_df : pd.DataFrame
        DataFrame with columns ['from','to','capacity','t0'].
    od_matrix : np.ndarray
        OD demand matrix shape (N_zones, N_zones).
    compute_ue_travel_time : func
        Function to compute total UE travel time.

    Returns
    -------
    reward : float
        (TT_unrepaired – TT_repaired) × (max_time – (current_time + theta)).
    new_time : float
        New real env time (hours) after repair.
    """
    # 1. Realize random repair times for selected links
    lengths = links_df.loc[action_links, 'length']  # if free-flow time = length, then unit is mile
    repair_times = generate_repair_times(lengths)
    theta = float(np.max(repair_times))

    # 2. Travel time with links still disrupted
    disrupted_before = [i for i, d in enumerate(state_links) if d == 1]
    tt_before = compute_ue_travel_time(
        n_nodes=links_df['from'].max(),
        links_df=links_df, od_matrix=od_matrix,
        disabled_links=disrupted_before
    )

    # 3. Travel time after repairs finish
    disrupted_after = [i for i in disrupted_before if i not in action_links]
    tt_after = compute_ue_travel_time(
        n_nodes=links_df['from'].max(),
        links_df=links_df, od_matrix=od_matrix,
        disabled_links=disrupted_after
    )

    # 4. Reward: time saved × remaining horizon
    time_remaining = max_time - (current_time + theta)
    return (tt_before - tt_after) * max(0, time_remaining), current_time + theta
