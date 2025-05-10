"""
Module for training road agent without interdependency
"""

# Training the road agent via Q learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from road_env import RoadEnv
from road_agent_Q_learning import RoadAgentQLearning

# 1. Load data
# ---------- Load SiouxFalls Network ----------
# Data from https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
links_df = pd.read_csv(
    "SiouxFalls_net.tntp",
    skiprows=8,  # skip metadata
    sep="\t"
)
links_df.columns = [c.strip().lower() for c in links_df.columns]
links_df.drop(['~', ';'], axis=1, inplace=True)
links_df = links_df.rename(columns={'init_node': 'from', 'term_node': 'to', 'free_flow_time': 't0'})

# ---------- Load SiouxFalls OD ----------
trips_file = 'SiouxFalls_trips.tntp'
lines = open(trips_file).read().splitlines()
start = 0
for idx, line in enumerate(lines):
    if '<END OF METADATA>' in line:
        start = idx + 1
        break
zones = 24  # number of nodes
od_matrix = np.zeros((zones, zones))
current_origin = None
for line in lines[start:]:
    line = line.strip()
    if line.startswith('Origin'):
        parts = line.split()
        current_origin = int(parts[1])
    elif ':' in line:
        entries = line.rstrip(';').split(';')
        for ent in entries:
            ent = ent.strip()
            if not ent: continue
            dest, flow = ent.split(':')
            od_matrix[current_origin - 1, int(dest) - 1] = float(flow)

# 2. Set parameters
# parameters
avg_hours = 24  # average repair time (hours) per unit length (mile, assuming free flow speed 60mph)
num_dis_links = 3  # IMPORTANT: how many links should be randomly disrupted
sigma_bar = avg_hours * np.array(links_df.loc[:, 'length'])
C = 2
R_max = 6
max_time = 30 * 24  # hours

# 3. Instantiate environment and agent
env = RoadEnv(
    links_df=links_df,
    od_matrix=od_matrix,
    sigma_bar=sigma_bar,
    C=C,
    R_max=R_max,
    max_time=max_time,
    num_dis_links=num_dis_links
)

agent = RoadAgentQLearning(
    env=env,
    learning_rate=0.25,
    initial_epsilon=1.0,
    epsilon_decay=0.995,
    final_epsilon=0.05,
    discount_factor=0.99,
)

# 4. Training loop

# Before training
env.return_queue = []
env.length_queue = []
agent.training_error = []
num_episodes = 3000

for ep in range(num_episodes):
    state, _ = env.reset()
    s_key = tuple(state.tolist())

    done = False
    total_reward = 0.0
    length = 0

    while not done:
        action = agent.get_action(s_key)
        next_state, reward, terminated, truncated, _ = env.step(action)
        ns_key = tuple(next_state.tolist())

        agent.update(s_key, action, reward, ns_key, terminated)
        # record error for plotting
        agent.training_error.append(abs(
            (reward + (0 if terminated else agent.gamma * max([0.0] if list(agent.q_values[ns_key].values()) == []
                                                              else agent.q_values[ns_key].values())))
            - agent.q_values[s_key][tuple(action)]
        ))

        s_key = ns_key
        # state = next_state
        total_reward += reward
        length += 1
        done = terminated or truncated

    agent.decay_epsilon()

    # At episode end, record
    env.return_queue.append(total_reward)
    env.length_queue.append(length)

    # print('episode {} done'.format(ep))
    if ep % 100 == 0:
        print(f"Episode {ep}/{num_episodes} — Reward: {total_reward:.1f}, Length: {length}, ε={agent.epsilon:.3f}")


# 5. Plot
def get_moving_avgs(arr, window, mode="valid"):
    """
    Compute moving average of `arr` with window size `window`.
    Uses numpy.convolve under the hood. :contentReference[oaicite:1]{index=1}
    """
    return np.convolve(
        np.array(arr, dtype=float).flatten(),
        np.ones(window, dtype=float),
        mode=mode
    ) / window


# 3‐panel figure
rolling_length = 500  # e.g., smooth over 500 episodes

fig, axs = plt.subplots(ncols=3, figsize=(15, 4))

# 1) Episode rewards
axs[0].set_title("Episode Rewards (Smoothed)")
reward_ma = get_moving_avgs(env.return_queue, rolling_length, mode="valid")
axs[0].plot(reward_ma)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Average Reward")

# 2) Episode lengths
axs[1].set_title("Episode Lengths (Smoothed)")
length_ma = get_moving_avgs(env.length_queue, rolling_length, mode="valid")
axs[1].plot(length_ma)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length (steps)")

# 3) Training Error (TD‐error)
axs[2].set_title("Training Error (Smoothed)")
error_ma = get_moving_avgs(agent.training_error, rolling_length, mode="same")
axs[2].plot(error_ma)
axs[2].set_xlabel("Update Step")
axs[2].set_ylabel("|TD Error|")

plt.tight_layout()
plt.show()