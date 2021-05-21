import numpy as np
import time

from gym.spaces.utils import flatdim


def rollout(env, policy, path_length, discount, render=False, speedup=5):

    Do = flatdim(env.observation_space)

    observations = np.zeros((path_length, Do))
    actions = np.zeros(path_length,)
    terminals = np.zeros((path_length,))
    rewards = np.zeros((path_length,))

    observation = env.reset()
    t = 0  # To make edge case path_length=0 work.

    for t in range(t, path_length):

        action, _ = policy.get_action(observation)

        next_obs, reward, terminal, _ = env.step(action)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t, :] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    # We need to compute the empirical return for each time step along the trajectory
    returns = []
    return_to_go = 0
    for t in range(len(rewards) - 1, -1, -1):
        return_to_go = rewards[t] + discount * return_to_go
        returns.append(return_to_go)
    # The returns are stored backwards in time, so we need to revert it
    returns = returns[::-1]

    path = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(terminals),
        rewards_to_go=np.array(returns)
    )

    return path


def rollouts(env, policy, path_length, discount, n_paths, render=False):
    paths = [
        rollout(env, policy, path_length, discount, render)
        for _ in range(n_paths)
    ]

    n_rewards_to_go = [p["rewards_to_go"] for p in paths]
    baselines = np.mean(n_rewards_to_go, axis=0)
    advantages = n_rewards_to_go - baselines

    for i, path in enumerate(paths):
        path.update(dict(advantages=np.array(advantages[i, :])))

    return paths
