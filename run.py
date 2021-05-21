import gym
import argparse

from algo.pg import PG
from policy.mlp_policy import MLP_Policy


SHARED_PARAMS = {
    "lr": 1E-3,
    "discount": 0.99,
    "hidden_layer_sizes": [64, 64],
    "n_trajectories": 256,
    "n_itr": 100,
}


ENV_PARAMS = {
    'CartPole': { # Da = 2
        'env_name': 'CartPole-v1',
        'max_path_length': 200,
    },
    'Acrobot': { # Da = 3
        'env_name': 'Acrobot-v1',
        'max_path_length': 200,
    },
    'MountainCar': { # Da = 3
        'env_name': 'MountainCar-v0',
        'max_path_length': 200,
    },
}

DEFAULT_ENV = 'CartPole'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV,
                        help='available environments')
    parser.add_argument('--n_itr',
                        type=int,
                        default=SHARED_PARAMS["n_itr"],
                        help='number of training iterations')
    parser.add_argument('--viz',
                        default=False,
                        action='store_true',
                        help='visualize current policy')
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params["n_itr"] = args.n_itr
    params["visualize"] = args.viz
    params.update(env_params)

    return params


def run_experiment(params):
    env = gym.make(params["env_name"])

    base_kwargs = dict(
        n_trajectories=params["n_trajectories"],
        max_path_length=params['max_path_length'],
        n_itr=params["n_itr"],
        lr=params["lr"],
        discount=params["discount"],
        eval_render=params["visualize"],
        eval_n_episodes=10,
        eval_deterministic=True,
    )

    policy = MLP_Policy(
        env=env,
        hidden_layer_sizes=params["hidden_layer_sizes"]
    )

    algorithm = PG(
        env=env,
        policy=policy,
        **base_kwargs
    )

    algorithm.train()


if __name__ == '__main__':
    args = parse_args()
    params = get_variants(args)
    run_experiment(params)
