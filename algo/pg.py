import numpy as np
import tensorflow as tf

from utils import tf_utils
from utils.sampler import rollouts
from gym.spaces.utils import flatdim


class PG:
    """ Policy Gradient Algorithm """

    def __init__(
            self,
            env,
            policy,
            n_trajectories=100,
            max_path_length=100,
            n_itr=100,
            lr=1E-3,
            discount=0.99,
            eval_deterministic=True,
            eval_n_episodes=4,
            eval_render=True
    ):
        """
        Args:
            env (`Gym.Env`): rllab environment object.
            policy (`NNPolicy`): A policy function approximator.
            n_trajectories (`int`) how many trajectories we use for
                a single iteration.
            max_path_length (`int`): Number of timesteps before resetting
                environment and policy.
            n_itr (`int`): number of iterations
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`bool`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`bool`): Whether or not to render the evaluation
                environment.
        """

        self._max_path_length = max_path_length
        self._n_trajectories = n_trajectories
        self._n_itr = n_itr
        self._lr = lr
        self._discount = discount

        self._eval_deterministic = eval_deterministic
        self._eval_n_episodes = eval_n_episodes
        self._eval_render = eval_render

        self._env = env
        self._policy = policy

        self._Da = flatdim(self._env.action_space)
        self._Do = flatdim(self._env.observation_space)

        self._training_ops = list()

        self._sess = tf_utils.get_default_session()

        self._init_placeholders()
        self._init_actor_update()

        self._sess.run(tf.global_variables_initializer())

    def train(self):
        """Run training of the VPG instance."""

        self._train(self._env, self._policy)

    def _init_placeholders(self):
        """Create input placeholders for the VPG algorithm.
        Creates `tf.placeholder`s for:
            - observation
            - action
            - reward_to_go
        """

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._reward_to_go_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='reward_to_go',
        )

    def _init_actor_update(self):
        """ Compute gradient for surrogate loss """

        logits = self._policy.get_distribution_for(
            self._obs_pl, reuse=True)
        negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=self._action_pl
        )  # N
        surrogate_loss = tf.reduce_mean(negative_likelihoods * self._reward_to_go_pl)

        policy_train_op = tf.train.AdamOptimizer(self._lr).minimize(
            loss=surrogate_loss,
            var_list=self._policy.get_params_internal()
        )

        self._training_ops.append(policy_train_op)

    def _train(self, env, policy):
        self._init_training(env, policy)

        with self._sess.as_default():
            for i in range(self._n_itr):
                # TODO: You need to implement policy gradient algorithm here,
                #       using building blocks elsewhere I defined for you.
                #       You should implement HERE in the for loop.
                """ In each iteration, you should:
                    1. rollout trajectories; (1 line of code)
                    2. do policy gradient with sampled trajectories; (1 line of code)
                    3. evaluate current policy. (1 line of code)
                """

            env.close()

        print('Training finished..')

    def _evaluate(self, itr):
        """Perform evaluation for the current policy.

        :param itr: The iteration number.
        :return: None
        """
        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(
                self._env, self._policy, self._max_path_length,
                self._discount, self._eval_n_episodes, render=self._eval_render,
            )

        average_return = np.mean([path['rewards'].sum() for path in paths])
        print('Average Return @ itr %d :' % itr, average_return)

    def _do_training(self, paths):
        """Runs the operations for training"""

        feed_dict = self._get_feed_dict(paths)
        self._sess.run(self._training_ops, feed_dict)

    def _get_feed_dict(self, paths):
        """Construct TensorFlow feed_dict from collected paths."""
        observations = [path["observations"] for path in paths]
        observations = np.reshape(observations, [-1, self._Do])

        actions = [path["actions"] for path in paths]
        actions = np.reshape(actions, [-1])
        actions = [int(a) for a in actions]
        actions = np.eye(self._Da)[actions]

        advantages = [path["advantages"] for path in paths]
        advantages = np.reshape(advantages, [-1])

        feed_dict = {
            self._obs_pl: observations,
            self._action_pl: actions,
            self._reward_to_go_pl: advantages
        }

        return feed_dict

    def _init_training(self, env, policy):
        """Method to be called at the start of training.
        :param env: Environment instance.
        :param policy:  Policy instance.
        :return: None
        """
        self._env = env
        self._policy = policy

        print('Training started ...', env)
