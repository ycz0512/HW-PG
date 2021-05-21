""" Discrete action policy. """

import numpy as np
import tensorflow as tf

from utils.mlp import mlp
from policy.nn_policy import NNPolicy
from contextlib import contextmanager
from gym.spaces.utils import flatdim


class MLP_Policy(NNPolicy):
    """Multi-layer Perceptron Policy"""
    def __init__(self, env, hidden_layer_sizes=(100, 100)):
        """
        Args:
            env (`Env`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multi-layer
                Perceptron hidden layers.
        """

        self._Da = flatdim(env.action_space)
        self._Ds = flatdim(env.observation_space)

        self._layer_sizes = list(hidden_layer_sizes) + [self._Da]

        self._is_deterministic = False

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Ds],
            name='observation',
        )

        self._act_prob = tf.nn.softmax(self.get_distribution_for(self._obs_pl))    # N x Da
        self._act_ind = tf.multinomial(logits=tf.log(self._act_prob), num_samples=1)   # N x 1

        super(MLP_Policy, self).__init__(
            env,
            self._obs_pl,
            self._act_ind,
            'policy'
        )

    def get_distribution_for(self, obs_t, reuse=False):
        """compute the category distribution logits."""

        with tf.variable_scope('policy', reuse=reuse):
            logits = mlp(
                inputs=obs_t,
                layer_sizes=self._layer_sizes,
                output_nonlinearity=None
            )       # N x Da

        return logits

    def get_action(self, obs):
        """Sample action based on the observations.
        If `self._is_deterministic` is True, returns a greedily sampled action
        for the observations. If False, return stochastically sampled action.
        """

        if not self._is_deterministic:
            return NNPolicy.get_action(self, obs)

        # Handle the deterministic case separately.
        feeds = {self._obs_pl: obs[None]}
        act_prob = tf.get_default_session().run(self._act_prob, feeds)[0]  # Da

        return np.argmax(act_prob), {}  # Da

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.
        See `self.get_action` for further information about the effect of
        self._is_deterministic.
        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        current = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = current
