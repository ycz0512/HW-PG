import tensorflow as tf


class NNPolicy:
    def __init__(self, env, obs_pl, action, scope_name=None):
        self._env = env
        self._obs_pl = obs_pl
        self._action = action
        self._scope_name = (
            tf.get_variable_scope().name if not scope_name else scope_name
        )

    def get_action(self, observation):
        return self.get_actions(observation[None])[0][0], None

    def get_actions(self, observations):
        feeds = {self._obs_pl: observations}
        actions = tf.get_default_session().run(self._action, feeds)
        return actions

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def env_spec(self):
        return self._env.spec

    def get_params_internal(self):
        scope = self._scope_name
        # Add "/" to 'scope' unless it's empty (otherwise get_collection will
        # return all parameters that start with 'scope'.
        scope = scope if scope == '' else scope + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
