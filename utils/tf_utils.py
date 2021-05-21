import tensorflow as tf


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    return tf.InteractiveSession(**kwargs)
