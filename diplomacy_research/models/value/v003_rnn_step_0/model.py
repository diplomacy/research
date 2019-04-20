# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
""" Value model (v003_rnn_step_0)
    - Contains the value model
"""
import logging
from diplomacy_research.models.state_space import NB_POWERS
from diplomacy_research.models.value.base_value_model import BaseValueModel, load_args as load_parent_args

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_parent_args() + [
        # Hyperparameters
        ('int', 'value_h1_size', 256, 'The size of the first hidden layer in the value calculation'),
    ]

class ValueModel(BaseValueModel):
    """ Value Model - Evaluates the value of a state for all powers """

    def _get_board_value(self, board_state, current_power, name='board_state_value', reuse=None):
        """ Computes the estimated value of a board state
            :param board_state: The board state - (batch, NB_NODES, NB_FEATURES)
            :param current_power: The power for which we want the board value - (batch,)
            :param name: The name to use for the operaton
            :param reuse: Whether to reuse or not the weights from another operation
            :return: The value of the board state for the specified power - (batch,)
        """
        from diplomacy_research.utils.tensorflow import tf
        LOGGER.warning('This model does not support computing the value of a board state. Returning 0.')
        return tf.zeros_like(current_power, dtype=tf.float32)

    def _build_value_initial(self):
        """ Builds the value model (initial step) """
        # Skipping - As we need the rnn_states from the policy to compute this value function

    def _build_value_final(self):
        """ Builds the value model (final step) """
        from diplomacy_research.utils.tensorflow import tf

        if not self.placeholders:
            self.placeholders = self.get_placeholders()
        else:
            self.placeholders.update(self.get_placeholders())

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]
        relu = tf.nn.relu

        # Training loop
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):

                # Outputs from the policy model
                assert 'rnn_states' in self.outputs

                # Inputs and Features
                rnn_states = self.outputs['rnn_states']
                current_power = self.features['current_power']                  # tf.int32   - (b,)
                value_target = self.features['value_target']                    # tf.float32 - (b,)

                # Placeholders
                stop_gradient_all = pholder('stop_gradient_all')

                # Computing the value
                value_h0 = tf.stop_gradient(rnn_states) if hps('stop_gradient_value') else rnn_states
                value_h0_pos_0 = value_h0[:, 0, :]                                              # (b, lstm_size)

                # Linear with relu
                # Then linear without relu
                value_h1_pos_0 = tf.layers.Dense(units=hps('value_h1_size'),                    # (b, 256)
                                                 use_bias=True,
                                                 activation=relu)(value_h0_pos_0)
                value_h2_pos_0 = tf.layers.Dense(units=NB_POWERS,                               # (b, 7)
                                                 use_bias=True,
                                                 activation=None)(value_h1_pos_0)

                # Computing for the current power
                current_power_mask = tf.one_hot(current_power, NB_POWERS, dtype=tf.float32)
                state_value = tf.reduce_sum(current_power_mask * value_h2_pos_0, axis=-1)       # (b,)

                # Computing value loss
                with tf.variable_scope('value_loss'):
                    value_loss = tf.reduce_mean(tf.square(value_target - state_value))
                    value_loss = tf.cond(stop_gradient_all,
                                         lambda: tf.stop_gradient(value_loss),                                          # pylint: disable=cell-var-from-loop
                                         lambda: value_loss)                                                            # pylint: disable=cell-var-from-loop

        # Building output tags
        outputs = {'tag/value/v003_rnn_step_0': True,
                   'state_value': state_value,
                   'value_loss': value_loss}

        # Adding features, placeholders and outputs to graph
        self.add_meta_information(outputs)
