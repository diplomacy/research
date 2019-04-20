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
""" Draw model (v001_draw_relu)
    - Contains the draw model
"""
import logging
from diplomacy_research.models.state_space import get_adjacency_matrix, NB_NODES, NB_FEATURES, NB_POWERS
from diplomacy_research.models.draw.base_draw_model import BaseDrawModel, load_args as load_parent_args

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_parent_args() + [
        # Hyperparameters
        ('int', 'draw_gcn_1_output_size', 25, 'Graph Conv 1 output size.'),
        ('int', 'draw_embedding_size', 128, 'Embedding size.'),
        ('int', 'draw_h1_size', 64, 'The size of the first hidden layer in the draw calculation'),
        ('int', 'draw_h2_size', 64, 'The size of the second hidden layer in the draw calculation')
    ]

class DrawModel(BaseDrawModel):
    """ Draw Model - Evaluates whether a given power should accept a draw or not """

    def _build_draw_initial(self):
        """ Builds the draw model (initial step) """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.graph_convolution import GraphConvolution, preprocess_adjacency
        from diplomacy_research.utils.tensorflow import to_float

        if not self.placeholders:
            self.placeholders = self.get_placeholders()
        else:
            self.placeholders.update(self.get_placeholders())

        # Quick function to retrieve hparams and placeholders and function shorthands
        hps = lambda hparam_name: self.hparams[hparam_name]
        pholder = lambda placeholder_name: self.placeholders[placeholder_name]
        relu = tf.nn.relu
        sigmoid = tf.nn.sigmoid

        # Training loop
        with tf.variable_scope('draw', reuse=tf.AUTO_REUSE):
            with tf.device(self.cluster_config.worker_device if self.cluster_config else None):

                # Features
                board_state = to_float(self.features['board_state'])        # tf.float32 - (b, NB_NODES, NB_FEATURES)
                current_power = self.features['current_power']              # tf.int32   - (b,)
                draw_target = self.features['draw_target']                  # tf.float32 - (b,)

                # Placeholders
                stop_gradient_all = pholder('stop_gradient_all')

                # Norm Adjacency
                batch_size = tf.shape(board_state)[0]
                norm_adjacency = preprocess_adjacency(get_adjacency_matrix())
                norm_adjacency = tf.tile(tf.expand_dims(norm_adjacency, axis=0), [batch_size, 1, 1])

                # Graph embeddings
                with tf.variable_scope('graph_conv_scope'):
                    board_state_h0 = board_state                                                        # (b, 81, 35)
                    board_state_h1 = GraphConvolution(input_dim=NB_FEATURES,
                                                      output_dim=hps('draw_gcn_1_output_size'),
                                                      norm_adjacency=norm_adjacency,
                                                      activation_fn=relu,
                                                      bias=True)(board_state_h0)                        # (b, 81, 25)

                    # board_state_h2: (b, 2025)
                    # board_state_h3: (b, 128)
                    board_state_h2 = tf.reshape(board_state_h1, shape=[-1, NB_NODES * hps('draw_gcn_1_output_size')])
                    board_state_graph_conv = tf.layers.Dense(units=hps('draw_embedding_size'),
                                                             activation=relu,
                                                             use_bias=True)(board_state_h2)

                # Calculating draw for all powers
                with tf.variable_scope('draw_scope'):
                    current_power_mask = tf.one_hot(current_power, NB_POWERS, dtype=tf.float32)

                    draw_h0 = board_state_graph_conv                                                    # (b, 128)
                    draw_h1 = tf.layers.Dense(units=hps('draw_h1_size'),                                # (b, 64)
                                              activation=relu,
                                              use_bias=True)(draw_h0)
                    draw_h2 = tf.layers.Dense(units=hps('draw_h2_size'),                                # (b, 64)
                                              activation=relu,
                                              use_bias=True)(draw_h1)
                    draw_probs = tf.layers.Dense(units=NB_POWERS,                                       # (b, 7)
                                                 activation=sigmoid,
                                                 use_bias=True)(draw_h2)
                    draw_prob = tf.reduce_sum(draw_probs * current_power_mask, axis=1)                  # (b,)

                # Computing draw loss
                with tf.variable_scope('draw_loss'):
                    draw_loss = tf.reduce_mean(tf.square(draw_target - draw_prob))
                    draw_loss = tf.cond(stop_gradient_all,
                                        lambda: tf.stop_gradient(draw_loss),                                            # pylint: disable=cell-var-from-loop
                                        lambda: draw_loss)                                                              # pylint: disable=cell-var-from-loop

        # Building output tags
        outputs = {'tag/draw/v001_draw_relu': True,
                   'draw_prob': draw_prob,
                   'draw_loss': draw_loss}

        # Adding features, placeholders and outputs to graph
        self.add_meta_information(outputs)
