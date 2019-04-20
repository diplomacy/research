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
""" Base Policy model
    - Contains the base policy model, which is used by all policy models.
"""
from collections import namedtuple
import logging
import re
from diplomacy_research.models.base_model import BaseModel, load_args as load_common_args
from diplomacy_research.settings import NO_PRESS_ALL_DATASET

# Constants
VALID_TAG = re.compile('^tag/policy/[a-z_]+/v[0-9]{3}_[a-z0-9_]+$')
LOGGER = logging.getLogger(__name__)

# Separate variable, so PolicyAdapter can load it
POLICY_BEAM_WIDTH = 10
GREEDY_DECODER = 0
TRAINING_DECODER = 1
SAMPLE_DECODER = 2
BEAM_SEARCH_DECODER = 3

class OrderProbTokenLogProbs(namedtuple('OrderProbTokenLogProbs', ('order',         # The order (e.g. A PAR - MAR)
                                                                   'probability',   # The (cond.) prob of the order
                                                                   'log_probs'))):  # The log probs of each token selec.
    """ A named tuple containing the order, its conditional prob., and the log prob of each selected token """

class StatsKey(namedtuple('StatsKey', ('prefix', 'power_name', 'order_type', 'season', 'phase', 'position'))):
    """ A named tuple representing a set of characteristics to compute accuracy """


def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return load_common_args() + [
        ('str', 'dataset', NO_PRESS_ALL_DATASET, 'The dataset builder to use for supervised learning'),
        ('float', 'learning_rate', 1e-3, 'Initial learning rate.'),
        ('float', 'lr_decay_factor', 0.93, 'Learning rate decay factor.'),
        ('float', 'max_gradient_norm', 5.0, 'Maximum gradient norm.'),
        ('int', 'beam_width', POLICY_BEAM_WIDTH, 'The number of beams to use for beam search'),
        ('int', 'beam_groups', POLICY_BEAM_WIDTH // 2, 'The number of groups for beam search'),
        ('float', 'dropout_rate', 0.5, 'Dropout rate to apply to all items in a training batch (Supervised learning).'),
        ('bool', 'use_v_dropout', True, 'Use variational dropout (same mask across all time steps)'),
        ('float', 'perc_epoch_for_training', 1.00, 'If < 1., runs evaluation every time x% of training epoch is done'),
        ('int', 'early_stopping_stop_after', 5, 'Stops after this many epochs if none of the tags have improved'),
        ('float', 'policy_coeff', 1.0, 'The coefficient to apply to the policy loss')
    ]

class BasePolicyModel(BaseModel):
    """ Base Policy Model"""

    def __init__(self, dataset, hparams):
        """ Initialization
            :param dataset: The dataset that is used to iterate over the data.
            :param hparams: A dictionary of hyper parameters with their values
            :type dataset: diplomacy_research.models.datasets.supervised_dataset.SupervisedDataset
            :type dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
        """
        BaseModel.__init__(self,
                           parent_model=None,
                           dataset=dataset,
                           hparams=hparams)

    def _encode_board(self, board_state, name, reuse=None):
        """ Encodes a board state or prev orders state
            :param board_state: The board state / prev orders state to encode - (batch, NB_NODES, initial_features)
            :param name: The name to use for the encoding
            :param reuse: Whether to reuse or not the weights from another encoding operation
            :return: The encoded board state / prev_orders state
        """
        raise NotImplementedError()

    def _get_board_state_conv(self, board_0yr_conv, is_training, prev_ord_conv=None):
        """ Computes the board state conv to use as the attention target (memory)

            :param board_0yr_conv: The board state encoding of the current (present) board state)
            :param is_training: Indicate whether we are doing training or inference
            :param prev_ord_conv: Optional. The encoding of the previous orders state.
            :return: The board state conv to use as the attention target (memory)
        """
        raise NotImplementedError()

    def _build_policy_initial(self):
        """ Builds the policy model (initial step) """
        raise NotImplementedError()

    def _build_policy_final(self):
        """ Builds the policy model (final step) """

    def _validate(self):
        """ Validates the built model """
        # Making sure all the required outputs are present
        assert 'board_state' in self.features
        assert 'current_power' in self.features
        assert 'targets' in self.outputs
        assert 'selected_tokens' in self.outputs
        assert 'argmax_tokens' in self.outputs
        assert 'logits' in self.outputs
        assert 'log_probs' in self.outputs
        assert 'beam_tokens' in self.outputs
        assert 'beam_log_probs' in self.outputs
        assert 'board_state_conv' in self.outputs
        assert 'board_state_0yr_conv' in self.outputs
        assert 'rnn_states' in self.outputs
        assert 'policy_loss' in self.outputs
        assert 'draw_prob' in self.outputs
        assert 'learning_rate' in self.outputs
        assert 'in_retreat_phase' in self.outputs

        # Making sure we have a name tag
        for tag in self.outputs:
            if VALID_TAG.match(tag):
                class_name = str(type(self))         # diplomacy_research.models.policy.token_based.vxxx_002.yyy
                expected_tag = '/'.join(['tag'] + class_name.split('.')[2:5])
                assert tag == expected_tag, 'Expected tag to be "%s".' % expected_tag
                break
        else:
            raise RuntimeError('Unable to find a name tag. Format: "tag/policy/xxxxx/v000_xxxxxx".')
