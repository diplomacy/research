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
""" Base Policy Dataset Builder
    - Abstract class responsible for generating the protocol buffers to be used by the model
    - All policies builders will inherit from this class
"""
from abc import ABCMeta
import logging
from diplomacy_research.models.datasets.base_builder import BaseBuilder

# Constants
LOGGER = logging.getLogger(__name__)

class BasePolicyBuilder(BaseBuilder, metaclass=ABCMeta):
    """ This object is responsible for maintaining the data and feeding it into the tensorflow model """

    @staticmethod
    def get_proto_fields():
        """ Returns the proto fields used by this dataset builder """
        raise NotImplementedError()

    @staticmethod
    def get_feedable_item(locs, state_proto, power_name, phase_history_proto, possible_orders_proto, **kwargs):
        """ Computes and return a feedable item (to be fed into the feedable queue)
            :param locs: A list of locations for which we want orders
            :param state_proto: A `.proto.game.State` representation of the state of the game.
            :param power_name: The power name for which we want the orders and the state values
            :param phase_history_proto: A list of `.proto.game.PhaseHistory`. This represents prev phases.
            :param possible_orders_proto: A `proto.game.PossibleOrders` object representing possible order for each loc.
            :return: A feedable item, with feature names as key and numpy arrays as values
        """
        # pylint: disable=arguments-differ
        raise NotImplementedError()

    @property
    def proto_generation_callable(self):
        """ Returns a callable required for proto files generation.
            e.g. return generate_proto(saved_game_bytes, is_validation_set)

            Note: Callable args are - saved_game_bytes: A `.proto.game.SavedGame` object from the dataset
                                    - phase_ix: The index of the phase we want to process
                                    - is_validation_set: Boolean that indicates if we are generating the validation set

            Note: Used bytes_to_proto from diplomacy_research.utils.proto to convert bytes to proto
                  The callable must return a list of tf.train.Example to put in the protocol buffer file
        """
        raise NotImplementedError()
