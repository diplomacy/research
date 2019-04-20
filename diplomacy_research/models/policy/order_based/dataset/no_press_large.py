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
""" Order Based NoPress Builder (Large) dataset
    - Class responsible for generating the protocol buffers for NoPress (Large) dataset
"""
import logging
import os
from diplomacy import Map
from diplomacy_research.models.policy.order_based.dataset.base import BaseDatasetBuilder, get_policy_data
from diplomacy_research.models.state_space import get_top_victors, get_current_season, GO_ID, POWER_VOCABULARY_KEY_TO_IX
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import bytes_to_proto
from diplomacy_research.settings import WORKING_DIR, PROTOBUF_DATE

# Constants
LOGGER = logging.getLogger(__name__)
DESC = "proto-policy-order_based-no_press_large"

class DatasetBuilder(BaseDatasetBuilder):
    """ This object is responsible for maintaining the data and feeding it into the tensorflow model """

    # Paths as class properties
    training_dataset_path = os.path.join(WORKING_DIR, '{}-{}.train.pbz'.format(PROTOBUF_DATE, DESC))
    validation_dataset_path = os.path.join(WORKING_DIR, '{}-{}.valid.pbz'.format(PROTOBUF_DATE, DESC))
    dataset_index_path = os.path.join(WORKING_DIR, '{}-{}.index.pkl'.format(PROTOBUF_DATE, DESC))

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
        return data_generator


# ---------- Multiprocessing methods to generate proto buffer ----------------
def keep_phase_in_dataset(saved_game_proto):
    """ Filter function that decides if we should put a phase in the dataset
        :param saved_game_proto: A `.proto.game.SavedGame` object from the dataset.
        :return: A boolean that indicates if we should keep this phase or not.
    """
    # Keeping all games on the standard map
    if saved_game_proto.map.startswith('standard'):
        return True
    return False

def data_generator(saved_game_bytes, is_validation_set):
    """ Converts a dataset game to protocol buffer format
        :param saved_game_bytes: A `.proto.game.SavedGame` object from the dataset.
        :param is_validation_set: Boolean that indicates if we are generating the validation set (otw. training set)
        :return: A dictionary with phase_ix as key and a dictionary {power_name: (msg_len, proto)} as value
    """
    saved_game_proto = bytes_to_proto(saved_game_bytes, SavedGameProto)
    if not keep_phase_in_dataset(saved_game_proto):
        return {phase_ix: [] for phase_ix, _ in enumerate(saved_game_proto.phases)}
    return root_data_generator(saved_game_proto, is_validation_set)

def root_data_generator(saved_game_proto, is_validation_set):
    """ Converts a dataset game to protocol buffer format
        :param saved_game_proto: A `.proto.game.SavedGame` object from the dataset.
        :param is_validation_set: Boolean that indicates if we are generating the validation set (otw. training set)
        :return: A dictionary with phase_ix as key and a dictionary {power_name: (msg_len, proto)} as value
    """
    # Finding top victors and supply centers at end of game
    map_object = Map(saved_game_proto.map)
    top_victors = get_top_victors(saved_game_proto, map_object)
    nb_phases = len(saved_game_proto.phases)
    proto_results = {phase_ix: {} for phase_ix in range(nb_phases)}

    # Getting policy data for the phase_ix
    policy_data = get_policy_data(saved_game_proto, power_names=top_victors, top_victors=top_victors)

    # Building results
    for phase_ix in range(nb_phases - 1):
        for power_name in top_victors:
            phase_policy = policy_data[phase_ix][power_name]

            if 'decoder_inputs' not in phase_policy:
                continue

            request_id = DatasetBuilder.get_request_id(saved_game_proto, phase_ix, power_name, is_validation_set)
            data = {'request_id': request_id,
                    'player_seed': 0,
                    'decoder_inputs': [GO_ID],
                    'noise': 0.,
                    'temperature': 0.,
                    'dropout_rate': 0.,
                    'current_power': POWER_VOCABULARY_KEY_TO_IX[power_name],
                    'current_season': get_current_season(saved_game_proto.phases[phase_ix].state)}
            data.update(phase_policy)

            # Saving results
            proto_result = BaseDatasetBuilder.build_example(data, BaseDatasetBuilder.get_proto_fields())
            proto_results[phase_ix][power_name] = (0, proto_result)

    # Returning data for buffer
    return proto_results
