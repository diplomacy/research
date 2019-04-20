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
""" Reinforcement - Players Constructors
    - Class responsible for returning a list of instantiated players
"""

import logging
from diplomacy_research.models.datasets.grpc_dataset import GRPCDataset
from diplomacy_research.models.state_space import NB_POWERS
from diplomacy_research.players import ModelBasedPlayer

# Constants
LOGGER = logging.getLogger(__name__)
DEFAULT_TEMPERATURE = 1.
DEFAULT_NOISE = 0.
DEFAULT_DROPOUT = 0.

def get_train_supervised_players(adapter_ctor, dataset_builder_ctor, tf_serving_port, cluster_config, hparams):
    """ Builds 1 RL player vs 6 supervised opponent
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param tf_serving_port: The port to connect to the TF serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param hparams: A dictionary of hyper-parameters.
        :return: A list of players
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    player_dataset = GRPCDataset(hostname='localhost',
                                 port=tf_serving_port,
                                 model_name='player',
                                 signature=adapter_ctor.get_signature(),
                                 dataset_builder=dataset_builder_ctor(),
                                 cluster_config=cluster_config)
    opponent_dataset = GRPCDataset(hostname='localhost',
                                   port=tf_serving_port,
                                   model_name='opponent',
                                   signature=adapter_ctor.get_signature(),
                                   dataset_builder=dataset_builder_ctor(),
                                   cluster_config=cluster_config)

    # Adapters
    player_adapter = adapter_ctor(player_dataset)
    opponent_adapter = adapter_ctor(opponent_dataset)

    # Creating players
    players = []
    players += [ModelBasedPlayer(policy_adapter=player_adapter,
                                 temperature=DEFAULT_TEMPERATURE,
                                 noise=DEFAULT_NOISE,
                                 dropout_rate=hparams['dropout_rate'],
                                 use_beam=hparams['use_beam'])]
    for _ in range(NB_POWERS - 1):
        players += [ModelBasedPlayer(policy_adapter=opponent_adapter,
                                     temperature=DEFAULT_TEMPERATURE,
                                     noise=DEFAULT_NOISE,
                                     dropout_rate=hparams['dropout_rate'],
                                     use_beam=hparams['use_beam'])]
    return players

def get_train_self_play_players(adapter_ctor, dataset_builder_ctor, tf_serving_port, cluster_config, hparams):
    """ Build 7 RL players
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param tf_serving_port: The port to connect to the TF serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param hparams: A dictionary of hyper-parameters.
        :return: A list of players
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    player_dataset = GRPCDataset(hostname='localhost',
                                 port=tf_serving_port,
                                 model_name='player',
                                 signature=adapter_ctor.get_signature(),
                                 dataset_builder=dataset_builder_ctor(),
                                 cluster_config=cluster_config)
    player_adapter = adapter_ctor(player_dataset)

    # Creating players
    players = []
    for _ in range(NB_POWERS):
        players += [ModelBasedPlayer(policy_adapter=player_adapter,
                                     temperature=DEFAULT_TEMPERATURE,
                                     noise=DEFAULT_NOISE,
                                     dropout_rate=hparams['dropout_rate'],
                                     use_beam=hparams['use_beam'])]
    return players

def get_train_staggered_players(adapter_ctor, dataset_builder_ctor, tf_serving_port, cluster_config, hparams):
    """ Builds 1 RL player vs 6 previous versions
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param tf_serving_port: The port to connect to the TF serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param hparams: A dictionary of hyper-parameters.
        :return: A list of players
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    return get_train_supervised_players(adapter_ctor=adapter_ctor,
                                        dataset_builder_ctor=dataset_builder_ctor,
                                        tf_serving_port=tf_serving_port,
                                        cluster_config=cluster_config,
                                        hparams=hparams)

def get_eval_supervised_0_players(adapter_ctor, dataset_builder_ctor, tf_serving_port, cluster_config, hparams):
    """ Builds 1 RL player vs 6 supervised (temperature 0)
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param tf_serving_port: The port to connect to the TF serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param hparams: A dictionary of hyper-parameters.
        :return: A list of players
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    del hparams                         # Unused args
    player_dataset = GRPCDataset(hostname='localhost',
                                 port=tf_serving_port,
                                 model_name='player',
                                 signature=adapter_ctor.get_signature(),
                                 dataset_builder=dataset_builder_ctor(),
                                 cluster_config=cluster_config)
    opponent_dataset = GRPCDataset(hostname='localhost',
                                   port=tf_serving_port,
                                   model_name='opponent',
                                   signature=adapter_ctor.get_signature(),
                                   dataset_builder=dataset_builder_ctor(),
                                   cluster_config=cluster_config)

    # Adapters
    player_adapter = adapter_ctor(player_dataset)
    opponent_adapter = adapter_ctor(opponent_dataset)

    # Creating players
    players = []
    players += [ModelBasedPlayer(policy_adapter=player_adapter,
                                 temperature=DEFAULT_TEMPERATURE,
                                 noise=DEFAULT_NOISE,
                                 dropout_rate=DEFAULT_DROPOUT)]
    for _ in range(NB_POWERS - 1):
        players += [ModelBasedPlayer(policy_adapter=opponent_adapter,
                                     schedule=[(0.75, 0.1), (1., 1.)],
                                     noise=DEFAULT_NOISE,
                                     dropout_rate=DEFAULT_DROPOUT)]
    return players

def get_eval_supervised_1_players(adapter_ctor, dataset_builder_ctor, tf_serving_port, cluster_config, hparams):
    """ Builds 1 RL player vs 6 supervised (temperature 1)
        :param adapter_ctor: The constructor to build the adapter to query orders, values and policy details
        :param dataset_builder_ctor: The constructor of `BaseBuilder` to set the required proto fields
        :param tf_serving_port: The port to connect to the TF serving server
        :param cluster_config: The cluster configuration to use for distributed training
        :param hparams: A dictionary of hyper-parameters.
        :return: A list of players
        :type adapter_ctor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
        :type dataset_builder_ctor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
    """
    del hparams                         # Unused args
    player_dataset = GRPCDataset(hostname='localhost',
                                 port=tf_serving_port,
                                 model_name='player',
                                 signature=adapter_ctor.get_signature(),
                                 dataset_builder=dataset_builder_ctor(),
                                 cluster_config=cluster_config)
    opponent_dataset = GRPCDataset(hostname='localhost',
                                   port=tf_serving_port,
                                   model_name='opponent',
                                   signature=adapter_ctor.get_signature(),
                                   dataset_builder=dataset_builder_ctor(),
                                   cluster_config=cluster_config)

    # Adapters
    player_adapter = adapter_ctor(player_dataset)
    opponent_adapter = adapter_ctor(opponent_dataset)

    # Creating players
    players = []
    players += [ModelBasedPlayer(policy_adapter=player_adapter,
                                 temperature=DEFAULT_TEMPERATURE,
                                 noise=DEFAULT_NOISE,
                                 dropout_rate=DEFAULT_DROPOUT)]
    for _ in range(NB_POWERS - 1):
        players += [ModelBasedPlayer(policy_adapter=opponent_adapter,
                                     temperature=DEFAULT_TEMPERATURE,
                                     noise=DEFAULT_NOISE,
                                     dropout_rate=DEFAULT_DROPOUT)]
    return players
