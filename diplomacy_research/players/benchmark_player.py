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
""" Benchmark Player
    - Contains classes to play published against published benchmarks
"""
import logging
import os
from multiprocessing import Process
import shutil
import time
import zipfile
from tornado import gen
from diplomacy import Game
from diplomacy_research.models.datasets.grpc_dataset import GRPCDataset, ModelConfig
from diplomacy_research.players.benchmarks import rl_neurips2019, sl_neurips2019
from diplomacy_research.players.model_based_player import ModelBasedPlayer
from diplomacy_research.utils.cluster import is_port_opened, kill_processes_using_port
from diplomacy_research.utils.process import start_tf_serving, download_file, kill_subprocesses_on_exit
from diplomacy_research.settings import WORKING_DIR

# Constants
LOGGER = logging.getLogger(__name__)
PERIOD_SECONDS = 10
MAX_SENTINEL_CHECKS = 3
MAX_TIME_BETWEEN_CHECKS = 300


class DipNetSLPlayer(ModelBasedPlayer):
    """ DipNet SL - NeurIPS 2019 Supervised Learning Benchmark Player """

    def __init__(self, temperature=0.1, use_beam=False, port=9501, name=None):
        """ Constructor
            :param temperature: The temperature to apply to the logits.
            :param use_beam: Boolean that indicates that we want to use a beam search.
            :param port: The port to use for the tf serving to query the model.
            :param name: Optional. The name of this player.
        """
        model_url = 'https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/neurips2019-sl_model.zip'

        # Creating serving if port is not open
        if not is_port_opened(port):
            launch_serving(model_url, port)

        # Creating adapter
        grpc_dataset = GRPCDataset(hostname='localhost',
                                   port=port,
                                   model_name='player',
                                   signature=sl_neurips2019.PolicyAdapter.get_signature(),
                                   dataset_builder=sl_neurips2019.BaseDatasetBuilder())
        policy_adapter = sl_neurips2019.PolicyAdapter(grpc_dataset)

        # Building benchmark model
        super(DipNetSLPlayer, self).__init__(policy_adapter=policy_adapter,
                                             temperature=temperature,
                                             use_beam=use_beam,
                                             name=name)

class DipNetRLPlayer(ModelBasedPlayer):
    """ DipNet RL - NeurIPS 2019 Reinforcement Learning Benchmark Player """

    def __init__(self, temperature=0.1, use_beam=False, port=9502, name=None):
        """ Constructor
            :param temperature: The temperature to apply to the logits.
            :param use_beam: Boolean that indicates that we want to use a beam search.
            :param port: The port to use for the tf serving to query the model.
            :param name: Optional. The name of this player.
        """
        model_url = 'https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/neurips2019-rl_model.zip'

        # Creating serving if port is not open
        if not is_port_opened(port):
            launch_serving(model_url, port)

        # Creating adapter
        grpc_dataset = GRPCDataset(hostname='localhost',
                                   port=port,
                                   model_name='player',
                                   signature=rl_neurips2019.PolicyAdapter.get_signature(),
                                   dataset_builder=rl_neurips2019.BaseDatasetBuilder())
        policy_adapter = rl_neurips2019.PolicyAdapter(grpc_dataset)

        # Building benchmark model
        super(DipNetRLPlayer, self).__init__(policy_adapter=policy_adapter,
                                             temperature=temperature,
                                             use_beam=use_beam,
                                             name=name)

class WebDiplomacyPlayer(ModelBasedPlayer):
    """ WebDiplomacy Player """

    def __init__(self, temperature=0.1, use_beam=False, port=9503, name=None):
        """ Constructor
            :param temperature: The temperature to apply to the logits.
            :param use_beam: Boolean that indicates that we want to use a beam search.
            :param port: The port to use for the tf serving to query the model.
            :param name: Optional. The name of this player.
        """
        model_url = 'https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/neurips2019-sl_model.zip'

        # Creating serving if port is not open
        if not is_port_opened(port):
            launch_serving(model_url, port)

        # Creating adapter
        grpc_dataset = GRPCDataset(hostname='localhost',
                                   port=port,
                                   model_name='player',
                                   signature=sl_neurips2019.PolicyAdapter.get_signature(),
                                   dataset_builder=sl_neurips2019.BaseDatasetBuilder())
        policy_adapter = sl_neurips2019.PolicyAdapter(grpc_dataset)

        # Building benchmark model
        super(WebDiplomacyPlayer, self).__init__(policy_adapter=policy_adapter,
                                                 temperature=temperature,
                                                 use_beam=use_beam,
                                                 name=name)


# ------ Utility Methods ------
def launch_serving(model_url, serving_port, first_launch=True):
    """ Launches or relaunches the TF Serving process
        :param model_url: The URL to use to download the model
        :param serving_port: The port to use for TF serving
        :param first_launch: Boolean that indicates if this is the first launch or a relaunch
    """
    model_url = model_url or ''
    bot_filename = model_url.split('/')[-1]
    bot_name = bot_filename.split('.')[0]
    bot_directory = os.path.join(WORKING_DIR, 'data', 'bot_%s' % bot_name)
    bot_model = os.path.join(bot_directory, bot_filename)

    # If first launch, downloading the model
    if first_launch:
        shutil.rmtree(bot_directory, ignore_errors=True)
        os.makedirs(bot_directory, exist_ok=True)

        # Downloading model
        download_file(model_url, bot_model, force=True)

        # Unzipping file
        zip_ref = zipfile.ZipFile(bot_model, 'r')
        zip_ref.extractall(bot_directory)
        zip_ref.close()

    # Otherwise, restarting the serving
    elif is_port_opened(serving_port):
        kill_processes_using_port(serving_port)

    # Launching a new process
    log_file_path = os.path.join(WORKING_DIR, 'data', 'log_serving_%d.txt' % serving_port)
    serving_process = Process(target=start_tf_serving,
                              args=(serving_port, WORKING_DIR),
                              kwargs={'force_cpu': True,
                                      'log_file_path': log_file_path})
    serving_process.start()
    kill_subprocesses_on_exit()

    # Waiting for port to be opened.
    for attempt_ix in range(90):
        time.sleep(10)
        if is_port_opened(serving_port):
            break
        LOGGER.info('Waiting for TF Serving to come online. - Attempt %d / %d', attempt_ix + 1, 90)
    else:
        LOGGER.error('TF Serving is not online after 15 minutes. Aborting.')
        raise RuntimeError()

    # Setting configuration
    new_config = ModelConfig(name='player', base_path='/work_dir/data/bot_%s' % bot_name, version_policy=None)
    for _ in range(30):
        if GRPCDataset.set_config('localhost', serving_port, new_config):
            LOGGER.info('Configuration set successfully.')
            break
        time.sleep(5.)
    else:
        LOGGER.error('Unable to set the configuration file.')

@gen.coroutine
def check_serving(player, serving_port):
    """ Makes sure the current serving process is still active, otherwise restarts it.
        :param player: A player object to query the server
        :param serving_port: The port to use for TF serving
    """
    game = Game()

    # Trying to check orders
    for _ in range(MAX_SENTINEL_CHECKS):
        orders = yield player.get_orders(game, 'FRANCE')
        if orders:
            return

    # Could not get orders x times in a row, restarting process
    LOGGER.warning('Could not retrieve orders from the serving process after %d attempts.', MAX_SENTINEL_CHECKS)
    LOGGER.warning('Restarting TF serving server.')
    launch_serving(None, serving_port)
