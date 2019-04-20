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
""" Reinforcement Trainer
    - Contains a trainer class for training reinforcement learning algorithms
"""
import logging
import os
import shutil
import signal
import sys
import time
from tornado import gen, ioloop
from diplomacy_research.models.datasets.queue_dataset import QueueDataset                                               # pylint: disable=unused-import
from diplomacy_research.models.self_play import reward_functions
from diplomacy_research.models.training.memory_buffer import MemoryBuffer                                               # pylint: disable=unused-import
from diplomacy_research.models.training.reinforcement.distributed import start_distributed_training
from diplomacy_research.models.training.reinforcement.serving import get_tf_serving_port
from diplomacy_research.models.training.reinforcement.standalone import start_standalone_training
from diplomacy_research.utils.cluster import is_ioloop_running, kill_processes_using_port

# Constants
LOGGER = logging.getLogger(__name__)

class ReinforcementTrainer():
    """ Performs reinforcement learning training on a model """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, policy_constructor, value_constructor, draw_constructor, dataset_builder_constructor,
                 adapter_constructor, algorithm_constructor, reward_fn, flags, cluster_config, process_pool=None):
        """ Constructor
            :param policy_constructor: The constructor to create the policy model (used by actors and learners)
            :param value_constructor: The constructor to create the value model (used by actors and learners)
            :param draw_constructor: The constructor to create the draw model (to accept a draw or not)
            :param dataset_builder_constructor: The constructor of `BaseBuilder` to set the required proto fields
            :param adapter_constructor: The constructor to build the adapter to query orders, values and policy details
            :param algorithm_constructor: The constructor to build the algorithm to train the model
            :param reward_fn: The reward function to use (Instance of.models.self_play.reward_functions`).
            :param flags: The parsed flags (Tensorflow arguments)
            :param cluster_config: The cluster configuration to use for distributed training
            :param process_pool: Optional. A ProcessPoolExecutor that was forked before TF and gRPC were loaded.
            :type policy_constructor: diplomacy_research.models.policy.base_policy_model.BasePolicyModel.__class__
            :type value_constructor: diplomacy_research.models.value.base_value_model.BaseValueModel.__class__
            :type draw_constructor: diplomacy_research.models.draw.base_draw_model.BaseDrawModel.__class__
            :type dataset_builder_constructor: diplomacy_research.models.datasets.base_builder.BaseBuilder.__class__
            :type adapter_constructor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
            :type algorithm_constructor: diplomacy_research.models.self_play.algorithms.base_algorithm.BaseAlgorithm.__class__   # pylint: disable=line-too-long
            :type reward_fn: diplomacy_research.models.self_play.reward_functions.AbstractRewardFunction
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
            :type process_pool: diplomacy_research.utils.executor.ProcessPoolExecutor
        """
        # pylint: disable=too-many-arguments
        self.server = None
        self.model = None
        self.queue_dataset = None                           # type: QueueDataset
        self.adapter = None
        self.algorithm = None
        self.reward_fn = reward_fn
        self.eval_reward_fns = [getattr(reward_functions, rwd_name)() for rwd_name in flags.eval_reward_fns.split(',')]
        self.advantage_fn = None
        self.flags = flags
        self.cluster_config = cluster_config
        self.process_pool = process_pool
        self.dataset_builder = dataset_builder_constructor(extra_proto_fields=algorithm_constructor.get_proto_fields())
        self.signature = adapter_constructor.get_signature()
        self.proto_fields = self.dataset_builder.proto_fields

        # Constructors
        self.policy_constructor = policy_constructor
        self.value_constructor = value_constructor
        self.draw_constructor = draw_constructor
        self.dataset_builder_constructor = dataset_builder_constructor
        self.adapter_constructor = adapter_constructor
        self.algorithm_constructor = algorithm_constructor

        # Sentinel - Monitors the TF serving server and acts as a proxy to send requests to it
        # Aggregator - Receives games from the actors on different processes and saves them to disk / memory_buffer
        # There is one sentinel per tf serving server, there might be multiple serving server launched from a process
        # So we use a serving_id to differentiate between the different serving_ids
        self.sentinels = {}                                     # {serving_id: Process}
        self.thread_pipes = {}                                  # {serving_id: Pipe}
        self.sentinel_pipes = {}                                # {serving_id: Pipe}
        self.aggregator = {'train': None, 'eval': None}         # {'train': Process, 'eval': Process}

        # Train thread is a thread to generate an infinite number of games
        # Eval thread is a thread that automatically generates stats every 'x' version updates
        self.train_thread = None
        self.eval_thread = None

        self.session = None
        self.server = None
        self.config_proto = None
        self.saver = None
        self.restore_saver = None
        self.memory_buffer = None                           # type: MemoryBuffer

        # Hooks
        self.hooks = []
        self.chief_only_hooks = []

        # Debug (batch) Mode is not possible for Reinforcement Learning
        if self.flags.debug_batch:
            LOGGER.warning('Debug (batch) mode in RL mode only clears the checkpoint directory (i.e. save_dir)')
            self.set_debug_batch_mode()

        # Validating hparams
        assert self.flags.mode in ['supervised', 'self-play', 'staggered']
        assert self.flags.eval_mode in ['supervised-1', 'supervised-0']

        # Extracting hparams
        self.hparams = flags.__dict__

        # Stats
        self.starting_time = int(time.time())
        self.last_version_time = time.time()
        self.last_checkpoint_time = 0
        self.last_stats_time = 0
        self.checkpoint_every = 600
        self.stats_every = 60
        self.stats = {}
        self.replay_samples = []

        # Tensorboard
        self.placeholders = {}
        self.summaries = {}
        self.merge_op = None
        self.writers = {}

    def set_debug_batch_mode(self):
        """ Puts the RL training in debug mode (i.e. only deletes the save_dir folder) """
        if os.path.exists(self.flags.save_dir):
            shutil.rmtree(self.flags.save_dir)

    @gen.coroutine
    def start(self):
        """ Starts training the RL model """
        if not is_ioloop_running():
            ioloop.IOLoop().run_sync(self.start)
            return

        # Handling CTRL+C
        def signal_handler(*args):
            """ Handles SIGINT and SIGTERM signals """
            del args  # unused argument
            print('INFO - CTRL-C received. Stopping training.')     # Not using LOGGER, might be already closed
            for sentinel in self.sentinels.values():
                if sentinel is not None:
                    sentinel.terminate()
            for process in self.aggregator.values():
                if process is not None:
                    process.terminate()
            if self.process_pool is not None:
                self.process_pool.shutdown()
                kill_processes_using_port(get_tf_serving_port(self.cluster_config, serving_id=0))
                kill_processes_using_port(get_tf_serving_port(self.cluster_config, serving_id=1))
            if self.model and getattr(self.model, 'sess', None) is not None:
                self.model.sess.close()                             # Should trigger the SessionRunHooks
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if not self.cluster_config:
            yield start_standalone_training(self)
        else:
            yield start_distributed_training(self)

    # ----------------------------------------------------
    # Public Methods - Utilities
    # ----------------------------------------------------
    @staticmethod
    def run_without_hooks(sess, fetches, feed_dict=None):
        """ Executes a raw TensorFlow session without executing the registered hooks
            :param sess: The wrapped TensorFlow session.
            :param fetches: The fetches to retrieve from the session
            :param feed_dict: The feed_dict to pass to the session
        """
        return ReinforcementTrainer.run_func_without_hooks(sess, lambda _sess: _sess.run(fetches, feed_dict=feed_dict))

    @staticmethod
    def run_func_without_hooks(sess, func):
        """ Executes a function that depends on a raw TensorFlow session without hooks
            :param sess: The wrapped TensorFlow session.
            :param func: A function to execute (args: session [the raw TF session])
        """
        if hasattr(sess, 'run_step_fn'):
            return sess.run_step_fn(lambda step_context: func(step_context.session))
        return func(sess)
