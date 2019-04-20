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
""" Supervised Trainer
    - Contains a trainer class for training supervised models
"""
import logging
import os
import shutil
import signal
import sys
import time
from diplomacy_research.models.training.memory_buffer import MemoryBuffer                                               # pylint: disable=unused-import
from diplomacy_research.models.training.supervised.common import has_already_early_stopped
from diplomacy_research.models.training.supervised.distributed import start_distributed_training
from diplomacy_research.models.training.supervised.standalone import start_standalone_training
from diplomacy_research.settings import SESSION_RUN_TIMEOUT

# Constants
LOGGER = logging.getLogger(__name__)

class SupervisedTrainer():
    """ Performs supervised training on a model """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, policy_constructor, value_constructor, draw_constructor, dataset_builder, adapter_constructor,
                 flags, cluster_config=None):
        """ Constructor
            :param policy_constructor: The constructor to create the policy model
            :param value_constructor: Optional. The constructor to create the value model
            :param draw_constructor: Optional. The constructor to create the draw model.
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param adapter_constructor: The constructor to build the adapter for the model type.
            :param flags: The parsed flags (Tensorflow arguments)
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
            :type policy_constructor: diplomacy_research.models.policy.base_policy_model.BasePolicyModel.__class__
            :type value_constructor: diplomacy_research.models.value.base_value_model.BaseValueModel.__class__
            :type draw_constructor: diplomacy_research.models.draw.base_draw_model.BaseDrawModel.__class__
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type adapter_constructor: diplomacy_research.models.policy.base_policy_adapter.BasePolicyAdapter.__class__
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        # pylint: disable=too-many-arguments
        self.cluster_config = cluster_config
        self.supervised_dataset = None
        self.dataset_builder = dataset_builder
        self.signature = adapter_constructor.get_signature()
        self.model = None
        self.server = None
        self.config_proto = None
        self.policy_constructor = policy_constructor
        self.value_constructor = value_constructor
        self.draw_constructor = draw_constructor

        # Hooks
        self.hooks = []
        self.chief_only_hooks = []

        # Activating debug (batch) mode if the debug flag is set
        self.flags = flags
        if self.flags.debug_batch:
            self.set_debug_batch_mode()
        self.hparams = flags.__dict__

        # Making sure 'learning_rate' and 'lr_decay_factor' exists:
        if not hasattr(self.flags, 'learning_rate'):
            raise RuntimeError('Your model is missing a "learning_rate" argument.')
        if not hasattr(self.flags, 'lr_decay_factor'):
            raise RuntimeError('Your model is missing a "lr_decay_factor" argument.')

        # Supervised settings
        self.status_every = 20 if self.flags.debug_batch else 60
        self.history_saver = None
        self.progress = (0, 0)                              # (nb_epochs_completed, current_progress)

        # Recording current rate
        self.learning_rate = self.flags.learning_rate if 'learning_rate' in self.flags else 0.

        # Recording starting time
        self.starting_time = int(time.time())
        self.status_time = self.starting_time
        self.step_last_status = 0

        # Early stopping
        self.performance = {}                               # {early_stopping_tag: {epoch_number: tag_value}}

        # Profiler settings
        self.profiler = None
        self.nb_profile_steps = 0
        self.nb_oom_steps = 0

        # Tensorboard variables
        self.placeholders = {}
        self.summaries = {}
        self.merge_ops = {}
        self.writers = {}

        # Memory Buffer (Barrier)
        self.memory_buffer = None                           # type: MemoryBuffer

    def set_debug_batch_mode(self):
        """ Puts the supervised training in debug mode (where we are overfitting a single minibatch """
        if self.cluster_config:
            LOGGER.error('Debug (batch) mode is not available while using distributed training.')
            return

        self.flags.batch_size = 8
        LOGGER.info('Training model in DEBUG (batch) mode - Overfitting a single mini-batch of 8 examples')

        # Disabling dropout
        for flag in self.flags.__dict__:
            if 'dropout' in flag and not isinstance(getattr(self.flags, flag), bool):
                setattr(self.flags, flag, 0.)
                LOGGER.info('The flag "%s" has been set to 0 because dropout is disabled in DEBUG (batch) mode.', flag)

        # Deleting training dir
        if os.path.exists(self.flags.save_dir):
            shutil.rmtree(self.flags.save_dir)

    def start(self):
        """ Starts training the supervised model """
        if has_already_early_stopped(self):
            LOGGER.info('Model has already triggered early stopping. Exiting.')
            return

        # Handling CTRL+C
        def signal_handler(*args):
            """ Handles SIGINT and SIGTERM signals """
            del args    # unused argument
            print('INFO - CTRL-C received. Stopping training.')     # Not using LOGGER, might be already closed
            if self.model.sess:
                self.model.sess.close()                             # Should trigger the SessionRunHooks
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if not self.cluster_config:
            start_standalone_training(self)
        else:
            start_distributed_training(self)

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
        return SupervisedTrainer.run_func_without_hooks(sess, lambda _sess: _sess.run(fetches, feed_dict=feed_dict))

    @staticmethod
    def run_func_without_hooks(sess, func):
        """ Executes a function that depends on a raw TensorFlow session without hooks
            :param sess: The wrapped TensorFlow session.
            :param func: A function to execute (args: session [the raw TF session])
        """
        if hasattr(sess, 'run_step_fn'):
            return sess.run_step_fn(lambda step_context: func(step_context.session))
        return func(sess)

    def run_session(self, sess, session_args, run_metadata=None):
        """ Executes the session.
            :param sess: The wrapped TensorFlow session.
            :param session_args: A dictionary with 'fetches' and 'feed_dict'
            :param run_metadata: Optional. Metadata to add to the session.run
            :return: The fetches
        """
        from diplomacy_research.utils.tensorflow import tf

        # Setting options and meta-data
        if not self.flags.debug:
            session_args['options'] = tf.RunOptions(timeout_in_ms=SESSION_RUN_TIMEOUT)
        if run_metadata is not None:
            session_args['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,                               # pylint: disable=no-member
                                                    timeout_in_ms=SESSION_RUN_TIMEOUT)
            session_args['run_metadata'] = run_metadata

        # Running
        return sess.run(**session_args)
