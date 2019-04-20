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
""" Supervised Dataset
    - Class responsible for using a training and validation dataset to feed data to the model through tf.data.dataset
"""
from enum import Enum
import logging
import os
import math
import multiprocessing
import pickle
import numpy as np
from diplomacy_research.settings import WORKING_DIR

# Constants
LOGGER = logging.getLogger(__name__)

class TrainingMode(Enum):
    """ Enumeration of training modes """
    TRAINING = 'train'
    VALIDATION = 'valid'


class SupervisedDataset():
    """ This object is responsible for generating entries to feed the model (using the tf.data.dataset API) """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, batch_size, dataset_builder, checkpoint_dir='', cluster_config=None, debug_batch=False,
                 no_iterator=False, do_infinite_training=False, perc_epoch_for_training=1.):
        """ Constructor
            :param batch_size: The size of a batch per tower
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param checkpoint_dir: The directory where the status is to be saved. None to disable, '' for default dir.
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
            :param debug_batch: Boolean flag to indicate to return the same batch over-and-over to debug our model
            :param no_iterator: Boolean flag that indicates to not create an iterator (it will be loaded from a ckpt)
            :param do_infinite_training: If set, supervised training will loop over the training set forever
                                         and will not switch to the validation set.
            :param perc_epoch_for_training: If set, the training epoch will be for this percentage of available steps
                                     before running another evaluation epoch (e.g. 2.5% train, valid, 2.5% train, ...)
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        # pylint: disable=too-many-arguments
        self._batch_size = batch_size
        self.dataset_builder = dataset_builder
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else WORKING_DIR       # None = disabled
        self.cluster_config = cluster_config
        self.debug_batch = debug_batch
        self.no_iterator = no_iterator
        self.perc_epoch_for_training = 1.00 if do_infinite_training else max(1e-3, min(1., perc_epoch_for_training))
        self.do_infinite_training = do_infinite_training
        self.is_closing = False
        self.session = None

        # Creating empty datasets
        self.training_dataset = None
        self.validation_dataset = None
        self.feedable_dataset = None

        # Creating iterator with init ops
        self.iterator = None
        self._iterator_initialized = False
        self.training_init_op = None
        self.validation_init_op = None
        self.output_features = None                 # This represents iterator.get_next()
        self.default_features = {}                  # Will be used as default if features are missing from queue

        # Steps
        self.nb_batches_to_skip = 0                 # Nb of batches to skip
        self.steps_in_current_mode = 0              # Step count in current mode
        self.training_progress = 0.

        # Number of items remaining in epoch
        self.total_nb_items_training_proto = 0
        self.total_nb_items_valid_proto = 0
        self.training_mode = TrainingMode.TRAINING
        self.nb_completed_epochs = 0
        self._dataset_is_done = False

        # Loading number of items remaining
        if os.path.exists(self.dataset_builder.dataset_index_path) \
                and os.path.getsize(self.dataset_builder.dataset_index_path):
            with open(self.dataset_builder.dataset_index_path, 'rb') as dataset_index:
                dataset_index = pickle.load(dataset_index)
            self.total_nb_items_training_proto = dataset_index['size_train_dataset']
            self.total_nb_items_valid_proto = dataset_index['size_valid_dataset']

        # Building the datasets
        self.build()

    @property
    def can_support_iterator(self):
        """ Determines if the dataset can support an iterator or if it is a remote (RPC) dataset """
        return True

    @property
    def batch_size(self):
        """ Getter for batch_size """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """ Setter for batch_size """
        if self.num_shards is not None:
            raise RuntimeError('You cannot change the batch_size when using shards')
        self._batch_size = value

    @property
    def num_shards(self):
        """ Returns the number of shards (if a cluster config is set), otherwise None """
        return self.cluster_config.num_shards if self.cluster_config else 1

    @property
    def nb_training_steps_per_epoch(self):
        """ Returns the number of training steps per epoch """
        nb_items_per_epoch = self.perc_epoch_for_training * self.total_nb_items_training_proto
        return int(math.ceil(nb_items_per_epoch / (self.batch_size * self.num_shards)))

    @property
    def nb_training_steps_per_full_epoch(self):                                                                         # pylint: disable=invalid-name
        """ Returns the number of training steps per full epoch """
        return int(math.ceil(self.total_nb_items_training_proto / (self.batch_size * self.num_shards)))

    @property
    def nb_validation_steps_per_epoch(self):
        """ Returns the number of validation steps per epoch """
        return int(math.ceil(self.total_nb_items_valid_proto / (self.batch_size * self.num_shards)))

    @property
    def nb_total_steps_per_epoch(self):
        """ Returns the total number of training and validation steps per epoch """
        return self.nb_training_steps_per_epoch + self.nb_validation_steps_per_epoch

    @property
    def nb_steps_per_epoch_current_mode(self):
        """ Returns the number of steps per epoch in the current mode (Training / Validation) """
        if self.training_mode == TrainingMode.VALIDATION:
            return self.nb_validation_steps_per_epoch
        return self.nb_training_steps_per_epoch

    @property
    def iterator_initialized(self):
        """ Determine if the iterator has been initialized """
        return self._iterator_initialized

    @property
    def status_path(self):
        """ Path to the status file on disk (where progress is saved) """
        if not self.checkpoint_dir:
            return None
        if not self.cluster_config:
            return os.path.join(self.checkpoint_dir, 'status.pkl')
        return os.path.join(self.checkpoint_dir, 'status', 'status-%03d.pkl' % self.cluster_config.task_id)

    @property
    def chief_status_path(self):
        """ Path to the chief status path (to validate our status) """
        if not self.cluster_config:
            return None
        return os.path.join(self.checkpoint_dir, 'status', 'status-%03d.pkl' % 0)

    @property
    def fallback_status_path(self):
        """ Path to an alternate status file if the primary is not available """
        fallbacks = [os.path.join(self.checkpoint_dir, 'status', 'status-%03d.pkl' % 0),
                     os.path.join(self.checkpoint_dir, 'status.pkl')]
        for fallback in fallbacks:
            if os.path.exists(fallback):
                return fallback
        return None

    @property
    def is_done(self):
        """ Returns True if the end of file has been reached """
        if self.do_infinite_training:
            return False
        return self._dataset_is_done or self.steps_in_current_mode >= self.nb_steps_per_epoch_current_mode

    def take_local_step(self):
        """ Increments the local step counter """
        if not self.is_done or self.do_infinite_training:
            self.steps_in_current_mode += 1
            if self.training_mode == TrainingMode.TRAINING:
                self.training_progress = (self.training_progress + 1. / self.nb_training_steps_per_full_epoch) % 1

    def mark_as_done(self):
        """ Marks the dataset as having reached the end of the file"""
        self._dataset_is_done = True

    def build(self):
        """ Builds the TensorFlow datasets """
        from diplomacy_research.utils.tensorflow import tf
        assert 'request_id' in self.dataset_builder.get_proto_fields(), 'You need to have a "request_id" field.'

        # Training dataset
        self.training_dataset = tf.data.TFRecordDataset(self.dataset_builder.training_dataset_path,
                                                        compression_type='GZIP')

        # Debug (batch) mode
        # Only taking one batch and looping over that batch forever
        if self.debug_batch:
            self.training_dataset = self.training_dataset.take(self.batch_size)
            self.training_dataset = self.training_dataset.repeat(count=-1)

        # Regular mode
        # Otherwise, sharding and shuffling the dataset
        # Repeating to make sure all workers can loop on the dataset at all times
        else:
            if self.cluster_config and self.num_shards > 1:
                LOGGER.info('Sharding dataset. There are %d shards. Current shard index: #%d.',
                            self.cluster_config.num_shards, self.cluster_config.shard_index)
                shard_fn = tf.data.experimental.filter_for_shard(num_shards=self.cluster_config.num_shards,
                                                                 shard_index=self.cluster_config.shard_index)
                self.training_dataset = self.training_dataset.apply(shard_fn)
                self.training_dataset = self.training_dataset.repeat()
            self.training_dataset = self.training_dataset.shuffle(100 * self.batch_size)

        # Batching with prefetching
        self.training_dataset = self.training_dataset.map(self.dataset_builder.parse_function,
                                                          num_parallel_calls=multiprocessing.cpu_count())
        self.training_dataset = self.training_dataset.prefetch(100 * self.batch_size)
        self.training_dataset = self.training_dataset.padded_batch(self.batch_size,
                                                                   padded_shapes=self.dataset_builder.padded_shapes)

        # Building a list of generic default values from the output types and output shapes
        self.default_features = {}
        for feature_name, feature_shape in self.dataset_builder.output_shapes.items():
            if self.dataset_builder.output_types[feature_name] == np.object:
                self.default_features[feature_name] = bytes('', 'utf-8')
            else:
                dtype = self.dataset_builder.output_types[feature_name]
                self.default_features[feature_name] = np.zeros(shape=feature_shape[1:], dtype=dtype)

        # -----------------------------
        # Validation dataset
        self.validation_dataset = tf.data.TFRecordDataset(self.dataset_builder.validation_dataset_path,
                                                          compression_type='GZIP')

        # Sharding, but no need to shuffle
        if self.cluster_config and self.num_shards > 1:
            shard_fn = tf.data.experimental.filter_for_shard(num_shards=self.cluster_config.num_shards,
                                                             shard_index=self.cluster_config.shard_index)
            self.validation_dataset = self.validation_dataset.apply(shard_fn)

        # Batching with prefetching
        self.validation_dataset = self.validation_dataset.map(self.dataset_builder.parse_function,
                                                              num_parallel_calls=multiprocessing.cpu_count())
        self.validation_dataset = self.validation_dataset.prefetch(20 * self.batch_size)
        self.validation_dataset = self.validation_dataset.padded_batch(self.batch_size,
                                                                       padded_shapes=self.dataset_builder.padded_shapes)

        # Creating iterator (with a new iterator_resource), unless specified otherwise
        if not self.no_iterator:
            self.create_iterator()

    def create_iterator(self, iterator_resource=None, shared_name=None, features=None):
        """ Creates an iterator object (optionally using a shared name and a specific iterator resource)

            :param iterator_resource: A tf.resource scalar tf.Tensor representing the iterator.
            :param shared_name: Optional. If non-empty, this iterator will be shared under the given name across
                                multiple sessions that share the same devices (e.g. when using a remote server).
            :param features: If an iterator_resource is specified, this corresponds to the output of iterator.get_next()
            :return: Nothing, but sets the self.iterator, self.features, and dataset init_ops
        """
        if iterator_resource is not None and not self.no_iterator:
            LOGGER.error('An iterator resource can only be set if the dataset was created with the "no_iterator" flag.')
            raise RuntimeError("Cannot create new iterator")
        if iterator_resource is not None and features is None:
            LOGGER.error('The iterator features are required when reloading a saved iterator.')
            raise ValueError()

        # Loading TensorFlow
        from diplomacy_research.utils.tensorflow import tf

        output_types = self.training_dataset.output_types
        output_shapes = self.training_dataset.output_shapes
        output_classes = self.training_dataset.output_classes

        # Making sure itertor is on the right device/worker
        with tf.device(self.cluster_config.iterator_device if self.cluster_config else None):

            # We have an iterator resource, so we use it
            if iterator_resource is not None:
                self.iterator = tf.data.Iterator(iterator_resource=iterator_resource,
                                                 initializer=None,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 output_classes=output_classes)
                if features:
                    self.output_features = features

            # Otherwise, we create a brand new iterator
            else:
                self.iterator = tf.data.Iterator.from_structure(output_types=output_types,
                                                                output_shapes=output_shapes,
                                                                output_classes=output_classes,
                                                                shared_name=shared_name)
                self.output_features = self.iterator.get_next()

            # Generating init op for each dataset
            # Using different names because we can't define initializers with the same name
            self._iterator_initialized = False
            self.training_init_op = self.iterator.make_initializer(self.training_dataset)
            self.validation_init_op = self.iterator.make_initializer(self.validation_dataset)

    def initialize_iterator(self, session):
        """ Initializes the current iterator
            :param session: The session used to initialize the init op
            :type session: tensorflow.python.client.session.Session
        """
        # We haven't created an iterator yet
        if self.iterator is None:
            return

        # Loading TensorFlow
        from diplomacy_research.utils.tensorflow import tf

        # Running init_op
        # If session is wrapped, executing it without hooks
        init_op = {TrainingMode.TRAINING: self.training_init_op,
                   TrainingMode.VALIDATION: self.validation_init_op}[self.training_mode]
        if hasattr(session, 'run_step_fn'):
            session.run_step_fn(lambda step_context: step_context.session.run(init_op))
        else:
            session.run(init_op)
        self._iterator_initialized = True
        self._dataset_is_done = False

        # For validation set, we can reset the steps since we are always starting from the beginning
        # For training, we might resume mid-epoch (from load_status()) - So we keep the current value
        if self.training_mode == TrainingMode.VALIDATION:
            self.steps_in_current_mode = 0

        # Resuming by skipping a certain number of already processed items
        if self.nb_batches_to_skip:
            LOGGER.info('Resuming training by skipping %d batches in the training dataset.', self.nb_batches_to_skip)
            try:
                for _ in range(self.nb_batches_to_skip):
                    if hasattr(session, 'run_step_fn'):
                        session.run_step_fn(
                            lambda step_context: step_context.session.run(self.output_features['request_id']))
                    else:
                        session.run(self.output_features['request_id'])
            except tf.errors.OutOfRangeError:
                self.mark_as_done()
        self.nb_batches_to_skip = 0

    def start_training_mode(self, session):
        """ Starts the dataset in training mode
            :param session: The session used to initialize the init op
            :type session: tensorflow.python.client.session.Session
        """
        if self.is_done:
            self.nb_completed_epochs += 1
            self.nb_batches_to_skip = int(self.training_progress * self.nb_training_steps_per_full_epoch)
        self.training_mode = TrainingMode.TRAINING
        self.steps_in_current_mode = 0
        self.initialize_iterator(session)

    def start_validation_mode(self, session):
        """ Starts the dataset in validation mode
            :param session: The session used to initialize the init op
            :type session: tensorflow.python.client.session.Session
        """
        if self.do_infinite_training:
            LOGGER.error('Dataset is currently in "infinite training" mode. Only the training set can be accessed.')
            raise RuntimeError('Invalid training mode specified.')
        self.training_mode = TrainingMode.VALIDATION
        self.steps_in_current_mode = 0
        self.initialize_iterator(session)

    def get_progress(self):
        """ Returns the number of completed epochs, and the current % of the epoch completed """
        if self.do_infinite_training:
            self.nb_completed_epochs = int(self.steps_in_current_mode / self.nb_training_steps_per_full_epoch)
        perc_epoch_completed = self.steps_in_current_mode / self.nb_steps_per_epoch_current_mode
        return self.nb_completed_epochs, perc_epoch_completed

    def save_status(self):
        """ Save current status to file to be able to resume later """
        # Not saving status if checkpoint_dir is None
        if not self.status_path:
            return

        # Recomputing nb of completed epochs when doing infinite training
        if self.do_infinite_training:
            self.nb_completed_epochs = int(self.steps_in_current_mode / self.nb_training_steps_per_full_epoch)

        # Creating directory and saving
        if not os.path.exists(os.path.dirname(self.status_path)):
            os.makedirs(os.path.dirname(self.status_path), exist_ok=True)

        status = {'training_mode': self.training_mode,
                  'nb_completed_epochs': self.nb_completed_epochs,
                  'steps_current_mode': self.steps_in_current_mode,
                  'training_progress': self.training_progress,
                  'num_shards': self.num_shards}
        with open(self.status_path, 'wb') as file:
            pickle.dump(status, file, pickle.HIGHEST_PROTOCOL)

    def load_status(self):
        """ Loads dataset status from disk and resume where we were """
        status = {}
        status_loaded = False

        # Not loading status if checkpoint_dir is None.
        if not self.status_path:
            return

        # Trying to load from primary path
        if os.path.exists(self.status_path) and os.path.getsize(self.status_path):
            with open(self.status_path, 'rb') as status:
                status = pickle.load(status)

            # Detecting num of shards change and deleting file if that's the case
            if self.num_shards == status['num_shards']:
                status_loaded = True
            else:
                LOGGER.info('Number of shards has changed from %d to %d', status['num_shards'], self.num_shards)

                # If we are chief, we do a cleanup on the status folder
                if self.cluster_config and self.cluster_config.is_chief:
                    for status_ix in range(self.num_shards, status['num_shards']):
                        if os.path.exists(os.path.join(self.checkpoint_dir, 'status', 'status-%03d.pkl' % status_ix)):
                            os.unlink(os.path.join(self.checkpoint_dir, 'status', 'status-%03d.pkl' % status_ix))

                # Otherwise, we just delete the worker status file
                else:
                    os.unlink(self.status_path)

        # We load the fallback status
        if not status_loaded and self.fallback_status_path:
            try:
                with open(self.fallback_status_path, 'rb') as status:
                    status = pickle.load(status)
                status_loaded = True
            except EOFError:
                pass

        # We load the chief status to validate that we have the same training_mode and nb_epochs
        if self.cluster_config and os.path.exists(self.chief_status_path) and os.path.getsize(self.chief_status_path):
            with open(self.chief_status_path, 'rb') as chief_status:
                chief_status = pickle.load(chief_status)
        else:
            chief_status = status

        # We couldn't find a status file to load, aborting
        if not status_loaded:
            return

        # If we have the same value as the chief, we load our status, otherwise we use the chief
        use_own_status = ((status['training_mode'] == chief_status['training_mode'])
                          and status['nb_completed_epochs'] == chief_status['nb_completed_epochs'])

        # Loading status
        self._iterator_initialized = False
        if use_own_status:
            self.training_mode = status['training_mode']
            self.nb_completed_epochs = status['nb_completed_epochs']
            self.steps_in_current_mode = status['steps_current_mode']
            self.training_progress = status['training_progress']
            if self.training_mode == TrainingMode.VALIDATION:
                self.steps_in_current_mode = 0
        else:
            LOGGER.warning('Status between worker and chief does not match. Resuming using chief status.')
            self.training_mode = chief_status['training_mode']
            self.nb_completed_epochs = chief_status['nb_completed_epochs']
            self.steps_in_current_mode = chief_status['steps_current_mode']
            self.training_progress = chief_status['training_progress']
            if self.training_mode == TrainingMode.VALIDATION:
                self.steps_in_current_mode = 0

        # If we were training the train dataset, we need to skip a certain number of batches
        # to get to the same training point
        if self.training_mode == TrainingMode.TRAINING:
            self.nb_batches_to_skip = int(self.training_progress * self.nb_training_steps_per_full_epoch)

    def make_session_run_hook(self):
        """ Builds a SessionRunHook for the MonitoredTrainingSession object """
        from diplomacy_research.utils.tensorflow import SupervisedDatasetSessionRunHook
        return SupervisedDatasetSessionRunHook(self)

    def close(self):
        """ Stops iterating the dataset """
        self.is_closing = True
        self.training_dataset = None
        self.validation_dataset = None
