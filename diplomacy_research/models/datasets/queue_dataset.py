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
""" Queue Dataset
    - Class responsible for putting entries in a queue and feeding them to the model through tf.data.dataset
"""
import logging
from queue import Queue
import time
from threading import Thread
import uuid

import numpy as np
from tornado.concurrent import Future

from diplomacy_research.models.datasets.base_builder import VarProtoField
from diplomacy_research.models.datasets.feedable_dataset import FeedableDataset
from diplomacy_research.utils.cluster import get_current_io_loop, PrefetchedItem
from diplomacy_research.settings import SESSION_RUN_TIMEOUT

# Constants
LOGGER = logging.getLogger(__name__)
RUNNING, PAUSED, CLOSED = 'RUNNING', 'PAUSED', 'CLOSED'
FILLING_QUEUE = '<filling>'

# ----------------------------------------------------------------------------
# ----------               THREADS METHODS                    ----------------
# ----------------------------------------------------------------------------
def run_single_or_list_callable(callable_or_list, *args, **kwargs):
    """ Calls a single callable or a list of callable and returns a list of results
        :param callable_or_list: A single callable or a list of callables
    """
    if isinstance(callable_or_list, list):
        return [func(*args, **kwargs) for func in callable_or_list]
    return [callable_or_list(*args, **kwargs)]

def process_queues(dataset, in_main_thread=False):
    """ This method will be launched in a separate thread and is in charge of rotating through the queues
        :param dataset: The instantiated feedable dataset
        :param in_main_thread: Boolean that indicates that we are running in the main Python thread.
        :type dataset: QueueDataset
    """
    from diplomacy_research.utils.tensorflow import tf
    assert dataset.session is not None, 'Error - The dataset must have a session object attached to it'
    def_options = tf.RunOptions(timeout_in_ms=SESSION_RUN_TIMEOUT)

    while not dataset.all_threads_closing:
        all_queues_are_empty = True

        # Thread is paused
        if dataset.all_threads_paused and not in_main_thread:
            dataset.thread_status['process_queues'] = PAUSED
            while dataset.all_threads_paused and not dataset.all_threads_closing:
                time.sleep(0.1)

        # Thread resumed / started
        if not in_main_thread:
            dataset.thread_status['process_queues'] = RUNNING

        # Filling queues - Pausing
        if dataset.last_queue == FILLING_QUEUE:
            time.sleep(0.1)
            continue

        # Looping through every queue
        for queue_name in list(dataset.feedable_queues):
            queue = dataset.feedable_queues[queue_name]['queue']
            outputs = dataset.feedable_queues[queue_name]['outputs']
            placeholders = dataset.feedable_queues[queue_name]['placeholders']
            with_status = dataset.feedable_queues[queue_name]['with_status']
            pre_condition_hook = dataset.feedable_queues[queue_name]['hooks'].get('pre_condition', None)
            pre_run_hook = dataset.feedable_queues[queue_name]['hooks'].get('pre_run', None)
            post_run_hook = dataset.feedable_queues[queue_name]['hooks'].get('post_run', None)
            pre_queue_hook = dataset.feedable_queues[queue_name]['hooks'].get('pre_queue', None)
            post_queue_hook = dataset.feedable_queues[queue_name]['hooks'].get('post_queue', None)
            queue_size = queue.qsize()

            # [Hook] Pre-Condition
            if pre_condition_hook is not None:
                if False in run_single_or_list_callable(pre_condition_hook, dataset):
                    continue

            # Setting this queue as active
            if queue_size:
                all_queues_are_empty = False
            dataset.active_queue = queue_name
            dataset.nb_items_to_pull_from_queue = queue_size

            # [Hook] Pre-Queue
            if pre_queue_hook is not None:
                run_single_or_list_callable(pre_queue_hook, dataset)

            # Initializing iterator
            dataset.initialize(dataset.session)
            remaining_dequeues = dataset.nb_items_to_pull_from_queue

            # Processing queue
            while remaining_dequeues > 0 and (dataset.nb_items_to_pull_from_queue > 0 or not dataset.is_done):
                try:
                    # [Hook] Pre-Run
                    if pre_run_hook is not None:
                        run_single_or_list_callable(pre_run_hook, dataset)

                    results = dataset.session.run(outputs, feed_dict=placeholders, options=def_options)
                    dataset.model_results += [results]
                    remaining_dequeues -= results[0].shape[0]

                    # Status message
                    if with_status:
                        nb_items = results[0].shape[0]
                        LOGGER.info('[%s] Processed %d items. Remaining: %d/%d',
                                    queue_name, nb_items, remaining_dequeues, queue_size)

                    # [Hook] Post-Run
                    if post_run_hook is not None:
                        run_single_or_list_callable(post_run_hook, dataset)
                except (tf.errors.UnavailableError, tf.errors.AbortedError) as err:
                    LOGGER.warning('Received a fatal error on queue %s', queue_name)
                    raise err
                except tf.errors.OutOfRangeError:
                    pass

            # [Hook] Post-Queue
            if post_queue_hook is not None:
                run_single_or_list_callable(post_queue_hook, dataset)

            # Processing results in main thread
            if in_main_thread:
                process_results(dataset, in_main_thread=in_main_thread)

        # Sleeping if all queues were empty, or exiting if in main thread.
        if all_queues_are_empty:
            if in_main_thread:
                break
            time.sleep(0.1)

    # Exiting
    if not in_main_thread:
        dataset.thread_status['process_queues'] = CLOSED

def process_results(dataset, in_main_thread=False):
    """ This method will be launched in a separate thread and is in charge of settings the results on the calling
        future objects, so that the method that has put an object in the queue knows that its results are ready.
        :param dataset: The instantiated feedable dataset
        :param in_main_thread: Boolean that indicates that we are running in the main Python thread.
        :type dataset: QueueDataset
    """
    while not dataset.all_threads_closing:

        # Thread is paused
        if dataset.all_threads_paused and not in_main_thread:
            dataset.thread_status['process_results'] = PAUSED
            while dataset.all_threads_paused and not dataset.all_threads_closing:
                time.sleep(0.1)

        # Thread resumed / started
        if not in_main_thread:
            dataset.thread_status['process_results'] = RUNNING

        # No items in results queue, we can sleep, or exit if in main thread.
        if not dataset.model_results:
            if in_main_thread:
                break
            time.sleep(0.1)
            continue

        # Processing all items in the results queue
        # The first item of the results is always the request_id
        while dataset.model_results:
            results = dataset.model_results.pop(0)
            request_ids, output_results = results[0], results[1:]
            nb_results = request_ids.shape[0]

            # Determining if we have a queue with results
            # As opposed to an operation queue that doesn't return tensors
            # If the output results are a tuple (as opposed to a list for each item)
            # We return the tuple to each item in the batch (i.e. results are shared)
            if len(results) == 1:
                has_results = False
            elif isinstance(output_results[0], np.ndarray):
                has_results = bool(output_results[0].shape and output_results[0].shape[0] == nb_results)
            elif isinstance(output_results[0], list):
                has_results = len(output_results[0]) == nb_results
            elif isinstance(output_results, list) and output_results:
                output_results = [[result] * nb_results for result in output_results]
                has_results = True
            else:
                has_results = False

            # Processing each request id
            for result_ix in range(nb_results):
                this_request_id = request_ids[result_ix].decode('utf-8')
                this_result = [result[result_ix] for result in output_results] if has_results else None

                # Unknown request_id, skipping
                if this_request_id not in dataset.futures_ioloop:
                    continue

                # Otherwise marking the future as completed
                future, io_loop = dataset.futures_ioloop[this_request_id]
                del dataset.futures_ioloop[this_request_id]
                io_loop.asyncio_loop.call_soon_threadsafe(future.set_result, this_result)

    # Exiting
    if not in_main_thread:
        dataset.thread_status['process_results'] = CLOSED

# ----------------------------------------------------------------------------
# ----------                      DATASET                     ----------------
# ----------------------------------------------------------------------------
class QueueDataset(FeedableDataset):
    """ This object is responsible for generating entries to feed the model (using the tf.data.dataset API) """

    def __init__(self, batch_size, dataset_builder, cluster_config=None, no_iterator=False):
        """ Constructor
            :param batch_size: The size of a batch per tower
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
            :param no_iterator: Boolean flag that indicates to not create an iterator (it will be loaded from a ckpt)
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        super(QueueDataset, self).__init__(dataset_builder=dataset_builder,
                                           cluster_config=cluster_config)
        self.batch_size = batch_size
        self.no_iterator = no_iterator
        self.tf_dataset = None

        # Creating iterator with init ops
        self.init_op = None
        self.output_features = None                     # This represents iterator.get_next()

        # Feedable queues
        self.feedable_queues = {}
        self.active_queue = None
        self.nb_items_to_pull_from_queue = 0
        self.model_results = []
        self.futures_ioloop = {}                        # Contains tuple (future, original thread io_loop)
        self._dataset_is_done = False
        self.last_queue = ''

        # Threads
        self.threads = {}
        self.thread_status = {'process_queues': CLOSED, 'process_results': CLOSED}
        self.all_threads_paused = False
        self.all_threads_closing = True

        # Building the dataset
        self.build()

    @property
    def can_support_iterator(self):
        """ Determines if the dataset can support an iterator or if it is a remote (RPC) dataset """
        return True

    @property
    def is_done(self):
        """ Returns True if the end of file has been reached """
        return self._dataset_is_done

    def mark_as_done(self):
        """ Marks the dataset as having reached the end of the file"""
        self._dataset_is_done = True

    def build(self):
        """ Builds the TensorFlow dataset """
        from diplomacy_research.utils.tensorflow import tf, np_to_tf
        assert 'request_id' in self.proto_fields, 'You need to have a "request_id" field.'

        def feedable_generator():
            """ Generator that feeds data into the feedable_dataset
                When this functions exits/returns, a tf.errors.OutOfRangeError is triggered
            """
            while True:
                next_batch = self.get_next_feedable_batch()
                if next_batch is None:
                    self.mark_as_done()
                    break
                yield next_batch

        # Padding output shapes with None
        output_types = self.dataset_builder.output_types
        output_shapes = self.dataset_builder.output_shapes
        output_shapes = {key: [None] + list(shape) for key, shape in output_shapes.items()}

        # Building a list of generic default values from the output types and output shapes
        for feature_name, feature_shape in output_shapes.items():
            if output_types[feature_name] == np.object:
                self.default_features[feature_name] = bytes('', 'utf-8')
            elif isinstance(self.proto_fields[feature_name], VarProtoField):
                self.default_features[feature_name] = np.array([], dtype=output_types[feature_name])
            else:
                self.default_features[feature_name] = np.zeros(shape=feature_shape[1:],
                                                               dtype=output_types[feature_name])

        # Creates dataset
        tf_output_types = {key: np_to_tf(dtype) for key, dtype in output_types.items()}
        tf_output_shapes = {key: tf.TensorShape(shape) for key, shape in output_shapes.items()}
        self.tf_dataset = tf.data.Dataset.from_generator(feedable_generator,
                                                         output_types=tf_output_types,
                                                         output_shapes=tf_output_shapes)
        self.tf_dataset = self.tf_dataset.prefetch(1)

        # Creating iterator (with a new iterator_resource), unless specified otherwise
        if not self.no_iterator:
            self.create_iterator()

    def start(self, session):
        """ Starts the dataset
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        if self.is_started:
            LOGGER.error('Dataset was already started. Not re-starting it.')
            return
        self.session = session
        self.initialize(session)
        self.start_threads()
        self._is_started = True

    def restart(self, session):
        """ Restarts the threads using a (new) session object
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        if not self._is_started:
            self.start(session)
            return
        self.session = session
        self.start_threads()

    def run(self):
        """ Run process queues on the current thread - This is to use the session recoverability
            Note: threads needs to be paused before calling this method
        """
        if not self.all_threads_paused:
            LOGGER.warning('You must pause the threads before calling run(). Aborting.')
            return
        process_queues(self, in_main_thread=True)

    def start_threads(self):
        """ (Re)-starts the threads """
        if self.thread_status['process_queues'] != CLOSED or self.thread_status['process_results'] != CLOSED:
            self.stop_threads()
        self.all_threads_paused = False
        self.all_threads_closing = False
        self.threads['process_queues'] = Thread(target=process_queues, args=(self,), daemon=True)
        self.threads['process_results'] = Thread(target=process_results, args=(self,), daemon=True)
        self.threads['process_queues'].start()
        self.threads['process_results'].start()

    def stop_threads(self):
        """ Stops the threads and waits for termination """
        self.all_threads_paused = False
        self.all_threads_closing = True
        if self.threads['process_queues'] is not None:
            self.threads['process_queues'].join()
            self.threads['process_queues'] = None
            self.thread_status['process_queues'] = CLOSED
        if self.threads['process_results'] is not None:
            self.threads['process_results'].join()
            self.threads['process_results'] = None
            self.thread_status['process_results'] = CLOSED

    def pause_threads(self):
        """ Pauses all running threads and wait for them to be paused """
        self.all_threads_paused = True
        self.all_threads_closing = False
        start_time = int(time.time())
        current_time = start_time

        # Waiting for threads to pause
        while (self.thread_status['process_queues'] == RUNNING
               or self.thread_status['process_results'] == RUNNING
               or current_time - start_time > 60):
            time.sleep(1.)
            current_time = int(time.time())

    def resume_threads(self):
        """ Resumes all running threads and wait for them to be resumed """
        self.all_threads_paused = False
        self.all_threads_closing = False
        start_time = int(time.time())
        current_time = start_time

        # Waiting for threads to pause
        while (self.thread_status['process_queues'] == PAUSED
               or self.thread_status['process_results'] == PAUSED
               or current_time - start_time > 60):
            time.sleep(1.)
            current_time = int(time.time())

    def initialize(self, session):
        """ Initializes the dataset (and its iterator)
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        # We haven't created an iterator yet
        if self.iterator is None:
            return

        # Running init_op
        # If session is wrapped, executing it without hooks
        if hasattr(session, 'run_step_fn'):
            session.run_step_fn(lambda step_context: step_context.session.run(self.init_op))
        else:
            session.run(self.init_op)
        self._is_initialized = True
        self._dataset_is_done = False

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

        # Loading Tensorflow
        from diplomacy_research.utils.tensorflow import tf

        output_types = self.tf_dataset.output_types
        output_shapes = self.tf_dataset.output_shapes
        output_classes = {key: tf.Tensor for key in output_types}

        # Making sure iterator is on the right device/worker
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
                self.iterator = tf.data.Iterator.from_structure(output_types,
                                                                output_shapes,
                                                                shared_name=shared_name)
                self.output_features = self.iterator.get_next()

            # Generating init op
            self._is_initialized = False
            self.init_op = self.iterator.make_initializer(self.tf_dataset)

    def create_queue(self, queue_name, outputs, default_values=None, placeholders=None, *,
                     pre_condition=None, pre_queue=None, post_queue=None, pre_run=None, post_run=None,
                     with_status=False):
        """ Creates a new feedable queue
            :param queue_name: The name of the queue to add
            :param outputs: A list of outputs the model needs to run and return for this queue
            :param default_values: A dictionary of default values that will be added to new items in the queue
            :param placeholders: A feed dict of placeholders to automatically feed when processing the queue
            :param pre_condition: [Hook] Callable or list of callables. Args: (dataset)
                                  Is run before selecting the current queue. Must return True for all callables, otherw.
                                  the queue is skipped.
            :param pre_queue: [Hook] Callable or list of callables. Args: (dataset).
                              This hook is ran after the queue as been selected, but before any session.run.
            :param post_queue: [Hook] Callable or list of callables. Args: (dataset).
                               This hook is ran after all session.run for a given queue.
            :param pre_run: [Hook] Callable or list of callables. Args: (dataset)
                            This hook is ran before each session.run
            :param post_run: [Hook] Callable or list of callables. Args: (dataset)
                             This hook is ran after each session.run
            :param with_status: Boolean that indicates that we need to display a status message after session.run()
            :return: Nothing
        """
        if self.has_queue(queue_name):
            LOGGER.warning('The feedable queue "%s" has already been defined. Please choose a new queue name.',
                           queue_name)
            return

        # Creating queue
        self.feedable_queues[queue_name] = {'queue': Queue(),
                                            'outputs': [self.output_features['request_id']] + list(outputs),
                                            'default_values': default_values or {},
                                            'placeholders': placeholders or {},
                                            'with_status': with_status,
                                            'hooks': {
                                                'pre_condition': pre_condition,
                                                'pre_queue': pre_queue,
                                                'post_queue': post_queue,
                                                'pre_run': pre_run,
                                                'post_run': post_run
                                            }}

    def has_queue(self, queue_name):
        """ Determines if the feedable dataset already has a queue with the specified name """
        return queue_name in self.feedable_queues

    def get_results(self, queue_name, item, retry_on_failure=True, **kwargs):
        """ Computes the outputs of a name using item as input
            :param queue_name: The name of the queue where to put the item
            :param item: A dictionary with the fields required for that queue
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :param kwargs: Additional optional kwargs:
                - prefetch: Boolean that indicates to return a PrefetchedItem, otherwise returns a future.
            :return: Either:
                - if not prefetch, a Future that will be set with the results when they become available
                - if prefetch, a PrefetchedItem that can be put in the queue.
        """
        if not self.has_queue(queue_name):
            LOGGER.warning('The queue "%s" could not be found.', queue_name)
            return Future().set_result(None)
        if not isinstance(item, dict):
            LOGGER.warning('The item object passed to get_results must be a dictionary.')
            return Future().set_result(None)

        request_id = str(uuid.uuid4())
        item['request_id'] = bytes(request_id, 'utf-8')
        item = self.prepare_item(item)

        # Adding default values provided
        for default_key, default_val in self.feedable_queues[queue_name]['default_values'].items():
            if default_key not in item:
                item[default_key] = default_val

        # Adding generic default values (all zeros)
        for default_key, default_val in self.default_features.items():
            if default_key not in item:
                item[default_key] = default_val

        # Prefetching - We return a prefetched item
        if kwargs.get('prefetch', False):
            return PrefetchedItem(queue_name, item)

        # Otherwise, we put the item in the queue and return a future
        return self.put_item_in_queue(queue_name, item)

    def put_item_in_queue(self, queue_name, item):
        """ Puts an item in the queue, so that it can be processed by a TensorFlow session.
            :param queue_name: The name of the queue where to put the item
            :param item: A dictionary with the fields required for that queue.
            :return: A Future that will be set with the results when they become available
        """
        item_future = Future()
        request_id = item['request_id'].decode('utf-8')
        self.futures_ioloop[request_id] = (item_future, get_current_io_loop())
        self.feedable_queues[queue_name]['queue'].put_nowait(item)
        return item_future

    def get_next_feedable_batch(self):
        """ Returns the next feedable batch in the active queue, None otherwise
            The batch is a dictionary with feature names as key, and list of numpy arrays as values
        """
        if not self.active_queue \
                or not self.has_queue(self.active_queue) \
                or self.nb_items_to_pull_from_queue <= 0:
            return None

        # Building batch
        max_items_in_batch = min(self.nb_items_to_pull_from_queue, self.batch_size)
        batch = {key: [] for key in self.output_features.keys()}
        nb_items_in_batch = 0
        max_var_len = {key: 0 for key in self.proto_fields if isinstance(self.proto_fields[key], VarProtoField)}

        # Building batch - No padding yet
        while nb_items_in_batch < max_items_in_batch:
            next_batch_of_data = self.feedable_queues[self.active_queue]['queue'].get()
            for key in self.output_features.keys():
                batch[key] += [next_batch_of_data[key]]
                if key in max_var_len:
                    max_var_len[key] = max(max_var_len[key], len(next_batch_of_data[key]))
            nb_items_in_batch += 1

        # Padding all var len features to the maximum length in batch
        for padded_feature in max_var_len:
            for item_ix in range(nb_items_in_batch):
                current_len = len(batch[padded_feature][item_ix])
                max_len = max_var_len[padded_feature]
                if current_len < max_len:
                    batch[padded_feature][item_ix] = np.pad(batch[padded_feature][item_ix],
                                                            (0, max_len - current_len),
                                                            mode='constant')

        # Sending to generator
        self.nb_items_to_pull_from_queue -= nb_items_in_batch
        return batch

    def make_session_run_hook(self):
        """ Builds a SessionRunHook for the MonitoredTrainingSession object """
        from diplomacy_research.utils.tensorflow import QueueDatasetSessionRunHook
        return QueueDatasetSessionRunHook(self)

    def close(self):
        """ Stops iterating the dataset """
        self._is_closing = True
        self.tf_dataset = None
        self.stop_threads()
