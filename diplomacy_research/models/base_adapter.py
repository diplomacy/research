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
""" Base Adapter
    - This module allows a player to interact with a model using a standardized interface
"""
from abc import ABCMeta
import logging
import time
from diplomacy_research.models.datasets.queue_dataset import QueueDataset
from diplomacy_research.utils.cluster import is_ioloop_running

# Note: The following is imported in `load_from_checkpoint()` to avoid cyclical imports
# from diplomacy_research.utils.checkpoint import load_frozen_graph, load_graph_from_ckpt

# Constants
LOGGER = logging.getLogger(__name__)

class BaseAdapter(metaclass=ABCMeta):
    """ Allows the evaluation of a policy adapter from a TensorFlow graph and session """

    def __init__(self, feedable_dataset, graph=None, session=None):
        """ Initializer
            :param feedable_dataset: The feedable dataset to use (must be initiated under the graph provided)
            :param graph: The graph object that contains the policy model to evaluate
            :param session: The session to use to interact with the graph
            :type feedable_dataset: diplomacy_research.models.datasets.feedable_dataset.FeedableDataset
            :type graph: tensorflow.python.framework.ops.Graph
            :type session: tensorflow.python.client.session.Session
        """
        self.graph = graph
        self.session = session
        self.feedable_dataset = feedable_dataset
        self.iterator = self.feedable_dataset.iterator
        self.features = {}
        self.placeholders = {}
        self.outputs = {}

        # Checking if the IOLoop is started
        if not is_ioloop_running():
            LOGGER.error('This object requires a running IO-Loop. Please start it before instantiating this object.')
            raise RuntimeError('IO Loop has not been started.')

        # Loading features, outputs, placeholders
        if graph is not None:
            self._load_features_placeholders()

        # Initializes the adapter
        if self.session:
            self.initialize(self.session)

        # Creating queues
        self.create_queues()

    @staticmethod
    def get_signature():
        """ Returns the signature of all the possible calls using this adapter
            Format: { method_signature_name: {'placeholders': {name: (value, numpy_dtype)},
                                              'outputs': [output_name, output_name] } }
            e.g. {'policy_evaluate': {'placeholders': {'decoder_type': ([SAMPLE_DECODER], np.uint8)},
                                      'outputs: ['selected_tokens', 'log_probs', 'draw_prob']}}
        """
        raise NotImplementedError()

    def _load_features_placeholders(self):
        """ Loads the features, outputs, and placeholders nodes from the model """
        from diplomacy_research.utils.tensorflow import tf
        graph = self.graph or tf.get_default_graph()
        collection_keys = graph.get_all_collection_keys()

        for key in collection_keys:
            # If list, getting first element
            key_value = graph.get_collection(key)
            if isinstance(key_value, list) and key_value:
                key_value = key_value[0]

            # Setting in self.
            if key.startswith('feature'):
                self.features[key.replace('feature_', '')] = key_value
            elif key.startswith('placeholder'):
                self.placeholders[key.replace('placeholder_', '')] = key_value
            else:
                self.outputs[key] = key_value

    @property
    def is_trainable(self):
        """ Returns a boolean that indicates if the policy model can be trained """
        return len([key for key in self.outputs if 'is_trainable' in key]) > 0

    def initialize(self, session):
        """ Initialize the adapter (init global vars and the dataset)
            :type session: tensorflow.python.client.session.Session
        """
        if not self.feedable_dataset.can_support_iterator or not self.iterator:
            return

        from diplomacy_research.utils.tensorflow import tf
        assert session, 'You must pass a session to initialize the adapter'
        assert isinstance(self.feedable_dataset, QueueDataset), 'The dataset must be a QueueDataset'
        self.session = session

        # Initializes uninit global vars
        graph = self.graph or tf.get_default_graph()
        if not graph.finalized:
            with graph.as_default():
                var_to_initialize = tf.global_variables() + tf.local_variables()
                is_initialized = self.session.run([tf.is_variable_initialized(var) for var in var_to_initialize])
                not_initialized_vars = [var for (var, is_init) in zip(var_to_initialize, is_initialized) if not is_init]
                if not_initialized_vars:
                    LOGGER.info('Initialized %d variables.', len(not_initialized_vars))
                    self.session.run(tf.variables_initializer(not_initialized_vars))

        # Initializing the dataset to use the feedable model
        if not self.feedable_dataset.is_started and self.session:
            self.feedable_dataset.start(self.session)
        elif not self.feedable_dataset.is_initialized and self.session:
            self.feedable_dataset.initialize(self.session)

    def load_from_checkpoint(self, checkpoint_path):
        """ Loads the variable from the checkpoint into the current graph
            :param checkpoint_path: Either 1) Path to a checkpoint (e.g. /path/model.ckpt-XXX) or
                                           2) Path to a frozen graph (e.g. /path/frozen.pb)
            :return: Nothing
        """
        assert self.feedable_dataset.can_support_iterator, 'The dataset must be able to support an iterator'
        assert isinstance(self.feedable_dataset, QueueDataset), 'The dataset must be a QueueDataset'

        # ---- <Import> ----
        # Loaded here to avoid cyclical imports
        from diplomacy_research.utils.checkpoint import load_frozen_graph, load_graph_from_ckpt  # pylint: disable=wrong-import-position
        # ---- </Import> ----

        # Loading graph from disk
        if checkpoint_path[-3:] == '.pb':
            load_frozen_graph(checkpoint_path, graph=self.graph, session=self.session)
        else:
            load_graph_from_ckpt(checkpoint_path, graph=self.graph, session=self.session)

        # Loading features, outputs, placeholders
        self._load_features_placeholders()

        # Making sure we have an iterator resource
        iterator_resource = [self.outputs[key] for key in self.outputs if 'iterator_resource' in key]
        if not iterator_resource:
            LOGGER.error('An "iterator_resource" key must be defined in checkpoints for models to be resumable.')
            raise RuntimeError('"iterator_resource" not present.')

        # Creating new iterator with the iterator_resource
        iterator_resource = iterator_resource[0]
        self.feedable_dataset.create_iterator(iterator_resource, features=self.features)
        self.feedable_dataset.initialize(self.session)

        # Rebuilding queues
        self.create_queues()

    def create_queues(self):
        """ Generates queues to feed data directly in the dataset in feedable mode """
        # The dataset must be a QueueDataset
        if not isinstance(self.feedable_dataset, QueueDataset):
            return

        # We haven't loaded a model yet (probably going to load a frozen checkpoint instead)
        # We can't build queue yets because the graph is not built.
        if not self.outputs or not self.features:
            return

        # Building queues
        signature = self.get_signature()
        for method_name in signature:
            placeholders = signature[method_name].get('placeholders', {})
            outputs = signature[method_name]['outputs']

            # Queue already created
            if self.feedable_dataset.has_queue(method_name):
                LOGGER.warning('Queue %s has already been created.', method_name)
                continue

            # Output not available
            missing_outputs = [output_name for output_name in outputs if output_name not in self.outputs]
            if missing_outputs:
                LOGGER.warning('Unable to create queue "%s" - Missing outputs: %s', method_name, missing_outputs)
                continue

            # Placeholder not available
            missing_pholders = [pholder_name for pholder_name in placeholders if pholder_name not in self.placeholders]
            if missing_pholders:
                LOGGER.warning('Unable to create queue "%s" - Missing placeholders: %s', method_name, missing_pholders)
                continue

            # Building queue
            self.feedable_dataset.create_queue(method_name,
                                               outputs=[self.outputs[output_name] for output_name in outputs],
                                               placeholders={self.placeholders[ph_name]: placeholders[ph_name][0]
                                                             for ph_name in placeholders},
                                               post_queue=lambda _: time.sleep(0.10))        # To collect many batches
