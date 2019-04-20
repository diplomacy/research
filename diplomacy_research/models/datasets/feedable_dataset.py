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
""" Feedable Dataset
    - Abstract class responsible for feeding data inside a model
"""
from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from diplomacy_research.models.datasets.base_builder import BaseBuilder, VarProtoField, FixedProtoField
from diplomacy_research.utils.model import pad_list

# Constants
LOGGER = logging.getLogger(__name__)

class FeedableDataset(metaclass=ABCMeta):
    """ This object is a generic feedable dataset """

    def __init__(self, dataset_builder, cluster_config=None):
        """ Constructor
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        self.dataset_builder = dataset_builder
        self.cluster_config = cluster_config
        self.proto_fields = BaseBuilder.parse_sparse_fields(dataset_builder.proto_fields)
        self.default_features = {}                      # Will be used as default if features are missing

        self.session = None
        self.iterator = None
        self._is_started = False
        self._is_initialized = False
        self._is_closing = False

    def __del__(self):
        """ Destructor """
        self.close()

    @property
    def can_support_iterator(self):
        """ Determines if the dataset can support an iterator or if it is a remote (RPC) dataset """
        raise NotImplementedError()

    @property
    def is_started(self):
        """ Determines if the dataset has been started """
        return self._is_started

    @property
    def is_initialized(self):
        """ Determines if the iterator is initialized """
        return self._is_initialized

    @property
    def is_closing(self):
        """ Determines if the dataset is closing """
        return self._is_closing

    @abstractmethod
    def build(self):
        """ Builds the dataset """
        raise NotImplementedError()

    @abstractmethod
    def start(self, session):
        """ Starts the dataset
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, session):
        """ Initializes the dataset (and its iterator)
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        raise NotImplementedError()

    def get_feedable_item(self, *args, **kwargs):
        """ Calls the dataset_builder get_feedable_item """
        return self.dataset_builder.get_feedable_item(*args, **kwargs)

    def prepare_item(self, item):
        """ Makes sure the item respects the required protofields and casts/pads the item accordingly """
        # Checking all features in items, padding them and converting them to the right dtype
        for feature in item:
            assert feature in self.proto_fields, 'Feature %s is not in proto fields.' % feature
            proto_field = self.proto_fields[feature]

            # Converting sets to lists
            if isinstance(item[feature], set):
                item[feature] = list(item[feature])

            # Var Len - Converting and flattening
            # Scalar - Just converting
            # Fixed Len - Padding and converting
            if proto_field.dtype is None:
                continue
            elif isinstance(proto_field, VarProtoField):
                item[feature] = np.array(item[feature], proto_field.dtype).flatten()
            elif not proto_field.shape:
                item[feature] = np.array(item[feature], proto_field.dtype)
            elif isinstance(proto_field, FixedProtoField):
                item[feature] = np.array(pad_list(item[feature], proto_field.shape), proto_field.dtype)

        # Returning item
        return item

    @abstractmethod
    def get_results(self, queue_name, item, retry_on_failure=True, **kwargs):
        """ Computes the outputs of a name using item as inout
            :param queue_name: The name of the queue where to put the item
            :param item: A dictionary with the fields required for that queue
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: A tornado.concurrent.Future that will be set with the results when they become available
        """
        raise NotImplementedError()

    @staticmethod
    def make_session_run_hook():
        """ Builds a SessionRunHook for the MonitoredTrainingSession object """

    def close(self):
        """ Stops the dataset """
        self._is_closing = True
