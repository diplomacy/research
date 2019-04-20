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
""" gRPC Dataset
    - Class responsible for retrieving outputs by sending gRPC request over the network
"""
from collections import namedtuple
import logging
import time

import numpy as np
from tornado import gen

from diplomacy_research.models.datasets.base_builder import VarProtoField
from diplomacy_research.models.datasets.feedable_dataset import FeedableDataset
from diplomacy_research.proto.tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from diplomacy_research.proto.tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest
from diplomacy_research.proto.tensorflow_serving.apis.predict_pb2 import PredictRequest
from diplomacy_research.proto.tensorflow_serving.config import model_server_config_pb2
from diplomacy_research.utils.cluster import is_port_opened, wrap_grpc_call
from diplomacy_research.utils.proto import make_tensor_proto, make_ndarray

# Constants
LOGGER = logging.getLogger(__name__)
UNKNOWN, START, LOADING, AVAILABLE, UNLOADING, END = 0, 10, 20, 30, 40, 50

class ModelConfig(namedtuple('ModelConfig', ('name',                    # The name of the model
                                             'base_path',               # The path to look for servable on disk
                                             'version_policy'))):       # e.g {'latest': 1} or {'specific': [1, 2, 3]}
    """ A ModelConfiguration to send to the TF Serving server """

class GRPCDataset(FeedableDataset):
    """ This object is responsible for retrieving model outputs over the network using remote procedure calls """

    def __init__(self, hostname, port, model_name, signature, dataset_builder, cluster_config=None, connect_timeout=300,
                 timeout=30, nb_retries=100):
        """ Constructor
            :param hostname: The hostname of the TensorFlow Serving server
            :param port: The port used by the TensorFlow Serving server.
            :param model_name: The name of the model being served by the TensorFlow Serving server.
            :param signature: The output of adapter.get_signature() - signature of all the possible calls
            :param dataset_builder: An instance of `BaseBuilder` containing the proto-fields and generation methods
            :param cluster_config: Optional. If set, the cluster configuration will be used for distributed training.
            :param connect_timeout: The timeout to try to connect to the TF serving server.
            :param timeout: The timeout (in seconds) to wait for a request
            :param nb_retries: The number of retries in case of grpc.RpcError before giving up.
            :type dataset_builder: diplomacy_research.models.datasets.base_builder.BaseBuilder
            :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        """
        # pylint: disable=too-many-arguments
        super(GRPCDataset, self).__init__(dataset_builder=dataset_builder,
                                          cluster_config=cluster_config)
        self.hostname = hostname
        self.port = port
        self.model_name = model_name
        self.signature = signature
        self.timeout = timeout
        self.nb_retries = nb_retries
        self.channel = None
        self.predict_stub = None
        self.config_stub = None

        # Waiting for port to be opened
        for retry_ix in range(connect_timeout):
            time.sleep(1)
            if is_port_opened(self.port, self.hostname):
                break
            if (retry_ix + 1) % 10 == 0:
                LOGGER.info('Trying to connect to TF Serving at %s:%d. Attempt %d / %d',
                            self.hostname, self.port, retry_ix + 1, connect_timeout)
        else:
            raise RuntimeError('Unable to connect to %s:%d. Max attempts reached.' % (self.hostname, self.port))

        # Building the dataset
        self.build()

    @property
    def can_support_iterator(self):
        """ Determines if the dataset can support an iterator or if it is a remote (RPC) dataset """
        return False

    def build(self):
        """ Builds the channel and the stub """
        import grpc
        from diplomacy_research.proto.tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

        assert 'request_id' in self.proto_fields, 'You need to have a "request_id" field.'
        assert is_port_opened(self.port, self.hostname), 'Unable to connect to %s:%d.' % (self.hostname, self.port)

        # Creating insecure channel with corresponding stubs
        self.channel = grpc.insecure_channel('%s:%d' % (self.hostname, self.port))
        self.predict_stub = PredictionServiceStub(self.channel)

        # Padding output shapes with None
        output_types = self.dataset_builder.output_types
        output_shapes = self.dataset_builder.output_shapes
        output_shapes = {key: [None] + list(shape) for key, shape in output_shapes.items()}

        # Building a list of generic default values from the output types and output shapes
        for feature_name, feature_shape in output_shapes.items():
            if output_types[feature_name] == np.object:
                self.default_features[feature_name] = make_tensor_proto(bytes('', 'utf-8'), dtype=np.object, shape=[1])
            elif isinstance(self.proto_fields[feature_name], VarProtoField):
                self.default_features[feature_name] = make_tensor_proto([],
                                                                        dtype=output_types[feature_name],
                                                                        shape=[1, 0])
            else:
                self.default_features[feature_name] = make_tensor_proto(0,
                                                                        dtype=output_types[feature_name],
                                                                        shape=[1] + feature_shape[1:])

    def start(self, session):
        """ Starts the dataset
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        self._is_started = True

    def initialize(self, session):
        """ Initializes the dataset (and its iterator)
            :param session: The TensorFlow session to use.
            :type session: tensorflow.python.client.session.Session
        """
        self._is_initialized = True

    def has_queue(self, queue_name):
        """ Determines if the feedable dataset already has a queue with the specified name """
        return queue_name in self.signature

    @staticmethod
    def set_config(hostname, port, new_config, timeout=30):
        """ Modify the server configuration by sending a ReloadConfigRequest
            :param hostname: The hostname of the TensorFlow Serving server
            :param port: The port used by the TensorFlow Serving server.
            :param new_config: A ModelConfig named tuple or a list of ModelConfig named tuple
            :param timeout: The timeout (in seconds) to wait for a request
            :return: Boolean that indicates if the request was successful
        """
        import grpc
        from diplomacy_research.proto.tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub

        config_list = []
        if not isinstance(new_config, list):
            new_config = [new_config]

        # Parsing the config
        for config in new_config:
            assert isinstance(config, ModelConfig), 'The configuration must be an instance of ModelConfig'
            this_config = {'name': config.name,
                           'base_path': config.base_path,
                           'model_platform': 'tensorflow'}
            if isinstance(config.version_policy, dict):
                if 'latest' in config.version_policy:
                    num_versions = int(config.version_policy.get('latest'))
                    this_config['model_version_policy'] = {'latest': {'num_versions': num_versions}}
                elif 'specific' in config.version_policy:
                    specific_versions = config.version_policy.get('specific')
                    if not isinstance(specific_versions, list):
                        specific_versions = [specific_versions]
                    this_config['model_version_policy'] = {'specific': {'versions': specific_versions}}
            config_list += [model_server_config_pb2.ModelConfig(**this_config)]

        # Building the request
        request = ReloadConfigRequest(config={'model_config_list': {'config': config_list}})

        # Sending the request
        try:
            channel = grpc.insecure_channel('%s:%d' % (hostname, port))
            config_stub = ModelServiceStub(channel)
            config_stub.HandleReloadConfigRequest(request, timeout=timeout)
            channel.close()
            return True
        except grpc.RpcError as grpc_error:
            LOGGER.error(grpc_error)
            return False

    @staticmethod
    def wait_for_model_to_load(hostname, port, model_name, version_id=None, timeout=10):
        """ Waits for the model to be loaded on the server
            :param hostname: The hostname of the TensorFlow Serving server
            :param port: The port used by the TensorFlow Serving server.
            :param model_name: The model name that needs to be loaded
            :param version_id: The version that should be loaded
            :param timeout: The maximum number of seconds to wait. Logs an error if reached.
        """
        import grpc
        from diplomacy_research.proto.tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub

        request = GetModelStatusRequest()
        request.model_spec.name = model_name                                                # pylint: disable=no-member
        if version_id:
            request.model_spec.version.value = version_id                                   # pylint: disable=no-member

        # Sending the request
        starting_time = time.time()
        channel = grpc.insecure_channel('%s:%d' % (hostname, port))
        config_stub = ModelServiceStub(channel)

        # Looping until we get the LOADED status
        while time.time() < starting_time + timeout:
            try:
                response = config_stub.GetModelStatus(request, timeout=10)
            except grpc.RpcError as grpc_error:
                if grpc_error.code() != grpc.StatusCode.NOT_FOUND:                          # pylint: disable=no-member
                    LOGGER.error(grpc_error)
                    channel.close()
                    return False

                # Not found - Sleeping and retrying
                time.sleep(1.)
                continue

            # Getting the status code of the latest version
            nb_versions = len(response.model_version_status)
            latest_version = -1
            status = UNKNOWN
            error_code = 0
            error_message = ""

            for response_ix in range(nb_versions):
                if response.model_version_status[response_ix].version > latest_version:
                    latest_version = response.model_version_status[response_ix].version
                    status = response.model_version_status[response_ix].state
                    error_code = response.model_version_status[response_ix].status.error_code
                    error_message = response.model_version_status[response_ix].status.error_message

            # Model has an error. Logging it and returning.
            if error_code:
                LOGGER.error('Model has an error. %s', error_message)
                channel.close()
                return False

            # Model is loaded and available
            if status == AVAILABLE:
                channel.close()
                return True

            # Otherwise, waiting and retrying
            time.sleep(0.1)

        # Model was still not available
        LOGGER.warning('The model "%s" is still not available after %d seconds.', model_name, timeout)
        channel.close()
        return False

    @gen.coroutine
    def get_results(self, queue_name, item, retry_on_failure=True, **kwargs):
        """ Computes the outputs of a name using item as inout
            :param queue_name: The name of the queue where to put the item (or model_name/queue_name)
            :param item: A dictionary with the fields required for that queue
            :param retry_on_failure: Boolean that indicates to retry querying from the model if an error is encountered.
            :return: A tornado.concurrent.Future that will be set with the results when they become available
        """
        import grpc

        if not self.has_queue(queue_name):
            LOGGER.warning('The method "%s" could not be found.', queue_name)
            return None
        if not isinstance(item, dict):
            LOGGER.warning('The item object passed to get_results must be a dictionary.')
            return None

        # Trying to infer the model_name from the queue_name
        model_name = self.model_name
        if '/' in queue_name:
            model_name, queue_name = queue_name.split('/')

        # Preparing the item
        item['request_id'] = bytes('', 'utf-8')
        item = self.prepare_item(item)

        # Building the request
        request = PredictRequest()
        request.model_spec.name = model_name                                                # pylint: disable=no-member
        request.model_spec.signature_name = queue_name                                      # pylint: disable=no-member

        # Setting the keys in items
        # Adding a leading batch dimension, so that TF Serving can batch items properly
        # i.e. (dim_1, dim_2) --> (1, dim_1, dim_2)
        for key in item:
            batched_item_key = item[key][None, ...] if isinstance(item[key], np.ndarray) else [item[key]]
            request.inputs[key].CopyFrom(make_tensor_proto(batched_item_key,                # pylint: disable=no-member
                                                           dtype=self.dataset_builder.output_types[key]))

        # Setting the placeholders defined in the signature
        placeholders = self.signature[queue_name].get('placeholders', {})
        for ph_name in placeholders:
            ph_value, ph_dtype = placeholders[ph_name]
            request.inputs[ph_name].CopyFrom(make_tensor_proto(ph_value, dtype=ph_dtype))   # pylint: disable=no-member

        # Adding generic default values (all zeros)
        for default_key, default_val in self.default_features.items():
            if default_key not in item:
                request.inputs[default_key].CopyFrom(default_val)                           # pylint: disable=no-member

        # Sending the request and processing the response
        # Trying for a maximum of self.nb_retries
        for attempt_ix in range(self.nb_retries):
            try:
                grpc_response = yield wrap_grpc_call(self.predict_stub.Predict.future(request, timeout=self.timeout))
                return [make_ndarray(grpc_response.outputs[key])[0, ...] for key in sorted(grpc_response.outputs)]
            except grpc.RpcError as grpc_error:
                if not retry_on_failure:
                    raise grpc_error
                yield gen.sleep(30)
            if (attempt_ix + 1) % 10 == 0:
                LOGGER.warning('Unable to get results. Attempt %d/%d', attempt_ix + 1, self.nb_retries)

        # Raising fatal exception
        raise RuntimeError('Unable to get results after %d attempts. Aborting' % self.nb_retries)

    def close(self):
        """ Closes the underlying channel """
        super(GRPCDataset, self).close()
        if self.channel:
            self.channel.close()
            self.channel = None
