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
""" Checkpoint
    - Contains functions to load a graph from checkpoint
    - Contains functions to freeze a graph, and to load the frozen graph
    - Contains functions to create a SavedModel
"""
from datetime import datetime
import glob
import logging
import os
from pydoc import locate
import shutil
import sys
import time
from pytz import timezone
from diplomacy_research.models.datasets.base_builder import BaseBuilder

# Constants
LOGGER = logging.getLogger(__name__)
MODEL_PATHS = {'/token_based/v': 'diplomacy_research/models/policy/token_based',
               '/order_based/v': 'diplomacy_research/models/policy/order_based'}

def load_graph_from_ckpt(checkpoint_path, meta_graph_path=None, graph=None, session=None):
    """ Builds a graph and a session from a specific checkpoint
        This loads the model into a new graph, and doesn't affect the default graph

        :param checkpoint_path: The checkpoint path. Can be a checkpoint directory, or a specific checkpoint in
                                that directory
        :param meta_graph_path: (Optional) The path to the saved meta graph (.meta). Will be detected automatically
                                if not provided
        :param graph: The graph object were to load the model. A new graph will be created if not provided.
        :param session: The session object to use to load the model. A new session will be created if not provided.
        :return: The graph and the session object where the checkpoint was loaded.
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    from diplomacy_research.utils.tensorflow import tf

    dir_path, filename = os.path.split(checkpoint_path)

    # checkpoint_path is a directory - Loading latest checkpoint in directory
    if os.path.isdir(checkpoint_path):
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if meta_graph_path is None:
            meta_graph_path = max(glob.iglob(os.path.join(checkpoint_path, '*.meta')), key=os.path.getctime)

    # checkpoint_path is a checkpoint file - Loading latest checkpoint in directory
    elif filename == 'checkpoint':
        checkpoint = tf.train.latest_checkpoint(dir_path, 'checkpoint')
        if meta_graph_path is None:
            meta_graph_path = max(glob.iglob(os.path.join(dir_path, '*.meta')), key=os.path.getctime)

    # Loading a specific checkpoint
    else:
        # Removing extension
        if len(filename.split('.')) > 2:
            checkpoint_path = os.path.join(dir_path, '.'.join(filename.split('.')[:2]))
        checkpoint = checkpoint_path
        if meta_graph_path is None:
            if os.path.exists('{}.meta'.format(checkpoint_path)):
                meta_graph_path = '{}.meta'.format(checkpoint_path)
            else:
                meta_graph_path = max(glob.iglob(os.path.join(dir_path, '*.meta')), key=os.path.getctime)

    # Loading the checkpoint in the graph
    graph = tf.Graph() if graph is None else graph
    with graph.as_default():
        session = tf.Session(graph=graph) if session is None else session
        saver = tf.train.import_meta_graph(meta_graph_path)
        saver.restore(session, checkpoint)

    # Returning graph and session
    return graph, session

def freeze_graph(frozen_dir, version_id, graph, session, history_saver=None):
    """ Freezes a graph and saves a checkpoint and the frozen graph to disk

        :param frozen_dir: The path where to save the checkpoint and frozen graph
        :param version_id: Integer. The version id to append to the filename.
        :param graph: The graph object to save
        :param session: The session associated with the graph
        :param history_saver: Optional. The saver to use to save historical checkpoints, otherwise no checkpoints will
                              be created and the graph will only be frozen.
        :return: Nothing
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    checkpoint_path = os.path.join(frozen_dir, 'checkpoint-v%09d' % version_id)
    frozen_path = os.path.join(frozen_dir, 'frozen_graph-v%09d.pb' % version_id)

    # Making sure frozen directory exists
    if not os.path.exists(frozen_dir):
        os.makedirs(frozen_dir, exist_ok=True)

    # Creating a checkpoint
    if history_saver is not None:
        with graph.as_default():
            history_saver.save(session, checkpoint_path)

    # Freezing graph
    convert_ckpt_to_frozen_graph(checkpoint_path=None,
                                 frozen_graph_path=frozen_path,
                                 graph=graph,
                                 session=session)

def build_saved_model(saved_model_dir, version_id, signature, proto_fields, graph, session, history_saver=None):
    """ Builds a SavedModel and a checkpoint from the graph

        :param saved_model_dir: The path where to save the checkpoint and SavedModel
        :param version_id: Integer. The version_id of the SavedModel to save.
        :param signature: The output of adapter.get_signature() - signature of all the possible calls
        :param proto_fields: A dictionary of features name with their proto field description
        :param graph: The graph object to save
        :param session: The session associated with the graph
        :param history_saver: Optional. The saver to use to save historical checkpoints, otherwise no checkpoints will
                              be created and the graph will only be converted to SavedModel.
        :return: Nothing
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    checkpoint_path = os.path.join(saved_model_dir, 'checkpoint-v%09d' % version_id)
    saved_model_path = os.path.join(saved_model_dir, '%09d' % version_id)

    # Making sure saved model directory exists
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    # Creating a checkpoint
    if history_saver is not None:
        with graph.as_default():
            history_saver.save(session, checkpoint_path)

    # Building saved model
    convert_ckpt_to_saved_model(checkpoint_path=None,
                                saved_model_path=saved_model_path,
                                signature=signature,
                                proto_fields=proto_fields,
                                graph=graph,
                                session=session)

def convert_ckpt_to_frozen_graph(checkpoint_path, frozen_graph_path, meta_graph_path=None, graph=None, session=None):
    """ Converts a checkpoint to a frozen (meta) graph with fixed weights for faster inference
        :param checkpoint_path: The path to the checkpoint file (can be a directly, or a checkpoint file)
        :param frozen_graph_path: The path where to saved the frozen_graph_path
        :param meta_graph_path: Optional. The path of the meta_graph. Will be detected automatically if not provided.
        :param graph: The graph object were to load the model. A new graph will be created if not provided.
        :param session: The session object to use to load the model. A new session will be created if not provided.
        :return: The graph and the session object used to load the checkpoint.
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    from diplomacy_research.utils.tensorflow import tf, graph_util, variables

    # Loading the checkpoint from disk
    if graph is None or session is None:
        graph, session = load_graph_from_ckpt(checkpoint_path,
                                              meta_graph_path=meta_graph_path,
                                              graph=graph,
                                              session=session)
    # Converting graph to constant
    input_graph_def = graph.as_graph_def()
    output_keys = [k for k in graph.get_all_collection_keys() if ('variable' not in k.lower()
                                                                  and '_step' not in k
                                                                  and '_op' not in k
                                                                  and '_context' not in k
                                                                  and not k.startswith('_')
                                                                  and not k.endswith('_ta')
                                                                  and 'summaries' not in k
                                                                  and 'is_trainable' not in k)]

    # Making sure we are saving an iterator, otherwise the model will not be usable
    if not [key for key in output_keys if 'iterator_resource' in key]:
        LOGGER.error('Trying to freeze a model without an "iterator_resource" key. Model will not be usable. Aborting')
        raise RuntimeError('Missing "iterator_resource" to freeze model.')

    # Finding output nodes and extra tags
    extra_tags = {}
    output_nodes = []
    for key in output_keys:
        nodes_in_collection = graph.get_collection(key)
        for node in nodes_in_collection:
            if isinstance(node, variables.PartitionedVariable):
                output_nodes += [var.name for var in node._get_variable_list()]                                         # pylint: disable=protected-access
            elif hasattr(node, 'name'):
                output_nodes += [node.name]
            else:
                extra_tags.setdefault(key, [])
                extra_tags[key] += [node]

    # Freezing graph
    output_graph_def = graph_util.convert_variables_to_constants(session,
                                                                 input_graph_def,
                                                                 [node.split(':')[0] for node in output_nodes])

    # Storing date/time, original filename, and launch args
    created_date = datetime.fromtimestamp(time.time(), timezone('America/Montreal'))
    extra_tags['tag/created_date'] = [created_date.strftime("%Y-%m-%d %H:%M:%S %Z")]
    extra_tags['tag/filename'] = [frozen_graph_path.split('/')[-1]]
    extra_tags['tag/launch_cmd'] = [' '.join(sys.argv)]

    # Importing in a new graph
    output_graph = tf.Graph()
    with output_graph.as_default():
        tf.import_graph_def(output_graph_def)

    # Transferring collections
    collection_keys = graph.get_all_collection_keys()
    for key in collection_keys:
        if 'variable' in key.lower() or '_op' in key:
            continue
        nodes = graph.get_collection(key)
        for node in nodes:
            if hasattr(node, 'name'):
                try:
                    tensor_name = 'import/{}{}'.format(node.name, ':0' if ':' not in node.name else '')
                    tensor_node = output_graph.get_tensor_by_name(tensor_name)
                    output_graph.add_to_collection(key, tensor_node)
                except KeyError:
                    pass

    # Adding extra tags
    for key in extra_tags:
        for value in extra_tags[key]:
            output_graph.add_to_collection(key, value)

    # Saving the frozen graph to disk
    with output_graph.as_default():
        tf.train.export_meta_graph(frozen_graph_path,
                                   graph_def=output_graph.as_graph_def(),
                                   clear_devices=True)

    # Returning
    return graph, session

def convert_ckpt_to_saved_model(checkpoint_path, saved_model_path, signature, proto_fields, meta_graph_path=None,
                                graph=None, session=None):
    """ Converts a checkpoint to a SavedModel with fixed weights for faster inference
        :param checkpoint_path: The path to the checkpoint file (can be a directly, or a checkpoint file)
        :param saved_model_path: The path where to saved the SavedModel
        :param signature: The output of adapter.get_signature() - signature of all the possible calls
        :param proto_fields: A dictionary of features name with their proto field description
        :param meta_graph_path: Optional. The path of the meta_graph. Will be detected automatically if not provided.
        :param graph: The graph object were to load the model. A new graph will be created if not provided.
        :param session: The session object to use to load the model. A new session will be created if not provided.
        :return: The graph and the session object used to load the checkpoint.
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    from diplomacy_research.utils.tensorflow import tf, graph_util, build_tensor_info, saved_model_builder, \
        signature_def_utils, tag_constants, variables, PREDICT_METHOD_NAME

    # Loading the checkpoint from disk
    if graph is None or session is None:
        graph, session = load_graph_from_ckpt(checkpoint_path,
                                              meta_graph_path=meta_graph_path,
                                              graph=graph,
                                              session=session)
    # Converting graph to constant
    input_graph_def = graph.as_graph_def()
    output_keys = [k for k in graph.get_all_collection_keys() if ('variable' not in k.lower()
                                                                  and '_step' not in k
                                                                  and '_op' not in k
                                                                  and '_context' not in k
                                                                  and not k.startswith('_')
                                                                  and not k.endswith('_ta')
                                                                  and 'summaries' not in k
                                                                  and 'is_trainable' not in k)]

    # Finding output nodes and extra tags
    extra_tags = {}
    output_nodes = []
    for key in output_keys:
        nodes_in_collection = graph.get_collection(key)
        for node in nodes_in_collection:
            if isinstance(node, variables.PartitionedVariable):
                output_nodes += [var.name for var in node._get_variable_list()]                                         # pylint: disable=protected-access
            elif hasattr(node, 'name'):
                output_nodes += [node.name]
            else:
                extra_tags.setdefault(key, [])
                extra_tags[key] += [node]

    # Converting graph to constant
    output_graph_def = graph_util.convert_variables_to_constants(session,
                                                                 input_graph_def,
                                                                 [node.split(':')[0] for node in output_nodes])

    # Storing date/time, original filename, and launch args
    created_date = datetime.fromtimestamp(time.time(), timezone('America/Montreal'))
    extra_tags['tag/created_date'] = [created_date.strftime("%Y-%m-%d %H:%M:%S %Z")]
    extra_tags['tag/filename'] = [saved_model_path.split('/')[-1]]
    extra_tags['tag/launch_cmd'] = [' '.join(sys.argv)]

    # Importing in a new graph
    output_graph = tf.Graph()
    with output_graph.as_default():
        tf.import_graph_def(output_graph_def)

    # Finding placeholders, features, and outputs
    features, placeholders, outputs = {}, {}, {}
    collection_keys = graph.get_all_collection_keys()
    for key in collection_keys:
        node = graph.get_collection(key)
        if isinstance(node, list) and node:                                     # If list, getting first element
            node = node[0]
        if key.startswith('feature'):
            features[key.replace('feature_', '')] = output_graph.get_tensor_by_name('import/' + node.name)
        elif key.startswith('placeholder'):
            placeholders[key.replace('placeholder_', '')] = output_graph.get_tensor_by_name('import/' + node.name)
        elif hasattr(node, 'name'):
            try:
                outputs[key] = output_graph.get_tensor_by_name('import/' + node.name)
            except (KeyError, ValueError):
                continue

    # Adding extra tags
    for key in extra_tags:
        for value in extra_tags[key]:
            output_graph.add_to_collection(key, value)

    # Converting sparse fields
    proto_fields = BaseBuilder.parse_sparse_fields(proto_fields)

    # Building signature
    signature_def = {}
    for method_name in signature:
        method_placeholders = signature.get(method_name).get('placeholders', {})
        method_outputs = signature.get(method_name).get('outputs', [])

        # Skipping method if we are missing some outputs
        missing_outputs = [output_name for output_name in method_outputs if output_name not in outputs]
        if missing_outputs:
            LOGGER.warning('Unable to build method %s using the provided signature.', method_name)
            continue

        signature_inputs = {feature_name: build_tensor_info(features[feature_name]) for feature_name in features
                            if feature_name in proto_fields}
        for ph_name in method_placeholders:
            signature_inputs[ph_name] = build_tensor_info(placeholders[ph_name])
        signature_outputs = {'%03d_%s' % (output_id, output_name): build_tensor_info(outputs[output_name])
                             for output_id, output_name in enumerate(method_outputs)}

        signature_def[method_name] = signature_def_utils.build_signature_def(inputs=signature_inputs,
                                                                             outputs=signature_outputs,
                                                                             method_name=PREDICT_METHOD_NAME)

    # Saving to disk
    with output_graph.as_default():
        temp_model_path = '/'.join(saved_model_path.split('/')[:-1] + ['__%s__' % saved_model_path.split('/')[-1]])

        # Deleting from disk to avoid 'Directory already exists'
        shutil.rmtree(saved_model_path, ignore_errors=True)
        shutil.rmtree(temp_model_path, ignore_errors=True)

        # Saving to a temporary path, to make sure the serving does not try to load the version before it is ready
        builder = saved_model_builder.SavedModelBuilder(temp_model_path)
        builder.add_meta_graph_and_variables(session,
                                             [tag_constants.SERVING],
                                             signature_def_map=signature_def,
                                             clear_devices=True)
        builder.save()

        # Renaming to the correct path
        shutil.move(temp_model_path, saved_model_path)

    # Returning
    return graph, session

def load_frozen_graph(frozen_graph_path, graph=None, session=None):
    """ Loads a frozen graph from disk
        :param frozen_graph_path: The path where the frozen graph is located
        :param graph: The graph object were to load the model. A new graph will be created if not provided.
        :param session: The session object to use to load the model. A new session will be created if not provided.
        :return: The graph and the session object used to load the frozen graph.
        :type graph: tensorflow.python.framework.ops.Graph
        :type session: tensorflow.python.client.session.Session
    """
    from diplomacy_research.utils.tensorflow import tf, tf_logging

    # Making sure the path exists
    if not os.path.exists(frozen_graph_path):
        LOGGER.error('The frozen graph %s does not exist.', frozen_graph_path)
        raise FileNotFoundError()

    # Load the frozen (meta) graph into a TF graph
    graph = tf.Graph() if graph is None else graph
    with graph.as_default():
        session = tf.Session(graph=graph) if session is None else session

        # Not showing "Saver not created because there are no variables in the graph to restore" messages
        tf_logging.set_verbosity('ERROR')
        tf.train.import_meta_graph(frozen_graph_path, clear_devices=True)
        tf_logging.set_verbosity('INFO')

    return graph, session

def get_constructors_from_frozen_graph(model_path):
    """ Finds the BaseDatasetBuilder and the PolicyAdapter from a frozen checkpoint from disk
        :param model_path: The path to the frozen checkpoint
        :return: The BaseDatasetBuilder and the PolicyAdapter object linked to this model, otherwise (None, None)
    """
    base_dir = None
    model_name = None

    # Making sure model exists
    if not os.path.exists(model_path):
        LOGGER.info('Unable to find model at %s', model_path)
        return None, None

    # Loading graph
    graph, _ = load_frozen_graph(model_path)

    # Detecting model type
    tags = sorted([key for key in graph.get_all_collection_keys() if 'tag/' in key])
    for tag_name in tags:
        if 'tag' in tag_name:
            for search_key in MODEL_PATHS:
                if search_key in tag_name:
                    base_dir = MODEL_PATHS[search_key]
                    model_name = tag_name

    # No base dir found
    if base_dir is None or model_name is None:
        LOGGER.info('Unable to detect the model used to generate this file.')
        return None, None

    # Loading the base dataset builder, and the policy adapter
    base_dataset_builder = locate('%s.BaseDatasetBuilder' % base_dir.replace('/', '.'))
    policy_adapter = locate('%s.PolicyAdapter' % base_dir.replace('/', '.'))

    # Returning
    return base_dataset_builder, policy_adapter
