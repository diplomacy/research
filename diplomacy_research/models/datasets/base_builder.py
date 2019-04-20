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
""" Base Dataset Builder
    - Abstract class responsible for generating the protocol buffers to be used by the model
"""
from abc import ABCMeta
from collections import namedtuple, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import logging
import math
import os
import pickle
from queue import Queue
import shutil
import sys
import traceback
import numpy as np
from tqdm import tqdm
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.model import pad_list, merge_dicts
from diplomacy_research.utils.proto import read_next_bytes, bytes_to_proto, proto_to_zlib, zlib_to_bytes
from diplomacy_research.settings import PROTO_DATASET_PATH, VALIDATION_SET_SPLIT, PHASES_COUNT_DATASET_PATH

# Constants
LOGGER = logging.getLogger(__name__)

class BaseField(namedtuple('ProtoField', ('shape',              # The shape w/o the batch dimension (e.g. [81, 35])
                                          'dtype')),            # The final dtype required
                metaclass=ABCMeta):
    """ Base Proto Field """

class FixedProtoField(BaseField):
    """ Represents a FixedLenFeature in tf.train.Example protocol buffer """

class VarProtoField(BaseField):
    """ Represents a VarLenFeature in tf.train.Example protocol buffer """

class SparseProtoField(BaseField):
    """ Represents a SparseFeature in a tf.train.Example protocol buffer """

class FixedLenFeature(namedtuple('FixedLenFeature', ('shape', 'dtype', 'default_value'))):
    """ FixedLenFeature (from tf.io.FixedLenFeature) """

class VarLenFeature(namedtuple('VarLenFeature', ['dtype'])):
    """ VarLenFeature (from tf.io.VarLenFeature) """


# ----------------------------------------------------------------------------
# ----------               THREADS METHODS                    ----------------
# ----------------------------------------------------------------------------
def handle_queues(task_ids, proto_callable, saved_game_bytes, is_validation_set):
    """ Handles the next phase """
    try:
        proto_results = proto_callable(saved_game_bytes, is_validation_set)
        results = []
        for phase_ix, task_id in enumerate(task_ids):
            if not proto_results[phase_ix]:
                results.append((task_id, None, 0, None))
                continue
            for power_name in proto_results[phase_ix]:
                message_lengths, proto_result = proto_results[phase_ix][power_name]
                results.append((task_id, power_name, message_lengths, proto_result))
        return results
    except Exception as exc:
        traceback.print_exc(file=sys.stdout)
        raise exc

# ----------------------------------------------------------------------------
# ----------               BUILDER METHODS                    ----------------
# ----------------------------------------------------------------------------
class BaseBuilder(metaclass=ABCMeta):
    """ This object is responsible for generating entries to feed the model (using the tf.data.dataset API) """
    # pylint: disable=too-many-instance-attributes

    # Paths as class properties
    training_dataset_path = None
    validation_dataset_path = None
    dataset_index_path = None

    def __init__(self, extra_proto_fields=None):
        """ Constructor
            :param extra_proto_fields: A dictionary of extra proto fields to use when building the iterator
        """
        self.features = {}
        self.padded_shapes = {}
        self.output_shapes = {}
        self.output_types = {}
        self.proto_fields = merge_dicts(self.get_proto_fields(), extra_proto_fields or {})

        # Parsing FixedLenFeature, VarLenFeature, and SparseFeature
        for feature_name, proto_field in self.proto_fields.items():

            # FixedLenFeature
            # Scalar are encoded directly (e.g. tf.int64 or tf.float32)
            # Arrays are encoded with tf.string and need to be decoded with tf.io.decode_raw
            if isinstance(proto_field, FixedProtoField):
                if not proto_field.shape:
                    feature = FixedLenFeature([], self.get_encoded_dtype(proto_field.dtype), None)
                else:
                    feature = FixedLenFeature([], np.object, None)
                output_shape = proto_field.shape
                padded_shape = proto_field.shape

            # VarLenFeature
            # Always encoded with tf.string and decoded with tf.io.decode_raw
            elif isinstance(proto_field, VarProtoField):
                feature = FixedLenFeature([], np.object, None)
                output_shape = [None]
                padded_shape = [None]

            # SparseFeature
            # Encoded with tf.string and decoded with tf.io.decode_raw
            elif isinstance(proto_field, SparseProtoField):
                # Adding an {}_indices
                self.features['%s_indices' % feature_name] = FixedLenFeature([], np.object, None)
                self.output_shapes['%s_indices' % feature_name] = [None]
                self.output_types['%s_indices' % feature_name] = np.int64
                self.padded_shapes['%s_indices' % feature_name] = [None]

                # Going to the next feature
                continue
            else:
                raise ValueError('Feature %s is not of a valid type.' % feature_name)

            # Storing results in respective dict
            self.features[feature_name] = feature
            self.output_shapes[feature_name] = output_shape
            self.output_types[feature_name] = proto_field.dtype or np.object
            self.padded_shapes[feature_name] = padded_shape

    @staticmethod
    def get_proto_fields():
        """ Returns the proto fields used by this dataset builder """
        raise NotImplementedError()

    @staticmethod
    def parse_sparse_fields(proto_fields):
        """ Creates a new proto fields by replacing the SparseProto with their components """
        new_proto_fields = deepcopy(proto_fields)
        for feature_name, proto_field in proto_fields.items():
            if not isinstance(proto_field, SparseProtoField):
                continue

            # Removing the field
            del new_proto_fields[feature_name]
            nb_dims = len(proto_field.shape)

            # Adding {}_indices
            new_proto_fields['%s_indices' % feature_name] = VarProtoField([None, nb_dims], dtype=np.int64)

        # Returning the new proto fields
        return new_proto_fields

    @staticmethod
    def get_encoded_dtype(numpy_dtype):
        """ Computes the most-efficient encoding for the given dtype """
        if numpy_dtype is None:
            return np.object
        if numpy_dtype in [np.bool, np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
            return np.int64
        if numpy_dtype in [np.float16, np.float32]:
            return np.float32
        raise RuntimeError('Invalid dtype specified: %s' % numpy_dtype)

    @staticmethod
    def get_feedable_item(*args, **kwargs):
        """ Computes and return a feedable item (to be fed into the feedable queue) """
        raise NotImplementedError()

    @staticmethod
    def get_request_id(saved_game_proto, phase_ix, power_name, is_validation_set):
        """ Returns the standardized request id for this game/phase """
        return '%s/%s/%d/%s/%s' % ('train' if not is_validation_set else 'valid',
                                   saved_game_proto.id,
                                   phase_ix,
                                   saved_game_proto.phases[phase_ix].name,
                                   power_name)

    @property
    def proto_generation_callable(self):
        """ Returns a callable required for proto files generation.
            e.g. return generate_proto(saved_game_bytes, is_validation_set)

            Note: Callable args are - saved_game_bytes: A `.proto.game.SavedGame` object from the dataset
                                    - phase_ix: The index of the phase we want to process
                                    - is_validation_set: Boolean that indicates if we are generating the validation set

            Note: Used bytes_to_proto from diplomacy_research.utils.proto to convert bytes to proto
                  The callable must return a list of tf.train.Example to put in the protocol buffer file
        """
        raise NotImplementedError()

    @property
    def sort_dataset_by_phase(self):
        """ Indicates that we want to have phase_ix 0 for all games, then phase_ix 1, ...
            Otherwise, we have all phases to game id 0, then all phases for game id 1, ...
        """
        return False

    @property
    def group_by_message_length(self):
        """ Indicates that we want to group phases with similar message length together to improve training speed.
            Otherwise, a batch might include either no messages at all, or messages of very different length
        """
        return False

    def parse_function(self, example_proto):
        """ Parses a stored protocol buffer """
        from diplomacy_research.utils.tensorflow import tf, np_to_tf

        # Converting features to tf.io.FixedLenFeature and tf.io.VarLenFeature
        tf_features = {}
        for feature_name in self.features:
            if isinstance(self.features[feature_name], FixedLenFeature):
                tf_features[feature_name] = tf.io.FixedLenFeature(**self.features[feature_name]._asdict())
            elif isinstance(self.features[feature_name], VarLenFeature):
                tf_features[feature_name] = tf.io.VarLenFeature(**self.features[feature_name]._asdict())
            else:
                raise RuntimeError('Unsupported feature type.')

        data = tf.parse_single_example(example_proto, tf_features)
        proto_fields = self.parse_sparse_fields(self.proto_fields)

        # Decoding from protocol buffer
        for feature_name, proto_field in proto_fields.items():
            current_dtype = np.object

            # Decoding tf.string
            if self.features[feature_name].dtype == np.object and proto_field.dtype is not None:
                encoded_dtype = np.uint8 if proto_field.dtype == np.bool else proto_field.dtype
                data[feature_name] = tf.io.decode_raw(data[feature_name], np_to_tf(encoded_dtype))
                current_dtype = encoded_dtype

            # Converting SparseTensor to Dense
            if isinstance(data[feature_name], tf.SparseTensor) and isinstance(proto_field, VarProtoField):
                data[feature_name] = tf.sparse.to_dense(data[feature_name])

            # Casting to final dtype
            if proto_field.dtype is not None and proto_field.dtype != current_dtype:
                data[feature_name] = tf.cast(data[feature_name], np_to_tf(proto_field.dtype))

            # Converting to final shape
            if isinstance(proto_field, FixedProtoField) and proto_field.shape:
                data[feature_name] = tf.reshape(data[feature_name], proto_field.shape)

        # Returning parsed data
        return data

    @staticmethod
    def build_example(features, proto_fields):
        """ Builds a tf.train.Example to store in the protocol buffer
            :param features: A dictionary of feature_name with their respective value to pad and convert
            :param proto_fields: The list of proto fields defined for this protocol buffer
            :return: A tf.train.Example
        """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.utils.tensorflow import generic_feature, strings_feature, ints_feature, floats_feature
        proto_fields = BaseBuilder.parse_sparse_fields(proto_fields)

        # 0) Adding missing fields
        for feature_name, proto_field in proto_fields.items():
            if feature_name not in features:
                if isinstance(proto_field, FixedProtoField):
                    features[feature_name] = np.zeros(shape=proto_field.shape, dtype=proto_field.dtype)
                elif isinstance(proto_field, VarProtoField):
                    features[feature_name] = []

        # 1) Padding all FixedLenFeatures
        for feature_name, proto_field in proto_fields.items():
            if isinstance(proto_field, FixedProtoField) and proto_field.shape:
                features[feature_name] = pad_list(features[feature_name], proto_field.shape)

        # 2) Converting features to proper proto features
        for feature_name, proto_field in proto_fields.items():

            # Scalar - Encoding directly
            if not proto_field.shape and proto_field.dtype:
                encoded_dtype = BaseBuilder.get_encoded_dtype(proto_field.dtype)
                if encoded_dtype == np.object:
                    features[feature_name] = strings_feature(features[feature_name])
                elif encoded_dtype == np.int64:
                    features[feature_name] = ints_feature(features[feature_name])
                elif encoded_dtype == np.float32:
                    features[feature_name] = floats_feature(features[feature_name])
                else:
                    RuntimeError('Invalid feature type')

            # Valid type according to tf.io.decode_raw
            elif proto_field.dtype in [np.float16, np.float32, np.float64, np.int32, np.uint16, np.uint8, np.int16,
                                       np.int8, np.int64, None]:
                features[feature_name] = generic_feature(features[feature_name], proto_field.dtype)

            # Bool - Casting to uint8
            elif proto_field.dtype == np.bool:
                features[feature_name] = generic_feature(features[feature_name], np.uint8)

            # Otherwise, unsupported dtype
            else:
                raise RuntimeError('Dtype %s is not supported by tf.io.decode_raw' % proto_field.dtype.name)

        # 3) Building a tf.train.Example
        return tf.train.Example(features=tf.train.Features(feature=features))

    def generate_proto_files(self, validation_perc=VALIDATION_SET_SPLIT):
        """ Generates train and validation protocol buffer files
            :param validation_perc: The percentage of the dataset to use to generate the validation dataset.
        """
        # pylint: disable=too-many-nested-blocks,too-many-statements,too-many-branches
        from diplomacy_research.utils.tensorflow import tf

        # The protocol buffers files have already been generated
        if os.path.exists(self.training_dataset_path) \
                and os.path.exists(self.validation_dataset_path) \
                and os.path.exists(self.dataset_index_path):
            return

        # Deleting the files if they exist
        shutil.rmtree(self.training_dataset_path, ignore_errors=True)
        shutil.rmtree(self.validation_dataset_path, ignore_errors=True)
        shutil.rmtree(self.dataset_index_path, ignore_errors=True)

        # Making sure the proto_generation_callable is callable
        proto_callable = self.proto_generation_callable
        assert callable(proto_callable), "The proto_generable_callable must be a callable function"

        # Loading index
        dataset_index = {}

        # Splitting games into training and validation
        with open(PHASES_COUNT_DATASET_PATH, 'rb') as file:
            game_ids = list(pickle.load(file).keys())
        nb_valid = int(validation_perc * len(game_ids))
        set_valid_ids = set(sorted(game_ids)[-nb_valid:])
        train_ids = list(sorted([game_id for game_id in game_ids if game_id not in set_valid_ids]))
        valid_ids = list(sorted(set_valid_ids))

        # Building a list of all task phases
        train_task_phases, valid_task_phases = [], []

        # Case 1) - We just build a list of game_ids with all their phase_ids (sorted by game id asc)
        if not self.sort_dataset_by_phase:
            with open(PHASES_COUNT_DATASET_PATH, 'rb') as phase_count_dataset:
                phase_count_dataset = pickle.load(phase_count_dataset)
                for game_id in train_ids:
                    for phase_ix in range(phase_count_dataset.get(game_id, 0)):
                        train_task_phases += [(phase_ix, game_id)]
                for game_id in valid_ids:
                    for phase_ix in range(phase_count_dataset.get(game_id, 0)):
                        valid_task_phases += [(phase_ix, game_id)]

        # Case 2) - We build 10 groups sorted by game_id asc, and for each group we put all phase_ix 0, then
        #           all phase_ix 1, ...
        #         - We need to split into groups so that the model can learn a mix of game beginning and endings
        #           otherwise, the model will only see beginnings and after 1 day only endings
        else:
            with open(PHASES_COUNT_DATASET_PATH, 'rb') as phase_count_dataset:
                phase_count_dataset = pickle.load(phase_count_dataset)
                nb_groups = 10
                nb_items_per_group = math.ceil(len(train_ids) / nb_groups)

                # Training
                for group_ix in range(nb_groups):
                    group_game_ids = train_ids[group_ix * nb_items_per_group:(group_ix + 1) * nb_items_per_group]
                    group_task_phases = []
                    for game_id in group_game_ids:
                        for phase_ix in range(phase_count_dataset.get(game_id, 0)):
                            group_task_phases += [(phase_ix, game_id)]
                    train_task_phases += list(sorted(group_task_phases))

                # Validation
                for game_id in valid_ids:
                    for phase_ix in range(phase_count_dataset.get(game_id, 0)):
                        valid_task_phases += [(phase_ix, game_id)]
                valid_task_phases = list(sorted(valid_task_phases))

        # Grouping tasks by buckets
        # buckets_pending contains a set of pending task ids in each bucket
        # buckets_keys contains a list of tuples so we can group items with similar message length together
        task_to_bucket = {}                                             # {task_id: bucket_id}
        train_buckets_pending, valid_buckets_pending = [], []           # [bucket_id: set()]
        train_buckets_keys, valid_buckets_keys = [], []                 # [bucket_id: (msg_len, task_id, power_name)]
        if self.group_by_message_length:
            nb_valid_buckets = int(VALIDATION_SET_SPLIT * 100)
            nb_train_buckets = 100

            # Train buckets
            task_id = 1
            nb_items_per_bucket = math.ceil(len(train_task_phases) / nb_train_buckets)
            for bucket_ix in range(nb_train_buckets):
                items = train_task_phases[bucket_ix * nb_items_per_bucket:(bucket_ix + 1) * nb_items_per_bucket]
                nb_items = len(items)
                train_buckets_pending.append(set())
                train_buckets_keys.append([])

                for _ in range(nb_items):
                    train_buckets_pending[bucket_ix].add(task_id)
                    task_to_bucket[task_id] = bucket_ix
                    task_id += 1

            # Valid buckets
            task_id = -1
            nb_items_per_bucket = math.ceil(len(valid_task_phases) / nb_valid_buckets)
            for bucket_ix in range(nb_valid_buckets):
                items = valid_task_phases[bucket_ix * nb_items_per_bucket:(bucket_ix + 1) * nb_items_per_bucket]
                nb_items = len(items)
                valid_buckets_pending.append(set())
                valid_buckets_keys.append([])

                for _ in range(nb_items):
                    valid_buckets_pending[bucket_ix].add(task_id)
                    task_to_bucket[task_id] = bucket_ix
                    task_id -= 1

        # Building a dictionary of {game_id: {phase_ix: task_id}}
        # Train tasks have a id >= 0, valid tasks have an id < 0
        task_id = 1
        task_id_per_game = {}
        for phase_ix, game_id in train_task_phases:
            task_id_per_game.setdefault(game_id, {})[phase_ix] = task_id
            task_id += 1

        task_id = -1
        for phase_ix, game_id in valid_task_phases:
            task_id_per_game.setdefault(game_id, {})[phase_ix] = task_id
            task_id -= 1

        # Building a dictionary of pending items, so we can write them to disk in the correct order
        nb_train_tasks = len(train_task_phases)
        nb_valid_tasks = len(valid_task_phases)
        pending_train_tasks = OrderedDict({task_id: None for task_id in range(1, nb_train_tasks + 1)})
        pending_valid_tasks = OrderedDict({task_id: None for task_id in range(1, nb_valid_tasks + 1)})

        # Computing batch_size, progress bar and creating a pool of processes
        batch_size = 5120
        progress_bar = tqdm(total=nb_train_tasks + nb_valid_tasks)
        process_pool = ProcessPoolExecutor()
        futures = set()

        # Creating buffer to write all protos to disk at once
        train_buffer, valid_buffer = Queue(), Queue()

        # Opening the proto file to read games
        proto_dataset = open(PROTO_DATASET_PATH, 'rb')
        nb_items_being_processed = 0

        # Creating training and validation dataset
        for training_mode in ['train', 'valid']:
            next_key = 1
            current_bucket = 0

            if training_mode == 'train':
                pending_tasks = pending_train_tasks
                buckets_pending = train_buckets_pending
                buckets_keys = train_buckets_keys
                buffer = train_buffer
                max_next_key = nb_train_tasks + 1
            else:
                pending_tasks = pending_valid_tasks
                buckets_pending = valid_buckets_pending
                buckets_keys = valid_buckets_keys
                buffer = valid_buffer
                max_next_key = nb_valid_tasks + 1
            dataset_index['size_{}_dataset'.format(training_mode)] = 0

            # Processing with a queue to avoid high memory usage
            while pending_tasks:

                # Filling queues
                while batch_size > nb_items_being_processed:
                    saved_game_bytes = read_next_bytes(proto_dataset)
                    if saved_game_bytes is None:
                        break
                    saved_game_proto = bytes_to_proto(saved_game_bytes, SavedGameProto)
                    game_id = saved_game_proto.id
                    if game_id not in task_id_per_game:
                        continue
                    nb_phases = len(saved_game_proto.phases)
                    task_ids = [task_id_per_game[game_id][phase_ix] for phase_ix in range(nb_phases)]
                    futures.add((tuple(task_ids), process_pool.submit(handle_queues,
                                                                      task_ids,
                                                                      proto_callable,
                                                                      saved_game_bytes,
                                                                      task_ids[0] < 0)))
                    nb_items_being_processed += nb_phases

                # Processing results
                for expected_task_ids, future in list(futures):
                    if not future.done():
                        continue
                    results = future.result()
                    current_task_ids = set()

                    # Storing in compressed format in memory
                    for task_id, power_name, message_lengths, proto_result in results:
                        current_task_ids.add(task_id)

                        if proto_result is not None:
                            zlib_result = proto_to_zlib(proto_result)
                            if task_id > 0:
                                if pending_train_tasks[abs(task_id)] is None:
                                    pending_train_tasks[abs(task_id)] = {}
                                pending_train_tasks[abs(task_id)][power_name] = zlib_result
                            else:
                                if pending_valid_tasks[abs(task_id)] is None:
                                    pending_valid_tasks[abs(task_id)] = {}
                                pending_valid_tasks[abs(task_id)][power_name] = zlib_result

                            if self.group_by_message_length:
                                task_bucket_id = task_to_bucket[task_id]
                                if task_id > 0:
                                    train_buckets_keys[task_bucket_id].append((message_lengths, task_id, power_name))
                                else:
                                    valid_buckets_keys[task_bucket_id].append((message_lengths, task_id, power_name))

                        # No results - Marking task id as done
                        elif task_id > 0 and pending_train_tasks[abs(task_id)] is None:
                            del pending_train_tasks[abs(task_id)]
                        elif task_id < 0 and pending_valid_tasks[abs(task_id)] is None:
                            del pending_valid_tasks[abs(task_id)]

                    # Missing some task ids
                    if set(expected_task_ids) != current_task_ids:
                        LOGGER.warning('Missing tasks ids. Got %s - Expected: %s', current_task_ids, expected_task_ids)
                        current_task_ids = expected_task_ids

                    # Marking tasks as completed
                    nb_items_being_processed -= len(expected_task_ids)
                    progress_bar.update(len(current_task_ids))

                    # Marking items as not pending in buckets
                    if self.group_by_message_length:
                        for task_id in current_task_ids:
                            task_bucket_id = task_to_bucket[task_id]
                            if task_id > 0:
                                train_buckets_pending[task_bucket_id].remove(task_id)
                            else:
                                valid_buckets_pending[task_bucket_id].remove(task_id)

                    # Deleting futures to release memory
                    futures.remove((expected_task_ids, future))
                    del future

                # Writing to disk
                while True:
                    if self.group_by_message_length:

                        # Done all buckets
                        if current_bucket >= len(buckets_pending):
                            break

                        # Still waiting for tasks in the current bucket
                        if buckets_pending[current_bucket]:
                            break

                        # Bucket was empty - We can look at next bucket
                        if not buckets_keys[current_bucket]:
                            current_bucket += 1
                            break

                        # Sorting items in bucket before writing them in buffer
                        items_in_bucket = list(sorted(buckets_keys[current_bucket]))
                        for _, task_id, power_name in items_in_bucket:
                            zlib_result = pending_tasks[abs(task_id)][power_name]
                            buffer.put(zlib_result)
                            dataset_index['size_{}_dataset'.format(training_mode)] += 1
                            del pending_tasks[abs(task_id)][power_name]
                            if not pending_tasks[abs(task_id)]:
                                del pending_tasks[abs(task_id)]
                        current_bucket += 1
                        del items_in_bucket
                        break

                    # Writing to buffer in the same order as they are received
                    if next_key >= max_next_key:
                        break
                    if next_key not in pending_tasks:
                        next_key += 1
                        continue
                    if pending_tasks[next_key] is None:
                        break
                    zlib_results = pending_tasks.pop(next_key)
                    for zlib_result in zlib_results.values():
                        buffer.put(zlib_result)
                        dataset_index['size_{}_dataset'.format(training_mode)] += 1
                    next_key += 1
                    del zlib_results

        # Stopping pool, and progress bar
        process_pool.shutdown(wait=True)
        progress_bar.close()
        proto_dataset.close()

        # Storing protos to disk
        LOGGER.info('Writing protos to disk...')
        progress_bar = tqdm(total=train_buffer.qsize() + valid_buffer.qsize())
        options = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)

        with tf.io.TFRecordWriter(self.training_dataset_path, options=options) as dataset_writer:
            while not train_buffer.empty():
                zlib_result = train_buffer.get()
                dataset_writer.write(zlib_to_bytes(zlib_result))
                progress_bar.update(1)

        with tf.io.TFRecordWriter(self.validation_dataset_path, options=options) as dataset_writer:
            while not valid_buffer.empty():
                zlib_result = valid_buffer.get()
                dataset_writer.write(zlib_to_bytes(zlib_result))
                progress_bar.update(1)

        with open(self.dataset_index_path, 'wb') as dataset_index_file:
            pickle.dump(dataset_index, dataset_index_file, pickle.HIGHEST_PROTOCOL)

        # Closing
        progress_bar.close()
