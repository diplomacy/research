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
""" General utility functions for models """
import argparse
from collections import OrderedDict
import logging
import os
from pydoc import locate
import sys
import numpy as np
import yaml
from diplomacy_research.settings import ROOT_DIR, WORKING_DIR, DEFAULT_SAVE_DIR, GIT_COMMIT_HASH, GIT_IS_DIRTY

# Constants
LOGGER = logging.getLogger(__name__)
PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class TimeoutException(Exception):
    """ Custom exception class that indicates code ran for too long """

def pad_list(list_to_pad, target_length):
    """ Pads a list or numpy to a specified length/shape. The padding is done with zeros. If the source is greater
        than the target dimension, it will be cropped to that dimension and a warning will be displayed.

        :param list_to_pad: The list to be padded (either a list object or a 2D numpy array)
        :param target_length: The length of the list (if a list object is provided) or the target shape (if numpy array)
        :return: The padded list / array
    """
    # N-Dimensional Array
    if isinstance(list_to_pad, np.ndarray) or (isinstance(target_length, (list, tuple)) and len(target_length) > 1):
        target_length = tuple(target_length)
        if not isinstance(list_to_pad, np.ndarray):
            list_to_pad = np.array(list_to_pad)
        assert isinstance(target_length, tuple)
        assert len(list_to_pad.shape) == len(target_length)

        n_dims = len(list_to_pad.shape)
        min_dims = tuple(min(target_length[i], list_to_pad.shape[i]) for i in range(n_dims))

        if [1 for dim_i in range(n_dims) if list_to_pad.shape[dim_i] > target_length[dim_i]]:
            LOGGER.warning('Trimmed array of shape %s to shape %s.', list_to_pad.shape, target_length)

        padded_array = np.zeros(target_length, dtype=list_to_pad.dtype)
        sliced_dims = tuple(slice(None, dim_i) for dim_i in min_dims)           # [:dim_1, :dim_2, :dim_3, ...]
        padded_array[sliced_dims] = list_to_pad[sliced_dims]
        return_var = padded_array

    # List
    else:
        target_length = target_length[0] if isinstance(target_length, (list, tuple)) else target_length
        flat_list = np.array(list_to_pad).flatten().tolist()
        if len(flat_list) > target_length:
            LOGGER.warning('Trimmed list of %d items to required length of %d.', len(flat_list), target_length)
            flat_list = flat_list[:target_length]
        return_var = flat_list + [0] * (target_length - len(flat_list))

    return return_var

def int_prod(list_ints):
    """ Computes the product of each element in the input and converts it to an integer
        :param list_ints: A list of integers
        :return: An integer representing the product of the list of integers
    """
    return int(np.prod(list_ints))

def apply_temperature(probability_dist, temperature):
    """ Applies temperature to force moves towards random play (high temperature) or greedy play (low temperature)
        :param probability_dist: The action probability distribution
        :param temperature: Moves can be pushed towards totally random (high temperature) or towards greedy play
                            (low temperature)
        :return: The normalized, temperature-adjusted distribution
    """
    # Detecting NaNs
    if np.isnan(probability_dist).any():
        raise RuntimeError('A NaN has been detected. Values: %s' % list(probability_dist))

    # Applying temperature
    log_prob = np.log(np.maximum(probability_dist, 1e-8)) * 1. / max(temperature, 1e-8)
    log_prob = log_prob - max(log_prob)
    probs = np.exp(log_prob)

    # Re-normalizing so probs sum to 1.
    return probs / max(1e-8, sum(probs))

def logsumexp(input_array, axis=None, constant_array=None, keepdims=False):
    """ Compute the log of the sum of exponentials of input elements.
        :param input_array: Input array (array_like)
        :param axis: None or int or tuple of ints, optional. Axis or axes over which the sum is taken
        :param constant_array: array-like, optional Scaling factor for exp(`a`) must be of the same shape as `a`.
        :param keepdims: Optional. If true, the reduced axes will be left with size 1.
        :return: The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically more stable way.
                 If return_sign is True, also returns an array of floats matching res with (+1, 0, or -1)
    """
    # Source: https://github.com/scipy/scipy/blob/v1.1.0/scipy/special/_logsumexp.py
    if constant_array is not None:
        input_array, constant_array = np.broadcast_arrays(input_array, constant_array)
        if np.any(constant_array == 0):
            input_array = input_array + 0.                              # promote to at least float
            input_array[constant_array == 0] = -np.inf

    # Computes the maximum over an axis
    a_max = np.amax(input_array, axis=axis, keepdims=True)

    # Masking
    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0                                                                                  # pylint: disable=invalid-unary-operand-type
    elif not np.isfinite(a_max):
        a_max = 0

    # Computing exp in a stable way
    if constant_array is not None:
        constant_array = np.asarray(constant_array)
        tmp = constant_array * np.exp(input_array - a_max)
    else:
        tmp = np.exp(input_array - a_max)

    # Summing, then computing log
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        sum_array = np.sum(tmp, axis=axis, keepdims=keepdims)
        out_array = np.log(np.maximum(sum_array, 1e-8))

    # Squeezing and returning
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out_array += a_max
    return out_array

def assert_normalized(probs, tolerance=0.01, raise_error=True):
    """ Normalize probs so they sum to one. The probability should already so to ~1, but not exactly 1 due to rounding.
        :param probs: A list of probabilities
        :param tolerance: The maximum tolerance (i.e. should sum to 1. +- tolerance)
        :param raise_error: Raises an error if the sum is outside the tolerated range.
        :return: The normalized probabilities that sum exactly to 1.
    """
    sum_probs = sum(probs)
    if sum_probs > (1. + tolerance) or sum_probs < (1. - tolerance):
        LOGGER.error('The probabilities do not sum to 1. Got a sum of %.8f.', sum_probs)
        if raise_error:
            raise ValueError('Probabilities do not sum to 1.')
    return [prob / sum_probs for prob in probs]

def merge_dicts(*dict_args):
    """ Merge a list of dictionaries """
    return_var = {}
    for dictionary in dict_args:
        return_var.update(dictionary)
    return return_var

def strip_keys(src_dict, keys_to_strip):
    """ Removes certain keys from a dict """
    return {key: value for key, value in src_dict.items() if key not in keys_to_strip}

def merge_complex_dicts(src_dict, new_dict):
    """ Merges two complex dictionnaries
        - Not existent key are added
        - Lists are appended to
        - Dictionaries are merged
        :param src_dict: The dict where the keys will be merged into
        :param new_dict: The dict where the keys will be merged from
        :return: The src_dict with its keys updated
    """
    for key, value in new_dict.items():
        if key not in src_dict:
            src_dict[key] = value
        elif isinstance(value, (list, tuple)):
            if not isinstance(src_dict[key], list):
                raise ValueError('Trying to update a non-list item with a list')
            src_dict[key] += list(value)
        elif isinstance(value, (dict, OrderedDict)):
            if not isinstance(src_dict[key], (dict, OrderedDict)):
                raise ValueError('Trying to update a non-dict item with a dict')
            src_dict[key].update(value)
        elif isinstance(value, (np.ndarray, np.generic)):
            if not isinstance(src_dict[key], (np.ndarray, np.generic)):
                raise ValueError('Trying to update a non-numpy array with a numpy array')
            if src_dict[key].shape != value.shape or src_dict[key].dtype != value.dtype:
                raise ValueError('The shape and/or dtype of the src and dest numpy arrays do not match')
            src_dict[key] = value
        else:
            raise ValueError('Unsupported value type %s' % value.__class__)
    return src_dict

def zip_same(*seqs):
    """ Zips sequences if they are the same length """
    len_seq_0 = len(seqs[0])
    assert all(len(seq) == len_seq_0 for seq in seqs[1:])
    return zip(*seqs)

def find_save_dir(arg_name='save_dir'):
    """ Finds the save_dir by parsing directly the command line flags
        :param arg_name: The name of the arg (i.e. --`arg_name`) to use to detect the save_dir
        :return: The save_dir (or the default save_dir if not found)
    """
    save_dir = DEFAULT_SAVE_DIR

    # Parsing command-line arguments
    for arg_ix, arg in enumerate(sys.argv[:-1]):
        if arg == '--%s' % arg_name:
            save_dir = sys.argv[arg_ix + 1]
            break

    # Returning
    return save_dir

def find_model_type(arg_name='model_type', model_type=None):
    """ Finds the base_dir and base_import_path of a given model_type
        :param arg_name: The name of the arg (i.e. --`arg_name`) to use to detect the model type to load
        :param model_type: The model type to use. Otherwise checks for '--model_type' in the arguments.
        :return: The model type base_dir, and the model type base_import_path
    """
    base_dir, base_import_path = '', ''

    # Trying to load from confile file first
    config_data = load_config_file()
    if arg_name in config_data:
        model_type = config_data[arg_name]

    # Determining model type
    if not model_type:
        for arg_ix, arg  in enumerate(sys.argv[:-1]):
            if arg == '--%s' % arg_name:
                model_type = sys.argv[arg_ix + 1].lower()
                break
        else:
            config_data = load_config_file()
            if arg_name in config_data:
                model_type = config_data[arg_name]
            else:
                raise RuntimeError('Unable to detect model_type. Make sure --%s is set.' % arg_name)

    # Determining base_dir
    for folder in os.listdir(os.path.join(ROOT_DIR, 'models')):
        target_path = os.path.join(ROOT_DIR, 'models', folder, model_type)
        if os.path.exists(target_path):
            base_dir = target_path
            break
    else:
        raise RuntimeError('Unable to find a base folder with the name "%s"' % model_type)

    # Determining base import path
    for folder_ix, folder in enumerate(base_dir.split('/')):
        if folder == 'diplomacy_research':
            base_import_path = '.'.join(base_dir.split('/')[folder_ix:])
            break

    # Returning
    return base_dir, base_import_path

def load_config_file():
    """ Tries to determine if there is a config file on disk and loads data from it """
    save_dir = find_save_dir()
    config = 'config.yaml'
    config_section = None
    config_data = {}
    config_file_path = None

    # Parse command line args
    for arg_ix, arg in enumerate(sys.argv[:-1]):
        if arg == '--config':
            config = sys.argv[arg_ix + 1]
        if arg == '--config_section':
            config_section = sys.argv[arg_ix + 1]

    # Finding the config full path
    if os.path.exists(os.path.join(save_dir, config)):
        config_file_path = os.path.join(save_dir, config)
    elif os.path.exists(os.path.join(WORKING_DIR, config)):
        config_file_path = os.path.join(WORKING_DIR, config)
    elif os.path.exists(config):
        config_file_path = config

    # Otherwise looking recursively from save_dir
    if not config_file_path:
        parts = save_dir.split('/')
        while len(parts) > 1:
            if os.path.exists('/'.join(parts + [config])):
                config_file_path = '/'.join(parts + [config])
                break
            parts = parts[:-1]

    # Loading YAML
    if config_file_path:
        LOGGER.info('Using configuration file: %s', config_file_path)
        with open(config_file_path, 'r') as stream:
            try:
                config_data = yaml.load(stream, Loader=yaml.SafeLoader)

                # If we have a config section, we take the root keys
                # and we update them with the values in the section
                if config_section:
                    config_data.update(config_data[config_section])
            except yaml.YAMLError:
                pass
    return config_data

def load_dynamically_from_model_id(classes, arg_name='model_id', base_dir=None, model_id=-1, on_error='raise'):
    """ Dynamically load classes constructor based on the '--model_id' parameter
        :param classes: A list of classes to load from the folder with the model id.
        :param arg_name: The name of the arg (i.e. --`arg_name`) to use to detect the id to load
        :param base_dir: The directory that contains the models (and also the train.py file)
        :param model_id: The model id to use. Otherwise checks for '--model_id' in the arguments.
        :param on_error: The behaviour when an error is detected (either 'raise' or 'ignore')
        :return: A list of class constructor for each class name in classes.
                 or a list of None is an error occurred and on_error == 'ignore'

        Note: if '--model_id 2' is passed on the command line and the classes are ['Class1', 'Class2'].
              The following classes will be returned:   - base_dir.v002_something.Class1
                                                        - base_dir.v002_something.Class2
    """
    assert on_error in ('raise', 'ignore'), 'Expected "on_error" to be either "raise" or "ignore".'

    # Detecting the directory containing the entry point
    base_dir = base_dir or os.path.dirname(sys.argv[0])
    if not os.path.exists(base_dir):
        base_dir = os.path.join(ROOT_DIR.replace('diplomacy_research', ''), base_dir)

    # Converting launch dir to diplomacy_research.package.path
    package_path = ''
    for folder_ix, folder in enumerate(base_dir.split('/')):
        if folder == 'diplomacy_research':
            package_path = '.'.join(base_dir.split('/')[folder_ix:])
            break

    # Detecting model_id
    for arg_ix, arg in enumerate(sys.argv[:-1]):
        if arg == '--%s' % arg_name:
            model_id = int(sys.argv[arg_ix + 1])
            break
    else:
        config_data = load_config_file()
        if arg_name in config_data:
            model_id = int(config_data[arg_name])

    # Looking for subfolders in the launch dir containing 'v000'
    prefix = 'v%03d' % model_id
    subfolders = [name for name in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, name)) and name.startswith(prefix)]

    # We didn't find it, raising error or returning a list of None
    if not subfolders:
        if on_error == 'ignore':
            return [None] * len(classes)

        LOGGER.error('Unable to find model id %d in folder: %s', model_id, base_dir)
        LOGGER.error('You can change the model id by passing a "--%s xxx" command line argument.', arg_name)
        raise FileNotFoundError()

    # Otherwise, loading the modules dynamically
    constructors = []
    for class_name in classes:
        LOGGER.info('Loading class: %s.%s.%s', package_path, subfolders[0], class_name)
        constructors += [locate('%s.%s.%s' % (package_path, subfolders[0], class_name))]

    # Returning constructors
    return constructors

def load_dataset_builder(dataset_name, base_dir=None):
    """ Find the correct dataset builder as specified in the command line args.
        :param dataset_name: The name of the dataset builder to load.
        :param base_dir: The directory that contains the models (and also the train.py file)
        :return: Return the dataset builder
    """
    # Detecting the directory containing the entry point
    base_dir = base_dir or os.path.dirname(sys.argv[0])
    if not os.path.exists(base_dir):
        base_dir = os.path.join(ROOT_DIR.replace('diplomacy_research', ''), base_dir)

    # Converting launch dir to diplomacy_research.package.path
    package_path = ''
    for folder_ix, folder in enumerate(base_dir.split('/')):
        if folder == 'diplomacy_research':
            package_path = '.'.join(base_dir.split('/')[folder_ix:])
            break

    # Detecting if dataset exists
    dataset_file_path = os.path.join(base_dir, 'dataset', '%s.py' % dataset_name)
    if not os.path.exists(dataset_file_path):
        LOGGER.error('The file %s does not exists. Unable to load dataset "%s".', dataset_file_path, dataset_name)
        LOGGER.error('You can change the dataset by passing a "--dataset xxx" command line argument.')
        raise FileNotFoundError()

    # Otherwise, returning the DatasetBuilder class
    LOGGER.info('Loading dataset "%s" using %s.dataset.%s.DatasetBuilder', dataset_name, package_path, dataset_name)
    return locate('%s.dataset.%s.DatasetBuilder' % (package_path, dataset_name))

def parse_args_into_flags(args):
    """ Parse args into a series of Tensorflow flags
        :param args: A list of tuples (type, name, value, desc)
        :return: The parsed tf.app.flags.FLAGS
    """
    global PARSER                                                   # pylint: disable=global-statement
    type_parser = {'bool': bool, 'int': int, 'str': str, 'float': float, '---': lambda x: x}

    # Keeping a dictionary of parse args to overwrite if provided multiple times
    # A type of '---' deletes the flags
    parsed_args = {}
    for arg in args:
        arg_type, arg_name, arg_value, arg_desc = arg
        parsed_args[arg_name] = (arg_type, arg_value, arg_desc)

    # Loading configuration file
    config_data = load_config_file()

    # Overwriting fields
    for field_name in config_data:
        if field_name in parsed_args:
            arg_type, _, arg_desc = parsed_args[field_name]
            new_value = type_parser[arg_type](config_data[field_name])
            parsed_args[field_name] = (arg_type, new_value, arg_desc)
        else:
            LOGGER.warning('Unknown field "%s" defined in config file.', field_name)

    # Defining args in parser
    for arg_name, (arg_type, arg_value, arg_desc) in parsed_args.items():
        arg_desc = arg_desc.replace('%', '%%')
        if arg_type == '---':           # Deleted field
            continue
        elif arg_type == 'bool':
            parser_group = PARSER.add_mutually_exclusive_group(required=False)
            parser_group.add_argument('--%s' % arg_name, dest=arg_name, action='store_true', help=arg_desc)
            parser_group.add_argument('--no-%s' % arg_name, dest=arg_name, action='store_false', help=arg_desc)
            parser_group.set_defaults(**{arg_name: arg_value})
        elif arg_type == 'int':
            PARSER.add_argument('--%s' % arg_name, type=int, default=arg_value, help=arg_desc)
        elif arg_type == 'str':
            PARSER.add_argument('--%s' % arg_name, type=str, default=arg_value, help=arg_desc)
        elif arg_type == 'float':
            PARSER.add_argument('--%s' % arg_name, type=float, default=arg_value, help=arg_desc)
        else:
            raise TypeError('Parser. Unsupported type %s.' % arg_type)
    return PARSER.parse_args()

def display_flags(flags):
    """ Displays a list of flags with their corresponding values """
    LOGGER.info('-' * 40)
    if GIT_IS_DIRTY:
        LOGGER.info('Commit Hash: %s --- *** GIT WORKING TREE IS DIRTY ***', GIT_COMMIT_HASH)
    else:
        LOGGER.info('Commit Hash: %s', GIT_COMMIT_HASH)
    LOGGER.info('-' * 40)
    for flag_name in flags.__dict__:
        LOGGER.info('Flag %s: %s', flag_name, str(getattr(flags, flag_name)))
    LOGGER.info('-' * 40)

def display_cluster_config(cluster_config):
    """ Displays the cluster config loaded """
    for name, value in cluster_config._asdict().items():
        str_value = str(value.__dict__).replace('\n', '') if hasattr(value, '__dict__') else str(value)
        LOGGER.info('Config %s: %s', name, str_value)
    LOGGER.info('-' * 40)

def run_app():
    """ Runs the application """
    main = sys.modules['__main__'].main
    sys.exit(main(sys.argv))
