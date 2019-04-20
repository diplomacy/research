#!/usr/bin/env python3
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
""" Dataset Builder
    - Builds the various datasets required from the zip dataset
"""
import argparse
import glob
import importlib
import inspect
import logging
import os

# Constants
LOGGER = logging.getLogger('diplomacy_research.scripts.build_dataset')

def parse_args():
    """ Returns the args parsed from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str,
                        help='Output logging file to collect processing info. If not specified, no logging info about '
                             'games processing will be printed.')
    parser.add_argument('--input-path', type=str, default=os.getcwd(),
                        help='Folder of raw games data. Should contain sub-folders curl_p, curl_w, and/ord curl_f.')
    parser.add_argument('--hostname', type=str, default='localhost',
                        help='Hostname to connect to sql_w MySQL database.')
    parser.add_argument('--port', type=int, default=3306,
                        help='port to connect to sql_w MySQL database.')
    parser.add_argument('--username', type=str, default='root',
                        help='username to connect to sql_w MySQL database.')
    parser.add_argument('--password', type=str, default='',
                        help='password to connect to sql_w MySQL database.')
    parser.add_argument('--database', type=str, default='webdiplomacy',
                        help='database name to connect to sql_w MySQL database.')
    parser.add_argument('--filter', type=str, default='',
                        help='A comma-separated list of proto dataset to generate (e.g. "order_based/no_press_all")')
    return parser.parse_args()

if __name__ == '__main__':
    from diplomacy_research.models.datasets.base_builder import BaseBuilder

    # Parsing arguments
    ARGS = parse_args()

    # General dataset builders
    # Loading all files in diplomacy_research/scripts/dataset
    # and running the run() function in each of them
    SCRIPTS_DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
    for dataset_path in sorted(glob.glob(os.path.join(SCRIPTS_DATASET_DIR, '*_???_*.py'))):
        module_path = 'diplomacy_research.scripts.dataset.%s' % dataset_path.split('/')[-1].replace('.py', '')
        imported_path = importlib.import_module(module_path)
        for run_fn in [obj for name, obj in inspect.getmembers(imported_path) if name == 'run']:
            LOGGER.info('========== %s ==========', module_path)
            run_fn(**vars(ARGS))

    # We only want to generate the common dataset scripts - Exiting here
    if ARGS.filter and ARGS.filter == 'common':
        exit(0)

    # Policy dataset builders
    # Loading all files in diplomacy_research/models/policy/xxxx/dataset
    # and calling the generate_proto_files() of each builder class
    for dataset_path in ['diplomacy_research.models.policy.order_based.dataset',
                         'diplomacy_research.models.policy.token_based.dataset']:
        imported_path = importlib.import_module(dataset_path)
        for obj_name, obj_class in [(name, obj) for name, obj in inspect.getmembers(imported_path)
                                    if inspect.isclass(obj) and issubclass(obj, BaseBuilder) and obj != BaseBuilder]:
            obj_filtered_class = '%s/%s' % (dataset_path.split('.')[-2], obj_class.__module__.split('.')[-1])
            if ARGS.filter and obj_filtered_class not in ARGS.filter.split(','):
                continue
            LOGGER.info('Checking if dataset builder "%s" in path "%s" exists...', obj_filtered_class, dataset_path)
            if not os.path.exists(obj_class.training_dataset_path):
                obj_class().generate_proto_files()
            else:
                LOGGER.info('Dataset for "%s" in path "%s" exist. Skipping...', obj_filtered_class, dataset_path)
