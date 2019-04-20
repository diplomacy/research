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
""" Settings
    - Contains the list of fixed settings to be used across the module
"""
import datetime
import logging
import os
import subprocess
import tempfile

# Checking if protobuf uses cpp
if os.environ.get('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', '') != 'cpp':
    logging.warning('The PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION variable should be set to "cpp".')

VALIDATION_SET_SPLIT = 0.05
SESSION_RUN_TIMEOUT = 60000         # 60s
LOG_INF_NAN_TENSORS = bool('LOG_INF_NAN_TENSORS' in os.environ)
NO_PRESS_DATASET = 'no_press'
NO_PRESS_ALL_DATASET = 'no_press_all'
NO_PRESS_LARGE_DATASET = 'no_press_large'
NO_PRESS_VALUE_DATASET = 'no_press_value'
NO_PRESS_VALUE_ALL_DATASET = 'no_press_value_all'
NO_PRESS_VALUE_LARGE_DATASET = 'no_press_value_large'
YYYY_MM_DD = datetime.datetime.today().strftime('%Y-%m-%d')
NB_PARTITIONS = 24

# Settings used to regenerate datasets
DATASET_DATE = '20190501'
PROTOBUF_DATE = '20190501'
DATASET_VERSION = 'v10'

# Using working directory if available
# 1) Looks at WORKING_DIR env variable
# 2) then defaults to /Tmp/$USER/diplomacy
# 3) otherwise, defaults to a temporary directory
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'build')
if 'WORKING_DIR' in os.environ:
    WORKING_DIR = os.environ['WORKING_DIR']
elif 'USER' in os.environ:
    WORKING_DIR = os.path.join('/Tmp/{}'.format(os.environ['USER']), 'diplomacy')
else:
    WORKING_DIR = tempfile.mkdtemp()

# Determining save_dir (to save checkpoints) in working_dir
# IN_PRODUCTION indicates that we are inside a container
IN_PRODUCTION = WORKING_DIR.startswith('/work_dir')
if IN_PRODUCTION:
    DEFAULT_SAVE_DIR = os.path.join(WORKING_DIR, 'checkpoint_data')
else:
    DEFAULT_SAVE_DIR = os.path.join(WORKING_DIR, '{}_saved_supervised'.format(YYYY_MM_DD))

# Dataset paths
ZIP_DATASET_PATH = os.path.join(WORKING_DIR, 'diplomacy-v1-27k-msgs.zip')
DATASET_PATH = os.path.join(WORKING_DIR, '{}_dataset_{}.hdf5'.format(DATASET_DATE, DATASET_VERSION))
PROTO_DATASET_PATH = os.path.join(WORKING_DIR, '{}_dataset_{}.pb'.format(DATASET_DATE, DATASET_VERSION))
DATASET_INDEX_PATH = os.path.join(WORKING_DIR, '{}_dataset_index.pkl'.format(DATASET_DATE))
HASH_DATASET_PATH = os.path.join(WORKING_DIR, '{}_hash.pkl'.format(DATASET_DATE))
TOKENS_DATASET_PATH = os.path.join(WORKING_DIR, '{}_tokens.pkl'.format(DATASET_DATE))
END_SCS_DATASET_PATH = os.path.join(WORKING_DIR, '{}_end_scs.pkl'.format(DATASET_DATE))
MOVES_COUNT_DATASET_PATH = os.path.join(WORKING_DIR, '{}_moves_count.pkl'.format(DATASET_DATE))
PHASES_COUNT_DATASET_PATH = os.path.join(WORKING_DIR, '{}_phases_count.pkl'.format(DATASET_DATE))
REDIS_DATASET_PATH = os.path.join(WORKING_DIR, '{}_redis.rdb'.format(DATASET_DATE))

# Redis Settings
REDIS_HOSTNAME = os.environ.get('REDIS_HOSTNAME', '127.0.0.1')

# Git Commit hash
GIT_COMMIT_HASH = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT_DIR)
GIT_COMMIT_HASH = GIT_COMMIT_HASH.decode('utf-8').replace('\n', '')
GIT_IS_DIRTY = subprocess.check_output(['git', '-c', 'core.fileMode=false', 'status', '--short'], cwd=ROOT_DIR)
GIT_IS_DIRTY = bool(GIT_IS_DIRTY.decode('utf-8').replace('\n', ''))

# Downloadable containers URLs
REDIS_DOWNLOAD_URL = 'https://f002.backblazeb2.com/file/ppaquette-public/containers/redis/redis-5.0.3-3.sif'
TF_SERVING_DOWNLOAD_URL = 'https://f002.backblazeb2.com/file/ppaquette-public/containers/tensorflow_serving/tensorflow_serving-f16e777-tf1.13-001.sif'  # pylint: disable=line-too-long
ALBERT_AI_DOWNLOAD_URL = 'https://f002.backblazeb2.com/file/ppaquette-public/containers/albert_ai/albert_dumbbot-1.2.sif'                               # pylint: disable=line-too-long
