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
""" Process - Contains various methods to launch subprocesses """
import atexit
import collections
import glob
import logging
import math
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import requests
import tqdm
from diplomacy_research.utils.cluster import is_port_opened
from diplomacy_research.settings import REDIS_DOWNLOAD_URL, TF_SERVING_DOWNLOAD_URL, ALBERT_AI_DOWNLOAD_URL, \
    WORKING_DIR, IN_PRODUCTION

# Constants
LOGGER = logging.getLogger(__name__)
SPAWN_CONTEXT = multiprocessing.get_context('spawn')

class BatchingParameters(
        collections.namedtuple('BatchingParameters',
                               ('max_batch_size',           # Maximum batch size (should use 2*max_bs nb_threads)
                                'batch_timeout_micros',     # Maximum timeout in micro-seconds before batching items
                                'max_enqueued_batches',     # Maximum number of batch in queues
                                'num_batch_threads',        # Number of threads used for batching
                                'pad_variable_length_inputs'))):    # Boolean that indicates to pad var len inputs
    """ Contains the BatchingParameters proto fields for TF Serving """
    # From: tensorflow_serving/servables/tensorflow/session_bundle_config.proto

    def __str__(self):
        """ Returns a string representation """
        str_format = 'max_batch_size { value: %d }\n' % self.max_batch_size
        str_format += 'batch_timeout_micros { value: %d }\n' % self.batch_timeout_micros
        str_format += 'max_enqueued_batches { value: %d }\n' % self.max_enqueued_batches
        str_format += 'num_batch_threads { value: %d }\n' % self.num_batch_threads
        str_format += 'pad_variable_length_inputs: %s\n' % 'true' if self.pad_variable_length_inputs else 'false'
        return str_format

def is_redis_running(hostname='127.0.0.1'):
    """ Checks is Redis is running on the specified hostname """
    return is_port_opened(port=6379, hostname=hostname)

def start_redis(save_dir, block_thread=True, log_file_path=None, clear=False, import_db_path=None):
    """ Starts the redis server locally
        :param save_dir: The current flags.save_dir
        :param block_thread: If true, blocks the thread and automatically kills the subprocess on exit
                             Otherwise, launches an instance that won't be killed on exit.
        :param log_file_path: Optional. Specify the path of log file where to output std.out and std.err
        :param clear: Boolean. If true, deletes the database from disk before starting the server.
        :param import_db_path: If set and a database doesn't exist on disk, copies the database and uses it
    """
    if is_redis_running():
        LOGGER.error('Redis is already running on the localhost. Not starting another instance.')
        return

    # If log_file_path is set, redirecting stdout and stderr to it.
    stdout = open(log_file_path, 'a') if log_file_path else None

    # In production, (inside container)
    # Launching directly
    if IN_PRODUCTION:
        os.makedirs(os.path.join('/work_dir', 'redis'), exist_ok=True)
        redis_folder = '/work_dir/redis/'
        redis_db_path = os.path.join(redis_folder, 'saved_redis.rdb')

        # Deleting file if less than 1 GB
        if os.path.exists(redis_db_path) and os.path.getsize(redis_db_path) < 2**30:
            os.unlink(redis_db_path)

        # Clearing previous database
        if clear and os.path.exists(redis_db_path):
            LOGGER.info('Clearing the current Redis database.')
            os.unlink(redis_db_path)

        # Copying import database
        if import_db_path and os.path.exists(import_db_path) and not os.path.exists(redis_db_path):
            LOGGER.info('Copying %s to use as the initial Redis database', import_db_path)
            shutil.copy(import_db_path, redis_db_path)

        # Deleting temporary files
        for temp_file_path in (glob.glob(os.path.join(redis_folder, 'temp*.rdb'))
                               + glob.glob(os.path.join(redis_folder, 'core*'))):
            if os.path.exists(temp_file_path):
                LOGGER.info('Deleting temporary file: %s', temp_file_path)
                os.unlink(temp_file_path)

        # Launching Redis
        command = ['redis-server', '/work_dir/redis/redis.conf']

    # Otherwise, downloading containers and starting singularity
    else:
        os.makedirs(os.path.join(save_dir, 'redis'), exist_ok=True)

        # Downloading container
        redis_img = os.path.join(WORKING_DIR, 'containers', REDIS_DOWNLOAD_URL.split('/')[-1])
        download_file(REDIS_DOWNLOAD_URL, redis_img)
        redis_folder = os.path.join(save_dir, 'redis')
        redis_db_path = os.path.join(redis_folder, 'saved_redis.rdb')

        # Deleting file if less than 1 GB
        if os.path.exists(redis_db_path) and os.path.getsize(redis_db_path) < 2**30:
            os.unlink(redis_db_path)

        # Clearing previous database
        if clear and os.path.exists(redis_db_path):
            LOGGER.info('Clearing the current Redis database.')
            os.unlink(redis_db_path)

        # Copying import database
        if import_db_path and os.path.exists(import_db_path) and not os.path.exists(redis_db_path):
            LOGGER.info('Copying %s to use as the initial Redis database', import_db_path)
            shutil.copy(import_db_path, redis_db_path)

        # Deleting temporary files
        for temp_file_path in (glob.glob(os.path.join(redis_folder, 'temp*.rdb'))
                               + glob.glob(os.path.join(redis_folder, 'core*'))):
            if os.path.exists(temp_file_path):
                LOGGER.info('Deleting temporary file: %s', temp_file_path)
                os.unlink(temp_file_path)

        # Launching an instance if block_thread = False
        if block_thread:
            command = ['singularity', 'run',
                       '-B', '%s:/work_dir' % os.path.join(save_dir, 'redis'),
                       redis_img,
                       'redis-server', '/work_dir/redis.conf']
        else:
            command = ['singularity', 'instance.start',
                       '-B', '%s:/work_dir' % os.path.join(save_dir, 'redis'),
                       redis_img, 'redis']

    # Launching process
    _start_process(command,
                   block_thread=block_thread,
                   check_fn=is_redis_running,
                   bufsize=0,
                   stdout=stdout,
                   stderr=stdout)

    # Logging status
    if not block_thread:
        if not is_redis_running():
            LOGGER.error('Unable to start Redis')
        else:
            LOGGER.info('Started Redis server locally.')

def start_tf_serving(port, save_dir, batching_parameters=None, cluster_config=None, poll_time=1, force_cpu=False,
                     log_file_path=None):
    """ Starts the tf serving server locally
        :param port: Integer. The port to open for incoming connections.
        :param save_dir: The current flags.save_dir
        :param batching_parameters: A BatchingParameters named tuple. Otherwise, uses the default.
        :param cluster_config: The cluster configuration used for distributed training.
        :param poll_time: The number of seconds between polls on the file system to check for new version.
        :param force_cpu: Boolean. If true, forces the serving to rn on CPU. Otherwise uses CUDA_VISIBLE_DEVICES.
        :param log_file_path: Optional. Specify the path of log file where to output std.out and std.err
        :type cluster_config: diplomacy_research.utils.cluster.ClusterConfig
        Note: This automatically blocks the thread
    """
    if is_port_opened(port):
        LOGGER.error('The port %d is already opened locally. Not starting TF Serving.', port)
        return

    # Creating serving directory
    os.makedirs(os.path.join(save_dir, 'serving'), exist_ok=True)
    file_suffix = '' if not cluster_config else '_%s.%03d' % (cluster_config.job_name, cluster_config.task_id)

    # Copying env variables
    new_env = os.environ.copy()
    if force_cpu:
        new_env['CUDA_VISIBLE_DEVICES'] = ''

    # If log_file_path is set, redirecting stdout and stderr to it.
    stdout = open(log_file_path, 'a') if log_file_path else None

    # Creating batch parameters config file
    batching_parameters = batching_parameters or BatchingParameters(max_batch_size=64,
                                                                    batch_timeout_micros=250000,
                                                                    max_enqueued_batches=256,
                                                                    num_batch_threads=multiprocessing.cpu_count(),
                                                                    pad_variable_length_inputs=True)
    filename = 'batch%s.txt' % file_suffix
    with open(os.path.join(save_dir, 'serving', filename), 'w') as file:
        file.write(str(batching_parameters))

    # In production, (inside container)
    # Launching directly
    if IN_PRODUCTION:
        command = ['tensorflow_model_server',
                   '--port=%d' % port,
                   '--enable_batching=true',
                   '--batching_parameters_file=%s' % os.path.join(save_dir, 'serving', filename),
                   '--model_base_path=/data/serving/',
                   '--file_system_poll_wait_seconds=%d' % poll_time]

    # Otherwise, downloading containers and starting singularity
    else:
        # Downloading container
        tf_serving_img = os.path.join(WORKING_DIR, 'containers', TF_SERVING_DOWNLOAD_URL.split('/')[-1])
        download_file(TF_SERVING_DOWNLOAD_URL, tf_serving_img)

        # Never launching an instance, since we need a different port every time
        command = ['singularity', 'exec',
                   '-B', '%s:/work_dir' % save_dir,
                   tf_serving_img,
                   'tensorflow_model_server',
                   '--port=%d' % port,
                   '--enable_batching=true',
                   '--batching_parameters_file=/work_dir/serving/%s' % filename,
                   '--model_base_path=/data/serving/',
                   '--file_system_poll_wait_seconds=%d' % poll_time]

    # Launching process
    _start_process(command,
                   block_thread=True,
                   check_fn=lambda: is_port_opened(port),
                   bufsize=0,
                   env=new_env,
                   stdout=stdout,
                   stderr=stdout)

def start_albert_player(hostname, port, log_file_path=None):
    """ Launches a Albert.AI player
        :param hostname: String. The hostname to connect to.
        :param port: Integer. The port to connect to.
        :param log_file_path: Optional. Specify the path of log file where to output std.out and std.err
        Note: This automatically blocks the thread
    """
    # Forcing CPU
    new_env = os.environ.copy()
    new_env['CUDA_VISIBLE_DEVICES'] = ''

    # If log_file_path is set, redirecting stdout and stderr to it.
    stdout = open(log_file_path, 'a') if log_file_path else None

    # In production, (inside container)
    # Launching directly
    if IN_PRODUCTION:
        if not os.path.exists('/data/albert/Albert.exe'):
            raise NotImplementedError('Albert AI is not supported in production.')

    # Otherwise, downloading containers and starting singularity
    else:
        # Downloading container
        bot_img = os.path.join(WORKING_DIR, 'containers', ALBERT_AI_DOWNLOAD_URL.split('/')[-1])
        download_file(ALBERT_AI_DOWNLOAD_URL, bot_img)

        # Launching with singularity run
        # Syntax: run "albert" <server port>
        command = ['singularity', 'run', '-C', bot_img, 'albert', str(hostname), str(port)]

        # Launching process
        _start_process(command,
                       block_thread=True,
                       check_fn=lambda: True,
                       bufsize=0,
                       env=new_env,
                       stdout=stdout,
                       stderr=stdout)

def start_dumbbot_player(hostname, port, log_file_path=None):
    """ Launches a DumbBot player
        :param hostname: String. The hostname to connect to.
        :param port: Integer. The port to connect to.
        :param log_file_path: Optional. Specify the path of log file where to output std.out and std.err
        Note: This automatically blocks the thread
    """
    # Forcing CPU
    new_env = os.environ.copy()
    new_env['CUDA_VISIBLE_DEVICES'] = ''

    # If log_file_path is set, redirecting stdout and stderr to it.
    stdout = open(log_file_path, 'a') if log_file_path else None

    # In production, (inside container)
    # Launching directly
    if IN_PRODUCTION:
        if not os.path.exists('/data/dumbbot/DumbBot.exe'):
            raise NotImplementedError('DumbBot is not supported in production.')

    # Otherwise, downloading containers and starting singularity
    else:
        # Downloading container
        bot_img = os.path.join(WORKING_DIR, 'containers', ALBERT_AI_DOWNLOAD_URL.split('/')[-1])
        download_file(ALBERT_AI_DOWNLOAD_URL, bot_img)

        # Launching with singularity run
        # Syntax: run "dumbbot" <server port>
        command = ['singularity', 'run', '-C', bot_img, 'dumbbot', str(hostname), str(port)]

        # Launching process
        _start_process(command,
                       block_thread=True,
                       check_fn=lambda: True,
                       bufsize=0,
                       env=new_env,
                       stdout=stdout,
                       stderr=stdout)

def download_file(url, target_file, force=False):
    """ Downloads a file to the target path if the file isn't already there
        :param url: The source URL to use to download the file
        :param target_file: The location (including the filename) where to save the file
        :param force: Boolean. If true, download the file even if it exists on disk.
    """
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    if force or not os.path.exists(target_file):
        LOGGER.info('Downloading image from %s', url)

        # Getting headers and download size
        req = requests.get(url, stream=True)
        content_length = int(req.headers.get('content-length'))
        chunk_size = 1024 ** 2                                              # 1 MB
        total_chunks = int(math.ceil(content_length / chunk_size))

        # Downloading
        with open(target_file, 'wb') as file:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size), total=total_chunks):
                if chunk:
                    file.write(chunk)
                    file.flush()
        LOGGER.info('Done downloading image from %s', url)

def _start_process(command, block_thread, check_fn, **kwargs):
    """ Starts a command (either by blocking the thread or by launching a separate process)
        :param command: A list of args to pass to subprocess.Popen (e.g. ['singularity', 'run', 'container']
        :param block_thread: If true, blocks the thread and automatically kills the subprocess on exit
                             Otherwise, launches an instance that won't be killed on exit.
        :param check_fn: A callable function that retuns a boolean to indicates if the process was started.
        :param **kwargs: Keyword args to pass to the subprocess.
    """
    # Always launching process from home
    kwargs['cwd'] = os.path.expanduser('~')

    # Running in a process group, and blocking
    if block_thread:
        os.setpgrp()
        kill_subprocesses_on_exit()
        subprocess.call(command, **kwargs)

    # Otherwise, launching separately
    else:
        kill_subprocesses_on_exit()
        subprocess.Popen(command, **kwargs)

        # Waiting for process to start
        nb_tries = 0
        while not check_fn() and nb_tries <= 30:
            nb_tries += 1
            time.sleep(1)

def kill_subprocesses_on_exit():
    """ Registers a function to kill the subprocesses automatically on exit """
    def on_exit():
        """ Cleanup for sys.exit() """
        LOGGER.info('sys.exit() called. Sending SIG_KILL to all processes in process group.')
        os.killpg(0, signal.SIGKILL)

    def on_sig_term(*args):
        """ Cleanup for SIGTERM """
        del args
        time.sleep(10.)
        sys.exit(0)

    # Registering handlers for proper termination
    os.setpgrp()
    atexit.register(on_exit)
    if threading.current_thread() == threading.main_thread():
        signal.signal(signal.SIGTERM, on_sig_term)


# ==== Run Tests in Separate Process ====
def _run_process(target, pipe):
    """ Runs a process """
    try:
        multiprocessing.Process.run(multiprocessing.Process(target=target))
        pipe.send(None)
    except Exception as exception:
        pipe.send((exception, traceback.format_exc()))
        raise exception

def run_in_separate_process(target, timeout):
    """ Launches the target in a separate process
        :param target: The callable to launch in the separate process
        :param timeout: The maximum time (in seconds) before killing the process (None for infinite)
    """
    parent_pipe, child_pipe = SPAWN_CONTEXT.Pipe()
    process = SPAWN_CONTEXT.Process(target=_run_process, args=(target, child_pipe))
    process.start()
    process.join(timeout=timeout)

    # Checking for timeout
    timed_out = False
    if timeout and process.is_alive():
        LOGGER.warning('Process with target %s is still alive after %d seconds. Waiting 30 secs.', target, timeout)

        # Waiting an additional 30 secs, then killing
        process.join(timeout=30)
        if process.is_alive():
            LOGGER.warning('Process with target %s is still alive after %d seconds. Killing.', target, timeout + 30)
            os.kill(process.pid, signal.SIGINT)
            process.join(timeout=10)
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
                process.join(timeout=10)
            timed_out = True

    # Checking for exception
    if parent_pipe.poll():
        data = parent_pipe.recv()
        if data is not None:
            exception, child_traceback = data
            print(child_traceback)
            raise exception

    # Raising TimeoutError
    if timed_out:
        raise TimeoutError()
