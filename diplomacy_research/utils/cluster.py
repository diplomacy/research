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
""" Cluster - Contains various methods to deal with distributed training """
import asyncio
import collections
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial, wraps
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import hostlist
from tornado import ioloop, gen, util
from tornado.gen import is_coroutine_function
from diplomacy_research.utils.model import display_flags, display_cluster_config

# Constants
LOGGER = logging.getLogger(__name__)
OTHER_ROLE_OFFSET = 100                               # gpu_ids >= 100 are assigned to other roles ('ps', 'redis', etc.)

class PrefetchedItem(
        collections.namedtuple('PrefetchedItem', ('queue_name', 'item'))):
    """ Prefetched item (in numpy format), not yet stored in a queue """

class ClusterConfig(
        collections.namedtuple('ClusterConfig',
                               ('cluster_spec',         # The tf.train.ClusterSpec that defines the cluster
                                'job_name',             # Job name for this process ('worker', 'ps')
                                'task_id',              # The process task id (supplied on process start)
                                'worker_device',        # The device to set on each worker
                                'iterator_device',      # The device to set for the iterator
                                'caching_device',       # The device to use to cache variables locally
                                'partitioner',          # The partitioner used to divide embedding variable
                                'is_chief',             # Chief indicator. Will save checkpoints if set.
                                'protocol',             # The communication protocol ('grpc' or 'grpc+verbs')
                                'num_shards',           # The number of GPU available on the cluster
                                'shard_index',          # The current GPU index (in the entire cluster)
                                'process_pool_cores',   # Number of cores to use for the process pool
                                'cuda_device_id'))):    # The value to set for CUDA_VISIBLE_DEVICE
    """ Contains a cluster configuration with all the required detected settings """
    def count(self, job_name):
        """ Returns the number of tasks assigned to a specified job """
        return len(self.cluster_spec.get(job_name, []))

def start_distributed_training(callable_fn, flags, get_cluster_config_fn, with_process_pool):
    """ Detects if distributed training should be used and starts either regular or distributed training
        :param callable_fn: A callable function that takes (cluster_config, process_pool) as a arg and starts training
        :param flags: The parsed Tensorflow flags
        :param get_cluster_config_fn: A callable function to use to retrieve the cluster configuration
                                      Args: (gpu_id, nb_param_servers, nb_cpu_workers, grpc_port, use_verbs)
        :param with_process_pool: Boolean that indicates to start a process pool and pass it to the callable fn.
    """
    LOGGER.info('Received command line arguments: %s', ' '.join(sys.argv))
    LOGGER.info('CUDA_VISIBLE_DEVICE: %s', os.environ.get('CUDA_VISIBLE_DEVICES', '<Not set>'))
    display_flags(flags)
    hparams = flags.__dict__
    cluster_config = None
    process_pool = None

    # Determining if we can launch regular or distributed training
    nb_gpus = flags.debug_nb_cpu_workers or get_nb_gpus()

    # Starting the process pool
    # 1) We need to have the with_process_pool arg set to True, and
    # 2a) launch in standalone mode, or
    # 2b) launch in distributed mode with a config that indicates that the role needs a process pool, and
    # 3) Have not loaded Tensorflow or gRPC because they are not fork-safe.
    if with_process_pool and ((nb_gpus <= 1 and not flags.debug_nb_cpu_workers) or flags.gpu_id >= 0):
        short_cluster_config = get_cluster_config_fn(full_config=False, **hparams)
        nb_cores = os.cpu_count() if short_cluster_config is None else short_cluster_config.process_pool_cores
        if nb_cores:
            assert 'tensorflow' not in sys.modules
            assert 'grpc' not in sys.modules
            process_pool = ProcessPoolExecutor(max_workers=nb_cores)
            process_pool.submit(dummy_function)             # Using a dummy function to make sure processes are created

    # Already in a sub-process, just retrieving cluster config and calling callable_fn
    if flags.gpu_id >= 0:
        cluster_config = get_cluster_config_fn(**hparams)
        display_cluster_config(cluster_config)

        if not cluster_config:
            LOGGER.warning('Got gpu id of %d, but a blank cluster config was returned. Aborting.', flags.gpu_id)
        else:
            callable_fn(cluster_config, process_pool)
        return

    # Regular training
    if nb_gpus <= 1 and not flags.debug_nb_cpu_workers:
        LOGGER.info('Launching non-distributed training loop.')
        callable_fn(cluster_config, process_pool)
        return

    # Distributed training
    LOGGER.info('Launching distributed training loop.')
    processes = []
    gpu_ids = list(range(nb_gpus))
    gpu_ids += [OTHER_ROLE_OFFSET + ps_id for ps_id in range(10)]

    # Creating output folder for job id
    slurm_job_id = '%05d.' % int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    slurm_job_id += os.environ.get('SLURM_JOB_ID', str(int(round(time.time(), -3))))
    if not os.path.exists(os.path.join(flags.save_dir, slurm_job_id)):
        os.makedirs(os.path.join(flags.save_dir, slurm_job_id), exist_ok=True)

    # Launching separate processes
    for gpu_id in gpu_ids:
        cluster_config = get_cluster_config_fn(gpu_id=gpu_id,
                                               full_config=False,
                                               **{key: value for key, value in hparams.items() if key != 'gpu_id'})
        if not cluster_config:
            continue

        # e.g. 'w.000.leto35-000.log'
        file_name = '%s.%03d.%s.%03d.log' % (cluster_config.job_name, cluster_config.task_id, get_hostname(), gpu_id)
        std_out = open(os.path.join(flags.save_dir, slurm_job_id, 'out.%s' % file_name), 'a')
        std_err = open(os.path.join(flags.save_dir, slurm_job_id, 'err.%s' % file_name), 'a')

        # Setting CUDA_VISIBLE_DEVICE
        # Also setting CUDA_DEVICES_COUNT to the total number of gpus, otherwise the launched process will also
        # detect one gpu
        new_env = os.environ.copy()
        if cluster_config.cuda_device_id is not None:
            new_env["CUDA_VISIBLE_DEVICES"] = cluster_config.cuda_device_id
            new_env["CUDA_DEVICES_COUNT"] = str(get_nb_gpus())

        # Launching
        LOGGER.info('Launching separate process for GPU_ID %d...', gpu_id)
        command = [sys.executable] + ['-u'] + sys.argv + ['--gpu_id', str(gpu_id)]
        process = subprocess.Popen(command, bufsize=0, stdout=std_out, stderr=std_err, env=new_env)
        processes += [process]

    # Handling Ctrl+C or SIGTERM
    def signal_handler(*args):
        """ Handles SIGINT and SIGTERM signals """
        del args  # unused argument
        LOGGER.info('INFO - CTRL-C received. Stopping distributed training.')

        # Terminating all processes
        for process in processes:
            process.terminate()

        # Waiting 30 seconds and killing processes
        LOGGER.info('... Waiting 30 secs for processes to stop ...')
        time.sleep(30)
        for process in processes:
            os.kill(process.pid, signal.SIGKILL)

        LOGGER.info('... All processes are now stopped ...')
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Waiting for CTRL-C to terminate
    while True:
        # Exiting if all processes have exited.
        nb_running_processes = sum([1 if is_process_running(process.pid) else 0 for process in processes])
        if not nb_running_processes:
            break

        # Otherwise, sleeping
        time.sleep(1)

def get_nb_gpus():
    """ Returns the number of GPUs on the current node """
    nb_gpus_env = 0
    if 'CUDA_DEVICES_COUNT' in os.environ:              # Set by the parent process
        nb_gpus_env = int(os.environ['CUDA_DEVICES_COUNT'])
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:          # Detecting gpus in parent process
        nb_gpus_env = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    return nb_gpus_env

def get_cluster_nodes():
    """ Returns a list of the nodes launched by SLURM - Otherwise {'%hostname%': '127.0.0.1'}
        :return: A dictionary with the nodes hostname as key, and its ip address as value
    """
    nodes = collections.OrderedDict()
    if 'SLURM_JOB_NODELIST' in os.environ:
        hosts = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'], sort=True)
        for host in hosts:
            nodes[host] = get_ip_address(host)
    else:
        nodes[get_hostname()] = 'localhost'
    return nodes

def get_hostname():
    """ Returns the hostname of the current node """
    return socket.gethostname().split('.')[0]

def get_ip_address(hostname):
    """ Returns the ip address of the given hostname
        Note: Will return none if the hostname is not found
    """
    if 'SLURM_JOB_NODELIST' not in os.environ and hostname == get_hostname():
        return 'localhost'
    try:
        _, _, ip_list = socket.gethostbyaddr(hostname)
        ip_address = None if not ip_list else ip_list[0]
    except socket.gaierror:
        ip_address = None
    return ip_address

def is_process_running(process_id):
    """ Checks if there is a process with a specific process id that is running """
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    else:
        return True

def is_port_opened(port, hostname='127.0.0.1'):
    """ Checks if the specified port is opened
        :param port: The port to check
        :param hostname: The hostname to check, defaults to '127.0.0.1'
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((hostname, port))
    if result == 0:
        return True
    return False

def kill_processes_using_port(port, force=False):
    """ Kills any remaining processes using the specified port """
    if not is_port_opened(port) and not force:
        return
    try:
        command = ['fuser', '-k', '%d/tcp' % port]
        subprocess.Popen(command).wait(timeout=30)
    except FileNotFoundError:
        LOGGER.error('The "fuser" command was not found. Unable to kill process.')

def dummy_function():
    """ Dummy function to initialize process pools """
    return None


# ==== Tornado IO Loops ====
class CompletedFuture(gen.Future):
    """ Wrapper around a future that marks it completed with a result """
    def __init__(self, result):
        """ Constructor """
        super(CompletedFuture, self).__init__()
        self.set_result(result)

def start_io_loop(on_start_callback, custom_io_loop=None, stop_on_interrupt=True, **kwargs):
    """ Starts an asynchronous IO Loop
        :param on_start_callback: The callback to call when the loop is started.
        :param custom_io_loop: A custom IO loop object to start, otherwise the .instance() loop will be started.
        :param stop_on_interrupt: Boolean. If true, the io loop will stop on CTRL-C.
        :return: Nothing
    """
    if not is_coroutine_function(on_start_callback):
        LOGGER.error('The on_start_callback must be a gen.coroutine()')
        raise RuntimeError()
    io_loop = custom_io_loop or ioloop.IOLoop.instance()
    io_loop.spawn_callback(on_start_callback, **kwargs)
    try:
        io_loop.start()
    except KeyboardInterrupt:
        if stop_on_interrupt:
            io_loop.stop()

def stop_io_loop(custom_io_loop=None):
    """ Stops an asynchronous IO loop
        :param custom_io_loop: A custom IO loop object to stop, otherwise the .instance() loop will be stopped.
        :return: Nothing
    """
    io_loop = custom_io_loop or ioloop.IOLoop.instance()
    io_loop.stop()

def is_ioloop_running(custom_io_loop=None):
    """ Detects if the ioloop is running
        :param custom_io_loop: A custom IO loop object to check, otherwise, the .instance() loop will be checked
        :return: A boolean that indicates if the loop has been started
    """
    io_loop = custom_io_loop or ioloop.IOLoop.instance()
    return io_loop.asyncio_loop.is_running()

def get_current_io_loop():
    """ Returns the current IO loop in the thread """
    return ioloop.IOLoop.instance()

@gen.coroutine
def yield_with_display(future_or_iterable, every, timeout=None):
    """ Yields for a future and display status every x seconds
        :param future_or_iterable: A future to yield on, or a list of futures
        :param every: The number of seconds between updates
        :param timeout: The total number of seconds to wait for the future, otherwise throws TimeoutError
    """
    start_time = time.time()
    last_status = start_time
    futures = [future_or_iterable] if not isinstance(future_or_iterable, list) else future_or_iterable

    # Looping until timeout or future is done
    while [1 for future in futures if not future.done()]:
        current_time = time.time()

        # Timeout reached
        if timeout and current_time - start_time > timeout:
            raise TimeoutError('Waited %d seconds for future.' % timeout)

        # Displaying status
        if current_time - last_status > every:
            last_status = current_time
            LOGGER.info('Still waiting for future(s). %s/%s', int(current_time - start_time), timeout or '---')

        # Sleeping
        yield gen.sleep(0.1)

    # Futures are done, rethrowing exception
    for future in futures:
        exception = future.exception()
        if exception is not None:
            raise exception

    # Returning results
    results = [future.result() for future in futures]
    if not isinstance(future_or_iterable, list):
        results = results[0]
    return results

@gen.coroutine
def with_timeout(timeout, future_or_iterable):
    """ Yields on a future (or a list of futures) for a maximum amount of time
        Returns the results of the completed futures completed after timeout (or None if the future is still pending)
        :param timeout: The number of seconds to wait for the future. 0 or None to disable.
        :param future_or_iterable: A future to yield on, or a list of futures
        :return: The results from the future or the list of futures, or None if the future didn't complete in time
    """
    # Waiting for a list
    if isinstance(future_or_iterable, list):
        futures = future_or_iterable
        try:
            if timeout:
                yield gen.with_timeout(timedelta(seconds=timeout), gen.multi(futures))
            else:
                yield gen.multi(futures)
        except util.TimeoutError:
            pass
        return [None if not future.done() else future.result() for future in futures]

    # Waiting for a single future
    try:
        if timeout:
            yield gen.with_timeout(timedelta(seconds=timeout), future_or_iterable)
        else:
            yield future_or_iterable
    except util.TimeoutError:
        pass
    return None if not future_or_iterable.done() else future_or_iterable.result()

@gen.coroutine
def process_fetches_dict(feedable_dataset, fetches):
    """ Fetches is a nested dictionary with str: (PrefetchedItem or Future)
        1) if the feedable_dataset is a QueueDataset, we convert the PrefetchedItems to futures,
        2) we then yield on all pending futures to get all the results
        3) we replace the futures with their results, so all PrefetchedItem in the original dict becomes results.

        :param feedable_dataset: The dataset where we can feed items to get a future.
        :param fetches: A hierarchy of dict and list, with PrefetchedItem as leaves
        :return: The same structure as fetches, but PrefetchedItem are replaced by their values
        :type feedable_dataset: diplomacy_research.models.datasets.feedable_dataset.FeedableDataset
    """
    from diplomacy_research.models.datasets.queue_dataset import QueueDataset
    if isinstance(feedable_dataset, QueueDataset):
        fetches = _replace_prefetched_items(feedable_dataset, fetches)
    pending_futures = _find_pending_futures(fetches)
    if pending_futures:
        yield pending_futures
    return _replace_futures(fetches)

def remove_completed_futures(fetches):
    """ Removes completed futures and replaces them with their results """
    if isinstance(fetches, (gen.Future, asyncio.Future)) and fetches.done():
        return remove_completed_futures(fetches.result())
    if isinstance(fetches, list):
        return [remove_completed_futures(item) for item in fetches]
    if isinstance(fetches, dict):
        return {key: remove_completed_futures(fetches[key]) for key in fetches}
    return fetches

def _find_pending_futures(fetches):
    """ Finds all pending futures in fetches and return a list of futures to yield on """
    pending_futures = []
    if isinstance(fetches, (gen.Future, asyncio.Future)):
        if fetches.done():
            pending_futures += _find_pending_futures(fetches.result())
        else:
            pending_futures += [fetches]
    if isinstance(fetches, list):
        for item in fetches:
            pending_futures += _find_pending_futures(item)
    if isinstance(fetches, dict):
        for key in fetches:
            pending_futures += _find_pending_futures(fetches[key])
    return pending_futures

def _replace_prefetched_items(queue_dataset, fetches):
    """ Converts all PrefetchedItems to futures
        :type queue_dataset: QueueDataset
    """
    if isinstance(fetches, PrefetchedItem):
        return queue_dataset.put_item_in_queue(fetches.queue_name, fetches.item)
    if isinstance(fetches, (gen.Future, asyncio.Future)) and fetches.done():
        return _replace_prefetched_items(queue_dataset, fetches.result())
    if isinstance(fetches, list):
        return [_replace_prefetched_items(queue_dataset, item) for item in fetches]
    if isinstance(fetches, dict):
        return {key: _replace_prefetched_items(queue_dataset, fetches[key]) for key in fetches}
    return fetches

def _replace_futures(fetches):
    """ Finds all futures in fetches and replaces them with future.result() """
    if isinstance(fetches, (gen.Future, asyncio.Future)):
        return _replace_futures(fetches.result())
    if isinstance(fetches, list):
        return [_replace_futures(item) for item in fetches]
    if isinstance(fetches, dict):
        return {key: _replace_futures(fetches[key]) for key in fetches}
    return fetches


# ==== gRPC Future wrappers ====
## Source: https://github.com/hubo1016/aiogrpc/blob/01b438fe28b3cb987a7fd96ab6447e85473a31d7/aiogrpc/utils.py
## Apache2 License

def _wrap_callback(callback, asyncio_loop):
    """ Wraps a callback to check if the asyncio loop is active """
    @wraps(callback)
    def _callback(*args, **kwargs):
        """ Callback """
        if not asyncio_loop.is_closed():
            asyncio_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))
    return _callback

def _wrap_active_test(func, test, asyncio_loop, executor=None):
    """ Wraps a function using a test """
    @wraps(func)
    @asyncio.coroutine
    def _func(*args, **kwargs):
        """ Wrapper """
        if test():
            return (yield asyncio_loop.run_in_executor(executor, partial(func, *args, **kwargs)))
        return func(*args, **kwargs)
    return _func

def _wrap_grpc_future(grpc_future, asyncio_loop):
    """ Wraps a gRPC future inside an asyncio future
        :param grpc_future: The gRPC future to wrap
        :param asyncio_loop: The current asyncio loop
        :return: An asyncio future
    """
    asyncio_future = asyncio_loop.create_future()

    def _set_state(grpc_fut, asyncio_fut):
        """ Sets the state of the asyncio future to that of the grpc future """
        assert grpc_fut.done()
        if asyncio_fut.cancelled():
            return
        assert not asyncio_fut.done()
        if grpc_fut.cancelled():
            asyncio_fut.cancel()
            return
        exception = grpc_fut.exception()
        if exception is not None:
            asyncio_fut.set_exception(exception)
        else:
            asyncio_fut.set_result(grpc_fut.result())

    def _call_check_cancel(asyncio_fut):
        """ Checks if the asyncio future is cancelled, and cancels the grpc future accordingly """
        if asyncio_fut.cancelled():
            grpc_future.cancel()

    def _call_set_state(grpc_fut):
        """ Checks if the loop is running and sets the asyncio future results """
        if not asyncio_loop.is_closed():
            asyncio_loop.call_soon_threadsafe(_set_state, grpc_fut, asyncio_future)

    # Setting callbacks
    asyncio_future.add_done_callback(_call_check_cancel)
    grpc_future.add_done_callback(_call_set_state)
    return asyncio_future

def wrap_grpc_call(grpc_future, asyncio_loop=None, executor=None):
    """ Wrapper around a gRPC call to make it compatible with asyncio
        :param grpc_future: The gRPC future returned from the gRPC call
        :param asyncio_loop: The current asyncio loop (otherwise get_event_loop is called)
        :param executor: The asyncio executor to use
        :return: An asyncio future
    """
    asyncio_loop = asyncio_loop or asyncio.get_event_loop()
    asyncio_future = _wrap_grpc_future(grpc_future, asyncio_loop)
    setattr(asyncio_future, 'is_active', getattr(grpc_future, 'is_active'))
    setattr(asyncio_future, 'time_remaining', getattr(grpc_future, 'time_remaining'))

    # Wrapping the add_callback function
    @wraps(grpc_future.add_callback)
    def _add_callback(callback):
        """ Wrapper for add_callback """
        grpc_future.add_callback(_wrap_callback(callback, asyncio_loop))
    asyncio_future.add_callback = _add_callback

    # Wraps internal methods
    wrapper = partial(_wrap_active_test, test=grpc_future.is_active, asyncio_loop=asyncio_loop, executor=executor)
    setattr(asyncio_future, 'initial_metadata', wrapper(getattr(grpc_future, 'initial_metadata')))
    setattr(asyncio_future, 'trailing_metadata', wrapper(getattr(grpc_future, 'trailing_metadata')))
    setattr(asyncio_future, 'code', wrapper(getattr(grpc_future, 'code')))
    setattr(asyncio_future, 'details', wrapper(getattr(grpc_future, 'details')))

    # Returning asyncio_future
    return asyncio_future
