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
""" Memory Buffer - Barrier
    - Class responsible for creating a sync barrier for distributed training on the memory buffer
"""
import logging
import time

# Constants
LOGGER = logging.getLogger(__name__)
__BARRIER__ = 'barrier.%s'                      ## [hash] barrier.              -> 0 running, 1 done, else nb_epochs
__BARRIER_LAST_CLEARED__ = 'barrier.cleared'    ## [hash] barrier.cleared       -> last timestamp of clear_barrier
__BARRIER_LOCK__ = 'lock.barrier'


def set_barrier_status(buffer, barrier_name, value):
    """ Sets our status on the barrier
        :param buffer: An instance of the memory buffer.
        :param barrier_name: The name of the barrier. (e.g. 'train' or 'eval')
        :param value: The value to set on the barrier (e.g. the number of completed epochs)
                      Typically, a value of 0 means started, a value of 1 means completed
                      but some algos (e.g. PPO) might require multiple value (one for each mini-epoch)
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    LOGGER.info('[%s] Setting barrier value to %d', barrier_name, value)
    token = '%s.%06d' % (buffer.cluster_config.job_name, buffer.cluster_config.task_id)
    with buffer.redis.lock(__BARRIER_LOCK__, timeout=120):
        buffer.redis.hset(__BARRIER__ % barrier_name, token, value)

def clear_barrier(buffer, barrier_name, cleared_time=None):
    """ Clears the barrier and instructs all workers to continue
        :param buffer: An instance of the memory buffer.
        :param barrier_name: The name of the barrier. (e.g. 'train' or 'eval')
        :param cleared_time: Optional. The time the barrier was cleared (time.time())
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    if not buffer.cluster_config.is_chief:
        return
    with buffer.redis.lock(__BARRIER_LOCK__, timeout=120):
        cleared_time = cleared_time if cleared_time is not None else int(time.time())
        buffer.redis.delete(__BARRIER__ % barrier_name)
        buffer.redis.hset(__BARRIER_LAST_CLEARED__, barrier_name, cleared_time)
        LOGGER.info('[%s] Barrier has been cleared at %d', barrier_name, cleared_time)

def wait_for_barrier(buffer, barrier_name, job_name=None, min_value=1, min_done=1):
    """ Waits until the barrier is cleared (or the wait condition is met).
        :param buffer: An instance of the memory buffer.
        :param barrier_name: The name of the barrier. (e.g. 'train' or 'eval')
        :param job_name: If set, only looks at workers performing the specific job (e.g. 'learner', 'actor')
        :param min_value: The min value required for the worker to be considered done.
        :param min_done: The min number of done worker to stop waiting for the barrier.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer

        i.e. if job_name is not set, we wait until clear_barrier() is called.
             otherwise, we wait until 1) there are no more pending workers, and
                                      2) min_done workers of type 'job_name' have set a value of at least min_value.
    """
    LOGGER.info('Waiting for barrier [%s] - Job name: %s - Value >= %d - Min done: %d',
                barrier_name, job_name, min_value, min_done)

    # Finding the last time that particular barrier was cleared
    # If we detect that the timestamp increases, we can assume the barrier was cleared and then recreated.
    last_cleared = int(buffer.redis.hget(__BARRIER_LAST_CLEARED__, barrier_name) or time.time())
    last_status = time.time()

    while True:
        print_status = False
        if time.time() > (last_status + 30):
            print_status = True
            last_status = time.time()
        if can_proceed_through_barrier(buffer, barrier_name, job_name, min_value, min_done, last_cleared, print_status):
            break
        time.sleep(1.)
    LOGGER.info('Done waiting for barrier [%s]. Job name: %s', barrier_name, job_name)

def can_proceed_through_barrier(buffer, barrier_name, job_name=None, min_value=1, min_done=1, last_cleared=0,
                                print_status=False):
    """ Indicates if the barrier is cleared (or the wait condition is met).
        :param buffer: An instance of the memory buffer.
        :param barrier_name: The name of the barrier. (e.g. 'train' or 'eval')
        :param job_name: If set, only looks at workers performing the specific job (e.g. 'learner', 'actor')
        :param min_value: The min value required for the worker to be considered done.
        :param min_done: The min number of done worker to stop waiting for the barrier.
        :param last_cleared: Optional. The last known timestamp when the barrier was cleared
        :param print_status: Boolean that indicates to print the status of the barrier.
        :return: A boolean indicating if we can proceed through the barrier
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer

        i.e. if job_name is not set, we must wait until clear_barrier() is called before we can proceed.
             otherwise, we wait until 1) there are no more pending workers, and
                                      2) min_done workers of type 'job_name' have set a value of at least min_value.
                        before proceeding.
    """
    nb_done, nb_pending = workers_on_barrier(buffer, barrier_name, job_name, min_value, print_status)
    if job_name is not None and (nb_pending + nb_done) > 0 and nb_done >= min_done:
        LOGGER.info('[OK] Barrier: %s - Job: %s - Pending: %d - Done: %d - Minimum: %d',
                    barrier_name, job_name, nb_pending, nb_done, min_done)
        return True
    if job_name is None and not buffer.redis.exists(__BARRIER__ % barrier_name):
        LOGGER.info('[OK] Barrier: %s - Job: None - Barrier has been cleared', barrier_name)
        return True

    # Checking if the barrier was last cleared after the last known last_cleared
    # This might indicate that the barrier was cleared and then recreated.
    if last_cleared > 0:
        new_last_cleared = int(buffer.redis.hget(__BARRIER_LAST_CLEARED__, barrier_name) or 0)
        if new_last_cleared > last_cleared:
            LOGGER.info('[OK] Barrier: %s - Last Cleared: %s - Cleared at: %s',
                        barrier_name, last_cleared, new_last_cleared)
            return True

    # Otherwise, we still need to wait
    return False

def workers_on_barrier(buffer, barrier_name, job_name=None, min_value=1, print_status=False):
    """ Returns the number of completed and incomplete workers on the barrier
        :param buffer: An instance of the memory buffer.
        :param barrier_name: The name of the barrier. (e.g. 'train' or 'eval')
        :param job_name: If set, only looks at workers performing the specific job (e.g. 'learner', 'actor')
        :param min_value: The min value required for the worker to be considered done.
        :param print_status: Boolean that indicates to print the list of workers on the barrier.
        :return: A tuple with 1) the number of completed workers (waiting for barrier),
                              2) the number of incomplete workers (still working)
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    done, pending = [], []
    key_values = {key.decode('utf-8'): value.decode('utf-8')
                  for key, value in buffer.redis.hgetall(__BARRIER__ % barrier_name).items()}
    for key, value in key_values.items():
        if job_name is None or job_name in key:
            if int(value) >= min_value:
                done += [key]
            else:
                pending += [key]
    if print_status:
        LOGGER.info('[Status] Barrier: %s - Done (%d): %s - Pending (%d): %s',
                    barrier_name, len(done), done, len(pending), pending)
    return len(done), len(pending)
