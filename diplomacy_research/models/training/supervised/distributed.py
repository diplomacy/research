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
""" Supervised - Distributed training
    - Class responsible for training a model in a distributed setting
"""
from collections import OrderedDict
import logging
import os
import pickle
import shutil
import time
from diplomacy_research.models.datasets.supervised_dataset import TrainingMode
from diplomacy_research.models.training.memory_buffer.memory_buffer import start_redis_server
from diplomacy_research.models.training.memory_buffer.barrier import clear_barrier, set_barrier_status, \
    workers_on_barrier, wait_for_barrier, can_proceed_through_barrier
from diplomacy_research.models.training.supervised.common import build_model_and_dataset, build_summaries, \
    build_config_proto, build_train_server, build_memory_buffer, decay_rates, save_model, run_gradient_step, \
    run_next_decoded_training_batch, run_next_decoded_validation_batch, load_performance_data, \
    can_stop_due_to_early_stopping
from diplomacy_research.models.training.supervised.statistics import display_training_stats, display_validation_stats, \
    display_final_validation_stats
from diplomacy_research.settings import SESSION_RUN_TIMEOUT

# Constants
LOGGER = logging.getLogger(__name__)
__TRAIN_BARRIER__ = 'supervised.train'
__EVAL_BARRIER__ = 'supervised.eval'


def start_distributed_training(trainer):
    """ Starts training in distributed mode.
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    # Making sure debug (tensorflow and batch) and profile are not set
    if trainer.flags.debug:
        raise RuntimeError('Debug (TensorFlow) mode is not supported in distributed mode.')
    if trainer.flags.debug_batch:
        raise RuntimeError('Debug (batch) mode is not supported in distributed mode.')
    if trainer.flags.profile:
        raise RuntimeError('Profile mode is not supported in distributed mode.')

    # Dispatching
    if trainer.cluster_config.job_name == 'ps':
        start_parameter_server(trainer)
    elif trainer.cluster_config.job_name == 'worker':
        start_worker(trainer)
    elif trainer.cluster_config.job_name == 'redis':
        start_redis_server(trainer, import_db=False)
    else:
        raise RuntimeError('Invalid configuration detected.')

def start_parameter_server(trainer):
    """ Starts a parameter server
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    build_train_server(trainer)
    LOGGER.info('Parameter server is now ready ...')
    trainer.server.join()

def start_worker(trainer):
    """ Starts a worker
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, SupervisedStartTrainingHook

    # Builds the model and the dataset
    build_model_and_dataset(trainer)
    build_summaries(trainer)
    build_config_proto(trainer)
    build_train_server(trainer)
    trainer.chief_only_hooks += [SupervisedStartTrainingHook(trainer)]

    # Builds the memory buffer, so we can use it as a barrier
    build_memory_buffer(trainer)

    # Adding SyncReplicas Hook
    if isinstance(trainer.model.optimizer, tf.train.SyncReplicasOptimizer):
        LOGGER.info('Detected SyncReplicasOptimizer. Automatically adding required hooks.')
        session_hook = trainer.model.optimizer.make_session_run_hook(trainer.cluster_config.is_chief)
        if session_hook is not None:
            trainer.hooks += [session_hook]

    # Starts the session and training loop
    try:
        saver = tf.train.Saver(max_to_keep=3, restore_sequentially=True, pad_step_number=True)
        trainer.history_saver = tf.train.Saver(max_to_keep=999)  # To save historical checkpoints in a sep. directory
        nb_total_steps_per_epoch = trainer.supervised_dataset.nb_total_steps_per_epoch

        # Clearing both barriers
        if trainer.cluster_config.is_chief:
            clear_barrier(trainer.memory_buffer, __TRAIN_BARRIER__, cleared_time=0)
            clear_barrier(trainer.memory_buffer, __EVAL_BARRIER__, cleared_time=0)

        # Starting monitored training session and restoring model from checkpoint
        with tf.train.MonitoredTrainingSession(master=trainer.server.target,
                                               is_chief=trainer.cluster_config.is_chief,
                                               checkpoint_dir=trainer.flags.save_dir,
                                               scaffold=tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                                                          saver=saver),
                                               log_step_count_steps=100 * nb_total_steps_per_epoch,
                                               hooks=trainer.hooks,
                                               chief_only_hooks=trainer.chief_only_hooks,
                                               config=tf.ConfigProto(**trainer.config_proto),
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as sess:
            # Saving session
            trainer.model.sess = sess
            LOGGER.info('Session successfully started.')

            # Restoring dataset status from status file
            trainer.supervised_dataset.load_status()
            load_performance_data(trainer)

            # Getting status from dataset
            trainer.progress = trainer.supervised_dataset.get_progress()

            # Running training
            while True:
                if hasattr(sess, 'should_stop') and sess.should_stop():
                    trainer.supervised_dataset.close()
                    break

                # Running training epoch, then validation epoch
                run_training_epoch(trainer, sess)
                run_validation_epoch(trainer, sess)

                # Stops if early stopping is triggered
                if can_stop_due_to_early_stopping(trainer):
                    LOGGER.info('Early stopping criteria triggered. Stopping training.')
                    break

    # CTRL-C closes the session to force a checkpoint
    # Tensorflow will throw a Runtime Error because the session is already closed.
    except RuntimeError as error:
        if trainer.model.sess is not None and trainer.model.sess._sess is not None:  # pylint: disable=protected-access
            raise error
    else:
        LOGGER.info('We have successfully trained the supervised model.')

def run_training_epoch(trainer, sess):
    """ Runs a training epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf

    # Initializing the dataset to use the training set
    if trainer.supervised_dataset.is_done:
        trainer.supervised_dataset.start_training_mode(sess)
    elif not trainer.supervised_dataset.iterator_initialized:
        trainer.supervised_dataset.initialize_iterator(sess)

    # If another dataset was loaded from load_status(), skipping Training
    if trainer.supervised_dataset.training_mode != TrainingMode.TRAINING:
        return

    # Indicating to the barrier that we are starting training
    if trainer.cluster_config.is_chief:
        clear_barrier(trainer.memory_buffer, __EVAL_BARRIER__)
    set_barrier_status(trainer.memory_buffer, __TRAIN_BARRIER__, 0)
    done, incomplete = workers_on_barrier(trainer.memory_buffer, __TRAIN_BARRIER__)
    LOGGER.info('Starting training - Barrier Status: Done %d - Incomplete: %d', done, incomplete)

    # For the first epoch, we only want to run validation to get an idea of performance with random weights
    nb_epochs_completed, _ = trainer.supervised_dataset.get_progress()
    if not nb_epochs_completed:
        LOGGER.info('Only running validation for the first epoch (to get pre-training performance).')
        trainer.supervised_dataset.mark_as_done()

    # Variable for stats and synchronization
    done_training = False
    done_time = 0
    nb_long_steps = 0
    nb_workers = trainer.cluster_config.count('worker')
    trainer.status_time = int(time.time())
    trainer.step_last_status = 0
    last_barrier_status = 0.

    # Running epoch
    while True:
        current_time = int(time.time())

        # For barriers, printing status every 30 secs
        print_barrier_status = False
        if time.time() > (last_barrier_status + 30):
            last_barrier_status = time.time()
            print_barrier_status = True

        # Checking if we need to stop training, or are done with the current epoch
        if hasattr(sess, 'should_stop') and sess.should_stop():
            trainer.supervised_dataset.close()
            break
        if trainer.supervised_dataset.is_done:
            if not done_training:
                set_barrier_status(trainer.memory_buffer, __TRAIN_BARRIER__, 1)
                done, incomplete = workers_on_barrier(trainer.memory_buffer, __TRAIN_BARRIER__)
                LOGGER.info('Waiting for barrier Status: Done %d - Incomplete: %d', done, incomplete)
                done_training = True
                done_time = time.time()

            # For the first epoch, everyone blocks until all workers have signaled the barrier
            # This is to prevent a worker from training before we first run the evaluation loop
            if not nb_epochs_completed:
                wait_for_barrier(trainer.memory_buffer, __TRAIN_BARRIER__, job_name='worker', min_done=nb_workers)
                break

            # Chief can break if everyone has marked as done
            if trainer.cluster_config.is_chief and  can_proceed_through_barrier(trainer.memory_buffer,
                                                                                __TRAIN_BARRIER__,
                                                                                job_name='worker',
                                                                                min_done=nb_workers,
                                                                                print_status=print_barrier_status):
                break

            # Others can only break when the barrier is cleared
            if can_proceed_through_barrier(trainer.memory_buffer,
                                           __TRAIN_BARRIER__,
                                           last_cleared=done_time,
                                           print_status=print_barrier_status):
                break

        # OutOfRangeError is thrown when we reach the end of the dataset
        try:
            run_gradient_step(trainer, sess)
        except tf.errors.OutOfRangeError:
            trainer.supervised_dataset.mark_as_done()
        except tf.errors.DeadlineExceededError:
            nb_long_steps += 1
            if nb_long_steps >= 5 or not done_training:
                LOGGER.warning('run_gradient_step took more than %d ms to run. Timeout error.', SESSION_RUN_TIMEOUT)

        # Printing status
        if (current_time - trainer.status_time) > trainer.status_every \
                or (trainer.supervised_dataset.is_done and not done_training):

            # Updating stats
            elasped_time = current_time - trainer.status_time
            elapsed_steps = trainer.supervised_dataset.steps_in_current_mode - trainer.step_last_status
            trainer.status_time = current_time
            trainer.step_last_status = trainer.supervised_dataset.steps_in_current_mode
            prev_nb_epochs_completed = trainer.progress[0]
            trainer.progress = trainer.supervised_dataset.get_progress()
            epoch_eta = 0
            if elapsed_steps > 0:
                epoch_eta = int(trainer.supervised_dataset.nb_total_steps_per_epoch * elasped_time / elapsed_steps)

            # Decaying rates
            decay_rates(trainer, sess)

            # Displaying status
            try:
                results, _ = run_next_decoded_training_batch(trainer, sess)
                display_training_stats(trainer, sess, results, epoch_eta)
            except tf.errors.OutOfRangeError:
                trainer.supervised_dataset.mark_as_done()

            # Saving dataset status to disk (to be able to resume)
            trainer.supervised_dataset.save_status()

            # Saving model in infinite training
            if trainer.supervised_dataset.do_infinite_training \
                    and trainer.cluster_config.is_chief \
                    and nb_epochs_completed > prev_nb_epochs_completed:
                save_model(trainer, sess, start_of_epoch=True)

def run_validation_epoch(trainer, sess):
    """ Runs a validation epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf

    # Initializing the dataset to use the validation set
    if trainer.supervised_dataset.training_mode != TrainingMode.VALIDATION:
        trainer.supervised_dataset.start_validation_mode(sess)
    elif not trainer.supervised_dataset.iterator_initialized:
        trainer.supervised_dataset.initialize_iterator(sess)

    # If there are no batches in the validation set, aborting
    nb_batches = trainer.supervised_dataset.nb_validation_steps_per_epoch
    if not nb_batches:
        return

    # Variables for stats
    results, detailed_results = OrderedDict(), OrderedDict()
    batch_results, batch_detailed_results = OrderedDict(), OrderedDict()
    trainer.progress = trainer.supervised_dataset.get_progress()

    # Indicating to the barrier that we are starting training
    if trainer.cluster_config.is_chief:
        clear_barrier(trainer.memory_buffer, __TRAIN_BARRIER__)
    set_barrier_status(trainer.memory_buffer, __EVAL_BARRIER__, 0)
    done, incomplete = workers_on_barrier(trainer.memory_buffer, __EVAL_BARRIER__)
    LOGGER.info('Starting validation - Barrier Status: Done %d - Incomplete: %d', done, incomplete)

    # For each loop in the validation set
    for eval_loop_ix in range(trainer.model.nb_evaluation_loops):
        last_status_perc = 0.

        # Making sure the dataset is initialized
        if not trainer.supervised_dataset.iterator_initialized or trainer.supervised_dataset.is_done:
            trainer.supervised_dataset.initialize_iterator(sess)

        # ---- Starting Validation Epoch -----
        print('-' * 80)

        # Running each batch sequentially
        for batch_ix in range(nb_batches):

            # Checking if we need to stop training, or are done with the current eval_loop_ix
            if hasattr(sess, 'should_stop') and sess.should_stop():
                trainer.supervised_dataset.close()
                break
            if trainer.supervised_dataset.is_done:
                break

            # Running single batch
            try:
                batch_results, batch_detailed_results = run_next_decoded_validation_batch(trainer, sess, eval_loop_ix)
                trainer.progress = trainer.progress[0], (batch_ix + 1) / max(1, nb_batches)

                # Storing batch results
                for result_name, result_value in batch_results.items():
                    results.setdefault(result_name, [])
                    if isinstance(result_value, list):
                        results[result_name] += result_value
                    else:
                        results[result_name] += [result_value]
                for result_name, result_value in batch_detailed_results.items():
                    assert isinstance(result_value, list), 'Detailed results must be a list.'
                    detailed_results.setdefault(result_name, [])
                    detailed_results[result_name] += result_value

            except tf.errors.OutOfRangeError:
                trainer.supervised_dataset.mark_as_done()
            except tf.errors.DeadlineExceededError:
                LOGGER.warning('Validation took more than %d ms to run. Timeout error.', SESSION_RUN_TIMEOUT)

            # Printing status every 10% completed
            current_perc_completed = (batch_ix + 1) / max(1., nb_batches)
            if current_perc_completed > last_status_perc + 0.10:
                last_status_perc = round(current_perc_completed, 1)
                display_validation_stats(trainer, sess, batch_results, batch_ix, eval_loop_ix)
                trainer.supervised_dataset.save_status()

    # Post-processing eval detailed results
    detailed_results = trainer.model.post_process_results(detailed_results)

    # Printing final validation status, and freezing graph
    display_final_validation_stats(trainer, sess, results, detailed_results, aggregated=False)
    if trainer.cluster_config.is_chief:
        save_model(trainer, sess)
    trainer.supervised_dataset.mark_as_done()

    # ---- Done Validation Epoch -----
    print('-' * 80)

    # Stopping barrier and dumping results to disk for chief to aggregate
    set_barrier_status(trainer.memory_buffer, __EVAL_BARRIER__, 1)
    save_results_to_disk(trainer, results, detailed_results)

    # Non-Chief - Wait for chief to do aggregation
    if not trainer.cluster_config.is_chief:
        done, incomplete = workers_on_barrier(trainer.memory_buffer, __EVAL_BARRIER__)
        LOGGER.info('Waiting for barrier Status: Done %d - Incomplete: %d', done, incomplete)
        wait_for_barrier(trainer.memory_buffer, __EVAL_BARRIER__)
        return

    # Chief - Performs aggregation across all workers
    aggregate_results(trainer, sess)
    print('-' * 80)

def save_results_to_disk(trainer, results, detailed_results):
    """ Save the validation epoch for each worker to disk
        :param trainer: A supervised trainer instance.
        :param results: An ordered dict containing a list of (weight, results) obtained during the epoch.
        :param detailed_results: An ordered dict containing a list of all details evaluation results.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    nb_tries = 0
    output_filename = 'results.%d.pkl' % trainer.cluster_config.task_id

    # Waiting for file on disk before continuing
    while not os.path.exists(output_filename) and nb_tries < 5:
        with open(os.path.join(trainer.flags.save_dir, 'validation', output_filename), 'wb') as file:
            pickle.dump({'results': results, 'detailed_results': detailed_results}, file, pickle.HIGHEST_PROTOCOL)
        time.sleep(5)
        nb_tries += 1

def aggregate_results(trainer, sess):
    """ Aggregating validation results from all workers
        :param trainer: A supervised trainer instance.
        :param sess: The Tensorflow session.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    results, detailed_results = OrderedDict(), OrderedDict()

    # Waiting for all workers to complete validation before aggregating
    done, incomplete = workers_on_barrier(trainer.memory_buffer, __EVAL_BARRIER__)
    LOGGER.info('Waiting for workers to aggregate results: Done %d - Incomplete: %d', done, incomplete)
    wait_for_barrier(trainer.memory_buffer, __EVAL_BARRIER__,
                     job_name='worker',
                     min_done=trainer.cluster_config.num_shards)

    # All workers are done, aggregating results
    for worker_id in range(trainer.cluster_config.num_shards):
        input_filename = 'results.%d.pkl' % worker_id
        input_filename = os.path.join(trainer.flags.save_dir, 'validation', input_filename)
        nb_tries = 0

        # Trying to access file for 60 seconds
        while nb_tries < 6:
            if os.path.exists(input_filename):
                break
            nb_tries += 1
            time.sleep(10)

        # File does not exist or is empty, Skipping
        if not os.path.exists(input_filename) or not os.path.getsize(input_filename):
            LOGGER.warning('Unable to find file %s. Skipping this worker.', input_filename)
            continue

        # Loading data and recording evaluation results
        try:
            with open(input_filename, 'rb') as worker_data:
                worker_data = pickle.load(worker_data)
            for result_name, result_value in worker_data['results'].items():
                results.setdefault(result_name, [])
                results[result_name] += result_value
            for result_name, result_value in worker_data['detailed_results'].items():
                detailed_results.setdefault(result_name, [])
                detailed_results[result_name] += result_value
        except EOFError:
            LOGGER.error('The worker file %s is corrupted. Skipping.', input_filename)

        # Deleting file on disk
        os.unlink(input_filename)

    # Deleting and recreating validation folder
    shutil.rmtree(os.path.join(trainer.flags.save_dir, 'validation'))
    os.mkdir(os.path.join(trainer.flags.save_dir, 'validation'))

    # Displaying aggregate data
    print('Aggregate data from %d workers' % trainer.cluster_config.num_shards)
    display_final_validation_stats(trainer, sess, results, detailed_results, aggregated=True)
