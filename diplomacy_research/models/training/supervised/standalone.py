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
""" Supervised - Standalone training
    - Class responsible for training a model in a non-distributed setting
"""
from collections import OrderedDict
import logging
import sys
import time
from diplomacy_research.models.datasets.supervised_dataset import TrainingMode
from diplomacy_research.models.training.supervised.common import build_model_and_dataset, build_summaries, \
    build_config_proto, decay_rates, save_model, run_gradient_step, run_profile_step, run_next_decoded_training_batch, \
    run_next_decoded_validation_batch, load_performance_data, can_stop_due_to_early_stopping
from diplomacy_research.models.training.supervised.statistics import display_training_stats, display_validation_stats, \
    display_final_validation_stats
from diplomacy_research.settings import SESSION_RUN_TIMEOUT

# Constants
LOGGER = logging.getLogger(__name__)

def start_standalone_training(trainer):
    """ Starts training in standalone mode.
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, tf_debug

    # Builds the model and the dataset
    build_model_and_dataset(trainer)
    build_summaries(trainer)
    build_config_proto(trainer)

    # Starts the session and training loop
    try:
        saver = tf.train.Saver(max_to_keep=3, restore_sequentially=True, pad_step_number=True)
        trainer.history_saver = tf.train.Saver(max_to_keep=999)  # To save historical checkpoints in a sep. directory
        nb_total_steps_per_epoch = trainer.supervised_dataset.nb_total_steps_per_epoch

        # Starting monitored training session and restoring model from checkpoint
        with tf.train.MonitoredTrainingSession(checkpoint_dir=trainer.flags.save_dir,
                                               scaffold=tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                                                          saver=saver),
                                               log_step_count_steps=100 * nb_total_steps_per_epoch,
                                               hooks=trainer.hooks,
                                               chief_only_hooks=trainer.chief_only_hooks,
                                               config=tf.ConfigProto(**trainer.config_proto),
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as sess:

            # Wrapping in a session debugger
            if trainer.flags.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Saving session
            trainer.model.sess = sess
            LOGGER.info('Session successfully started.')

            # Restoring dataset status from status file
            # If in debug (tensorflow, batch) mode, starting training with TRAINING dataset
            if trainer.flags.debug or trainer.flags.debug_batch:
                trainer.supervised_dataset.start_training_mode(sess)
            else:
                trainer.supervised_dataset.load_status()

            # Getting status from dataset
            trainer.progress = trainer.supervised_dataset.get_progress()
            load_performance_data(trainer)

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
        if trainer.model.sess is not None and trainer.model.sess._sess is not None:                                     # pylint: disable=protected-access
            raise error
    else:
        LOGGER.info('We have successfully trained the supervised model.')

def run_training_epoch(trainer, sess):
    """ Runs a training epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, ALL_ADVICE

    # Initializing the dataset to use the training set
    if trainer.supervised_dataset.is_done:
        trainer.supervised_dataset.start_training_mode(sess)
    elif not trainer.supervised_dataset.iterator_initialized:
        trainer.supervised_dataset.initialize_iterator(sess)

    # If another dataset was loaded from load_status(), skipping Training
    if trainer.supervised_dataset.training_mode != TrainingMode.TRAINING:
        return

    # For the first epoch, we only want to run validation to get an idea of performance with random weights
    nb_epochs_completed, _ = trainer.supervised_dataset.get_progress()
    if not nb_epochs_completed \
            and not trainer.flags.debug \
            and not trainer.flags.debug_batch \
            and not trainer.flags.profile:
        LOGGER.info('Only running validation for the first epoch (to get pre-training performance).')
        trainer.supervised_dataset.mark_as_done()

    # Variables for stats
    loss = 0.
    trainer.status_time = int(time.time())
    trainer.step_last_status = 0

    # Method to run
    method_to_run = run_profile_step if trainer.flags.profile else run_gradient_step

    # Running epoch
    while True:
        current_time = int(time.time())

        # Checking if we need to stop training, or are done with the current epoch
        if hasattr(sess, 'should_stop') and sess.should_stop():
            trainer.supervised_dataset.close()
            break
        if trainer.supervised_dataset.is_done:
            break

        # OutOfRangeError is thrown when we reach the end of the dataset
        try:
            loss = method_to_run(trainer, sess)
        except tf.errors.OutOfRangeError:
            trainer.supervised_dataset.mark_as_done()
        except tf.errors.DeadlineExceededError:
            LOGGER.warning('%s took more than %d ms to run. Timeout error.', method_to_run, SESSION_RUN_TIMEOUT)

        # Printing status
        if (current_time - trainer.status_time) > trainer.status_every or trainer.supervised_dataset.is_done:

            # Updating stats
            elasped_time = current_time - trainer.status_time
            elapsed_steps = trainer.supervised_dataset.steps_in_current_mode - trainer.step_last_status
            trainer.status_time = current_time
            trainer.step_last_status = trainer.supervised_dataset.steps_in_current_mode
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

            # Exiting debug (batch) mode
            if trainer.flags.debug_batch and loss < 2e-3:
                LOGGER.info('Reached a loss of 0.001 - Successfully converged. Exiting')
                trainer.supervised_dataset.close()
                sys.exit(0)

            # Exiting profile mode
            if trainer.flags.profile:
                trainer.profiler.advise(options=ALL_ADVICE)
                LOGGER.info('Saved profiling information. Exiting')
                trainer.supervised_dataset.close()
                sys.exit(0)

def run_validation_epoch(trainer, sess):
    """ Runs a validation epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf

    # Initializing the dataset to use the training set
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

    # Post-processing detailed results
    detailed_results = trainer.model.post_process_results(detailed_results)

    # Printing final validation status, and saving model
    display_final_validation_stats(trainer, sess, results, detailed_results, aggregated=False)
    save_model(trainer, sess)
    trainer.supervised_dataset.mark_as_done()

    # ---- Done Validation Epoch -----
    print('-' * 80)
