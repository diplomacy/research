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
""" Supervised - Statistics
    - Class responsible for computing statistics and printing them during training
"""
import datetime
import logging
import os
import pickle
import time
import numpy as np
from diplomacy_research.models.policy.base_policy_model import StatsKey

# Constants
LOGGER = logging.getLogger(__name__)

def _get_weighted_average(results):
    """ Computes the weighted average results
        :param results: An ordered dictionary with result_name as key and (weight, value) as value (or list of (w, v))
        :return: A dictionary with result_name as key, and the weighted value as value
    """
    weighted_average = {}
    for result_name, result_value in results.items():
        total_weight, total = 0, 0
        if not isinstance(result_value, list):
            result_value = [result_value]
        for weight, result in result_value:
            total_weight += weight
            total += weight * result
        weighted_average[result_name] = total / total_weight if total_weight else -1
    return weighted_average

def display_training_stats(trainer, sess, results, epoch_eta):
    """ Displays intermediary stats during a training epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :param results: An ordered dictionary with result_name as key and (weight, value) as value
        :param epoch_eta: An estimation in seconds of the time required to complete an epoch.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    nb_completed_epochs, current_progress = trainer.progress
    current_epoch = int(nb_completed_epochs) + 1

    # 1) Compute the weighted averages of the results (to adjust for different nb of items per batch)
    weighted_average = _get_weighted_average(results)

    # 2) Printing message
    current_time = int(time.time())
    running_time = datetime.timedelta(seconds=(current_time - trainer.starting_time))
    epoch_eta = datetime.timedelta(seconds=epoch_eta)

    message = []
    message += ['Epoch {:3} ({:.2f}%)'.format(current_epoch, 100. * current_progress)]
    message += ['Step {:09}'.format(trainer.supervised_dataset.steps_in_current_mode)]
    for result_name in results:
        message += ['{} {:.2f}'.format(result_name, weighted_average[result_name])]
    message += ['Learning {:.2e}'.format(trainer.learning_rate)]
    message += ['Time/Epoch {}'.format(epoch_eta)]
    message += ['Time {}'.format(running_time)]
    print(' | '.join(message))

    # We can exit here if we are a non-chief in a distributed setting
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return

    # 3) Save training summary to disk
    feed_dict = {}
    current_tags = trainer.model.get_evaluation_tags()[0]
    remaining_tags = set(current_tags)

    for result_name in results:
        if result_name not in remaining_tags:
            continue
        remaining_tags.remove(result_name)
        feed_dict[trainer.placeholders[result_name]] = weighted_average[result_name]

    if not remaining_tags:
        global_step, summary_op = trainer.run_without_hooks(sess,
                                                            [trainer.model.global_step, trainer.merge_ops[0]],
                                                            feed_dict=feed_dict)
        trainer.writers['train'].add_summary(summary_op, global_step)
        trainer.writers['train'].flush()
    else:
        LOGGER.warning('Not writing validation summary. Missing tags: "%s"', remaining_tags)

def display_validation_stats(trainer, sess, results, batch_ix, eval_loop_ix):
    """ Displays intermediary stats during a training epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :param results: An ordered dictionary with result_name as key and (weight, value) as value
        :param batch_ix: The current validation batch index.
        :param eval_loop_ix: The current validation loop ix (i.e. [0, self.nb_evaluation_loops])
        :return: Nothing
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    nb_epochs_completed, current_progress = trainer.progress
    current_epoch = int(nb_epochs_completed) + 1

    # 1) Compute the weighted averages of the results (to adjust for different nb of items per batch)
    weighted_average = _get_weighted_average(results)

    # 2) Printing message
    message = []
    message += ['Validation - Epoch {:3} ({:.2f}%)'.format(current_epoch, 100. * current_progress)]
    for result_name in results:
        message += ['{} {:.2f}'.format(result_name, weighted_average[result_name])]
    print(' | '.join(message))

    # We can exit here if we are a non-chief in a distributed setting
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return

    # 3) Save validation summary to disk
    feed_dict = {}
    current_tags = trainer.model.get_evaluation_tags()[eval_loop_ix]
    remaining_tags = set(current_tags)

    for result_name in results:
        if result_name not in remaining_tags:
            continue
        remaining_tags.remove(result_name)
        feed_dict[trainer.placeholders[result_name]] = weighted_average[result_name]

    if not remaining_tags:
        global_step, summary_op = trainer.run_without_hooks(sess,
                                                            [trainer.model.global_step,
                                                             trainer.merge_ops[eval_loop_ix]],
                                                            feed_dict=feed_dict)
        trainer.writers['valid'].add_summary(summary_op, global_step + batch_ix)
        trainer.writers['valid'].flush()
    else:
        LOGGER.warning('Not writing validation summary. Missing tags: "%s"', remaining_tags)

def display_final_validation_stats(trainer, sess, results, detailed_results, aggregated=False):
    """ Display final results at the end of a validation epoch
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :param results: An ordered dict containing a list of (weight, results) obtained during the epoch.
        :param detailed_results: An ordered dict containing a list of all details evaluation results.
        :param aggregated: Boolean that indicates if the results were aggregated from different workers.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    current_epoch = int(trainer.progress[0]) + 1

    # --- Utility functions ---
    def weighted_average(items):
        """ Computes the weighted average """
        total_weight, total = 0, 0
        for weight, item in items:
            total_weight += weight
            total += weight * item
        return total / total_weight if total_weight else -1

    def min_item(items):
        """ Computes the minimum item """
        _, items = zip(*items)
        return -1 if not items else np.min(items)

    def max_item(items):
        """ Computes the maximum item """
        _, items = zip(*items)
        return -1 if not items else np.max(items)

    display_modes = [('Average', weighted_average),
                     ('Minimum', min_item),
                     ('Maximum', max_item),
                     ('Count', len)]

    # 1) Displaying the Average, Minimum, and Maximum for each result
    for display_mode_name, display_mode_func in display_modes:
        message = []
        message += ['Validation - Epoch {:3} (100.0%){}'.format(current_epoch, ' (All Workers)' if aggregated else '')]
        message += [display_mode_name]
        for result_name, result_value in results.items():
            message += ['{} {:.2f}'.format(result_name, display_mode_func(result_value))]
        print(' | '.join(message))

    # 2) Displaying the detailed statistics
    for result_name, result_value in detailed_results.items():
        # result_name could a be named_tuple
        # not printing those, at they are used only to compute joint accuracy
        if not result_value or not isinstance(result_name, str):
            continue
        message = []
        message += ['Validation - Epoch {:3} (100.0%){}'.format(current_epoch, ' (All Workers)' if aggregated else '')]
        mean_result_value = 100. * np.mean(result_value) if result_value else -1
        message += ['(Detailed)', '{} (Avg) {:.2f}'.format(result_name, mean_result_value)]
        message += ['{} (Count) {}'.format(result_name, len(result_value))]
        print(' | '.join(message))

    # We can exit here if we are a non-chief in a distributed setting
    if trainer.cluster_config and not trainer.cluster_config.is_chief:
        return

    # 3) Detecting if we need to write aggregated info to disk
    write_aggregated = False
    if aggregated and trainer.cluster_config and trainer.cluster_config.is_chief:
        write_aggregated = True
    elif not aggregated and not trainer.cluster_config:
        write_aggregated = True

    # 4) Saving validation / aggregated summary info to disk
    for eval_loop_ix in range(trainer.model.nb_evaluation_loops):
        feed_dict = {}
        current_tags = trainer.model.get_evaluation_tags()[eval_loop_ix]
        _, early_stopping_tag_names = zip(*trainer.model.get_early_stopping_tags())
        remaining_tags = set(current_tags)
        nb_valid_steps_per_epoch = trainer.supervised_dataset.nb_validation_steps_per_epoch

        for result_name, result_value in results.items():
            if result_name not in remaining_tags:
                continue
            remaining_tags.remove(result_name)
            feed_dict[trainer.placeholders[result_name]] = weighted_average(result_value)
            if result_name in early_stopping_tag_names:
                trainer.performance.setdefault(result_name, {})[current_epoch] = weighted_average(result_value)

        if remaining_tags:
            LOGGER.warning('Not writing validation summary. Missing tags: "%s"', remaining_tags)
            continue

        global_step, summary_op = trainer.run_without_hooks(sess,
                                                            [trainer.model.global_step,
                                                             trainer.merge_ops[eval_loop_ix]],
                                                            feed_dict=feed_dict)
        if not aggregated:
            trainer.writers['valid'].add_summary(summary_op, global_step + nb_valid_steps_per_epoch)
            trainer.writers['valid'].flush()
        if write_aggregated:
            trainer.writers['aggreg'].add_summary(summary_op, global_step + nb_valid_steps_per_epoch)
            trainer.writers['aggreg'].flush()

    # 5) Writing detailed statistics on disk
    if write_aggregated:
        os.makedirs(os.path.join(trainer.flags.save_dir, 'statistics'), exist_ok=True)
        stats_filename = os.path.join(trainer.flags.save_dir, 'statistics', 'stats_%03d.pkl' % current_epoch)
        with open(stats_filename, 'wb') as file:
            pickle.dump({key: value for key, value in detailed_results.items() if isinstance(key, StatsKey)},
                        file, pickle.HIGHEST_PROTOCOL)
        LOGGER.info('Complete statistics for epoch are available at: %s', stats_filename)

    # 6) Writing performance data (for early stopping) on disk
    if write_aggregated:
        performance_filename = os.path.join(trainer.flags.save_dir, 'performance.pkl')
        with open(performance_filename, 'wb') as file:
            pickle.dump(trainer.performance, file, pickle.HIGHEST_PROTOCOL)
        LOGGER.info('Performance data are available at: %s', performance_filename)
