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
""" Supervised - Common ops
    - Class responsible for implementing build operations (common among standalone and distributed training)
"""
from enum import Enum
import logging
import os
import pickle
import shutil
import sys
from threading import Thread
from diplomacy_research.models.base_model import BaseModel
from diplomacy_research.models.datasets.supervised_dataset import SupervisedDataset
from diplomacy_research.models.training.memory_buffer import MemoryBuffer
from diplomacy_research.utils.checkpoint import build_saved_model
from diplomacy_research.utils.cluster import kill_processes_using_port, is_port_opened

# Constants
LOGGER = logging.getLogger(__name__)

class ProfilingMode(Enum):
    """ Enumeration of profiling modes """
    GRAPH = 'graph'
    OPERATION = 'op'
    SCOPE = 'scope'
    DEVICE = 'device'

def build_summaries(trainer):
    """ Builds the fields required to log stats in TensorBoard
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, get_placeholder
    escape = lambda str: str.replace('[', '.').replace(']', '.')

    # Make sure we already have a model
    assert isinstance(trainer.model, BaseModel), 'A model is required to build the summaries'

    # Retrieving evaluation tags
    evaluation_tags = trainer.model.get_evaluation_tags()
    assert len(evaluation_tags) == trainer.model.nb_evaluation_loops, \
        'Expected %d evaluation tags, Got %d.' % (trainer.model.nb_evaluation_loops, len(evaluation_tags))

    # Creating placeholders and summaries
    for eval_loop_ix in range(trainer.model.nb_evaluation_loops):
        for tag_name in evaluation_tags[eval_loop_ix]:
            if tag_name not in trainer.placeholders:
                trainer.placeholders[tag_name] = get_placeholder(escape(tag_name), tf.float32, (), for_summary=True)
                trainer.summaries[tag_name] = tf.summary.scalar(escape(tag_name), trainer.placeholders[tag_name])

    # Creating merge_op
    for eval_loop_ix in range(trainer.model.nb_evaluation_loops):
        merge_summaries = []
        for tag_name in evaluation_tags[eval_loop_ix]:
            merge_summaries += [trainer.summaries[tag_name]]
        trainer.merge_ops[eval_loop_ix] = tf.summary.merge(merge_summaries)

    # Creating writers
    os.makedirs(os.path.join(trainer.flags.save_dir, 'summary'), exist_ok=True)
    trainer.writers['train'] = tf.summary.FileWriter(os.path.join(trainer.flags.save_dir, 'summary', 'train'))
    trainer.writers['valid'] = tf.summary.FileWriter(os.path.join(trainer.flags.save_dir, 'summary', 'valid'))
    trainer.writers['aggreg'] = tf.summary.FileWriter(os.path.join(trainer.flags.save_dir, 'summary', 'aggreg'))

def build_model_and_dataset(trainer):
    """ Builds the model and the dataset
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf

    trainer.supervised_dataset = SupervisedDataset(batch_size=trainer.flags.batch_size,
                                                   dataset_builder=trainer.dataset_builder,
                                                   checkpoint_dir=trainer.flags.save_dir,
                                                   cluster_config=trainer.cluster_config,
                                                   debug_batch=trainer.flags.debug_batch,
                                                   do_infinite_training=trainer.flags.use_verbs,
                                                   perc_epoch_for_training=trainer.flags.perc_epoch_for_training)

    # Policy Model
    assert trainer.policy_constructor is not None, 'A policy constructor is required to build the model.'
    trainer.model = trainer.policy_constructor(dataset=trainer.supervised_dataset,
                                               hparams=trainer.hparams)

    # Value Model
    if trainer.value_constructor:
        trainer.model = trainer.value_constructor(parent_model=trainer.model,
                                                  dataset=trainer.supervised_dataset,
                                                  hparams=trainer.hparams)

    # Draw Model
    if trainer.draw_constructor:
        trainer.model = trainer.draw_constructor(parent_model=trainer.model,
                                                 dataset=trainer.supervised_dataset,
                                                 hparams=trainer.hparams)

    # Finalizing and validation
    trainer.model.finalize_build()
    trainer.model.validate()

    # Building a list of cost, with the scope to optimize
    cost_and_scope = []

    # Sum of individual losses
    scope = ['policy']
    ignored_scope = None
    total_loss = trainer.flags.policy_coeff * trainer.model.outputs['policy_loss']
    if trainer.value_constructor:
        scope += ['value']
        total_loss += trainer.flags.value_coeff * trainer.model.outputs['value_loss']
    if trainer.draw_constructor:
        scope += ['draw']
        total_loss += trainer.flags.draw_coeff * trainer.model.outputs['draw_loss']
    total_loss = tf.identity(total_loss, name='total_loss')
    cost_and_scope += [(total_loss, scope, ignored_scope)]

    # Creating optimizer
    optimizer_op = trainer.model.create_optimizer_op(cost_and_scope, max_gradient_norm=trainer.flags.max_gradient_norm)
    trainer.model.add_output('optimizer_op', optimizer_op)

    # Installing hooks
    session_hook = trainer.supervised_dataset.make_session_run_hook()
    if session_hook is not None:
        trainer.hooks += [session_hook]

def build_train_server(trainer):
    """ Builds the Tensorflow tf.train.Server
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    if not trainer.cluster_config:
        return

    from diplomacy_research.utils.tensorflow import tf
    task_address = trainer.cluster_config.cluster_spec[trainer.cluster_config.job_name][trainer.cluster_config.task_id]

    # Making port is not used by another process
    task_port = int(task_address.split(':')[1])
    LOGGER.info('Killing any processes that have port %d opened.', task_port)
    kill_processes_using_port(task_port)

    # Starting server
    LOGGER.info('Starting server with task id %d - Address: %s', trainer.cluster_config.task_id, task_address)
    LOGGER.info('Checking if port %d is already opened: %s', task_port, str(is_port_opened(task_port)))
    trainer.server = tf.train.Server(tf.train.ClusterSpec(trainer.cluster_config.cluster_spec),
                                     job_name=trainer.cluster_config.job_name,
                                     task_index=trainer.cluster_config.task_id,
                                     protocol=trainer.cluster_config.protocol)
    LOGGER.info('Server successfully started. Trying to contact other nodes...')

def build_config_proto(trainer):
    """ Builds the session config proto
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    gpu_options = tf.GPUOptions()
    if trainer.flags.allow_gpu_growth:
        LOGGER.info('"allow_gpu_growth" flag set. GPU memory will not be pre-allocated and will grow as needed.')
        gpu_options = tf.GPUOptions(allow_growth=True)
    trainer.config_proto = {'allow_soft_placement': True, 'gpu_options': gpu_options}
    if trainer.cluster_config:
        trainer.config_proto['device_filters'] = ['/job:ps', '/job:worker/task:%d' % trainer.cluster_config.task_id]
    if trainer.flags.profile:
        trainer.config_proto['log_device_placement'] = True
    if trainer.flags.use_xla:
        LOGGER.info('Using XLA to compile the graph.')
        graph_options = tf.GraphOptions()
        graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1                                     # pylint: disable=no-member
        trainer.config_proto['graph_options'] = graph_options

def build_memory_buffer(trainer):
    """ Builds the memory buffer (barrier) and connects to it
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    if trainer.cluster_config:
        trainer.memory_buffer = MemoryBuffer(trainer.cluster_config, trainer.hparams)

def save_model(trainer, sess, start_of_epoch=False):
    """ Saves the current graph to a saved model checkpoint on disk (using a separate thread)
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session
        :param start_of_epoch: Boolean that indicates that we are saving the model at the start of a new epoch.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    if not isinstance(sess, tf.Session):
        trainer.run_func_without_hooks(sess, lambda _sess: save_model(trainer, _sess))
        return

    assert isinstance(sess, tf.Session), 'Session must be a raw TensorFlow session'
    version_id = int(trainer.progress[0]) + (0 if start_of_epoch else 1)
    output_dir = os.path.join(trainer.flags.save_dir, 'history')
    graph = tf.get_default_graph()
    model_thread = Thread(target=build_saved_model,
                          kwargs={'saved_model_dir': output_dir,
                                  'version_id': version_id,
                                  'signature': trainer.signature,
                                  'proto_fields': trainer.dataset_builder.get_proto_fields(),
                                  'graph': graph,
                                  'session': sess,
                                  'history_saver': trainer.history_saver},
                          daemon=True)
    model_thread.start()

def decay_rates(trainer, sess):
    """ Decays the learning rate exponentially
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :return: Nothing
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    nb_completed_epochs, current_progress = trainer.progress
    total_progress = nb_completed_epochs + current_progress

    # Decaying learning rate exponentially
    if hasattr(trainer.model, 'decay_learning_rate'):
        new_learning_rate = trainer.flags.learning_rate * trainer.flags.lr_decay_factor ** total_progress
        trainer.run_without_hooks(sess,
                                  trainer.model.decay_learning_rate,
                                  feed_dict={trainer.model.placeholders['learning_rate']: new_learning_rate})
        trainer.learning_rate = new_learning_rate

def run_gradient_step(trainer, sess):
    """ Runs a regular gradient step in the model
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :return: The step loss
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf
    try:
        session_args = trainer.model.get_session_args()
        fetches = trainer.run_session(sess, session_args)
        loss = sum([fetch_value for fetch_name, fetch_value in fetches.items() if 'loss' in fetch_name])
        trainer.supervised_dataset.take_local_step()
        trainer.nb_oom_steps = 0

    # Out of Memory - Skipping batch unless we had 50 consecutive OOM batches
    except tf.errors.ResourceExhaustedError as err:
        trainer.nb_oom_steps += 1
        LOGGER.warning('run_gradient_step() triggered an ResourceExhaustedError / Out of Memory.')
        if trainer.nb_oom_steps >= 50:
            raise err
        loss = 0.
        trainer.supervised_dataset.take_local_step()

    return loss

def run_profile_step(trainer, sess):
    """ Runs a profiling step in the model and starts the profiler if needed
        :param trainer: A supervised trainer instance.
        :param sess: The TensorFlow session.
        :return: The step loss and the step number.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    from diplomacy_research.utils.tensorflow import tf, timeline

    # Making sure a valid profiling mode was provided
    assert trainer.flags.profile in [mode.value for mode in ProfilingMode], 'Invalid profiling mode detected.'

    # We can exit here if we are only profiling device placement (no need to run a step)
    if trainer.flags.profile == ProfilingMode.DEVICE.value:
        LOGGER.info('Done profiling in "device" mode. Exiting.')
        trainer.supervised_dataset.close()
        sys.exit(0)

    # Starting profiler and creating output directory
    if not trainer.profiler:
        LOGGER.info('Profiling model in "%s" mode', trainer.flags.profile)
        trainer.profiler = tf.profiler.Profiler(sess.graph)
        if os.path.exists(os.path.join(trainer.flags.save_dir, 'profile')):
            shutil.rmtree(os.path.join(trainer.flags.save_dir, 'profile'))
        os.mkdir(os.path.join(trainer.flags.save_dir, 'profile'))

    # Running a profiling step and saving it
    trainer.nb_profile_steps += 1
    run_meta = tf.RunMetadata()
    session_args = trainer.model.get_session_args()
    fetches = trainer.run_session(sess, session_args, run_metadata=run_meta)
    loss = sum([fetch_value for fetch_name, fetch_value in fetches.items() if 'loss' in fetch_name])
    trainer.supervised_dataset.take_local_step()

    # Profiling the parameters
    if trainer.flags.profile == ProfilingMode.SCOPE.value:
        trainer.profiler.add_step(trainer.nb_profile_steps, run_meta)
        trainer.profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()))

    # Profiling the op timing
    if trainer.flags.profile == ProfilingMode.OPERATION.value:
        trainer.profiler.add_step(trainer.nb_profile_steps, run_meta)
        opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
        trainer.profiler.profile_operations(options=opts)

    # Profile the graph with a timeline
    if trainer.flags.profile == ProfilingMode.GRAPH.value:
        output_file_path = os.path.join(trainer.flags.save_dir,
                                        'profile',
                                        'timeline_%d.json' % trainer.nb_profile_steps)
        LOGGER.info('Writing timeline to %s', output_file_path)
        fetched_timeline = timeline.Timeline(run_meta.step_stats)       # pylint: disable=no-member
        chrome_trace = fetched_timeline.generate_chrome_trace_format(run_meta)
        with open(output_file_path, 'w') as output_file:
            output_file.write(chrome_trace)

    # Returning loss
    return loss

def run_next_decoded_training_batch(trainer, sess):
    """ Runs the next training batch and decodes the results
        :param trainer: A supervised trainer instance.
        :param sess: The Tensorflow session
        :return: A tuple of 1) The evaluation_results ordered dictionary from the evaluate() method
                            2) The detailed evaluation results ordered dictionary from the evaluate() method
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    return run_next_decoded_batch(trainer, sess, is_training=True)

def run_next_decoded_validation_batch(trainer, sess, eval_loop_ix):
    """ Runs the next validation batch and decodes the results
        :param trainer: A supervised trainer instance.
        :param sess: The Tensorflow session
        :param eval_loop_ix: The index of the validation loop (0-index) if the model supports multiple passes
                             over the validation set
        :return: A tuple of 1) The evaluation_results ordered dictionary from the evaluate() method
                            2) The detailed evaluation results ordered dictionary from the evaluate() method
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    return run_next_decoded_batch(trainer, sess, is_training=False, eval_loop_ix=eval_loop_ix)

def run_next_decoded_batch(trainer, sess, is_training, eval_loop_ix=None):
    """ Runs the next batch and decodes the results
        :param trainer: A supervised trainer instance.
        :param sess: The Tensorflow session
        :param is_training: Boolean that indicates if we are in the training or validation set.
        :param eval_loop_ix: The index of the validation loop (0-index) if the model supports multiple passes
                             over the validation set
        :return: A tuple of 1) The evaluation_results ordered dictionary from the evaluate() method
                            2) The detailed evaluation results ordered dictionary from the evaluate() method
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    session_args = trainer.model.get_session_args(decode=True, eval_loop_ix=eval_loop_ix)
    fetches = trainer.run_func_without_hooks(sess, lambda _sess: trainer.run_session(_sess, session_args))
    decoded_results = trainer.model.decode(**fetches)
    results, detailed_results = trainer.model.evaluate(decoded_results,
                                                       session_args.get('feed_dict', {}),
                                                       eval_loop_ix=-1 if is_training else eval_loop_ix,
                                                       incl_detailed=not is_training)
    trainer.supervised_dataset.take_local_step()
    return results, detailed_results

def load_performance_data(trainer):
    """ Loads performance data from disk (to detect early stopping)
        :param trainer: A supervised trainer instance.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    performance_filename = os.path.join(trainer.flags.save_dir, 'performance.pkl')
    if os.path.exists(performance_filename):
        with open(performance_filename, 'rb') as performance:
            trainer.performance = pickle.load(performance)

def has_already_early_stopped(trainer):
    """ Determines if training has already stopped due to early stopping
        :param trainer: A supervised trainer instance.
        :return: Boolean that indicates if training has already stopped due to early stopping
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    early_stopping_file = os.path.join(trainer.flags.save_dir, 'stopped_early.txt')
    return os.path.exists(early_stopping_file)

def can_stop_due_to_early_stopping(trainer):
    """ Determines if we can stop training due to early stopping
        :param trainer: A supervised trainer instance.
        :return: Boolean that indicates if we can stop training due to early stopping.
        :type trainer: diplomacy_research.models.training.supervised.trainer.SupervisedTrainer
    """
    if has_already_early_stopped(trainer):
        LOGGER.info('Early stopping file detected on disk. Stopping early.')
        return True
    if not trainer.flags.early_stopping_stop_after:
        return False
    load_performance_data(trainer)

    # Finding the latest model who performed the best on any threshold
    latest_epoch = 0
    current_epoch = int(trainer.progress[0]) + 1
    for tag_type, tag_name in trainer.model.get_early_stopping_tags():
        best_value = float('-inf') if tag_type == 'max' else float('inf')
        best_epoch = -1

        # Tag not found in performance data - Skipping
        if tag_name not in trainer.performance:
            LOGGER.warning('Unable to find tag %s in performance data for early stopping.', tag_name)
            continue

        # Finding best epoch
        for epoch_ix in range(1, current_epoch + 1):
            if epoch_ix not in trainer.performance[tag_name]:
                continue
            if tag_type == 'max' and trainer.performance[tag_name][epoch_ix] > (1.005 * best_value):
                best_value = trainer.performance[tag_name][epoch_ix]
                best_epoch = epoch_ix
            elif tag_type == 'min' and trainer.performance[tag_name][epoch_ix] < (0.995 * best_value):
                best_value = trainer.performance[tag_name][epoch_ix]
                best_epoch = epoch_ix

        # Setting latest epoch
        LOGGER.info('Best epoch for "%s" is %d', tag_name, best_epoch)
        latest_epoch = max(latest_epoch, best_epoch)

    # Stopping if we did 'stop_after' epochs since the latest_epoch
    # If stopping early, leaving a file on disk to avoid continuing training if training is restarted
    can_stop_early = bool((current_epoch - latest_epoch) >= trainer.flags.early_stopping_stop_after)
    if can_stop_early:
        open(os.path.join(trainer.flags.save_dir, 'stopped_early.txt'), 'a').close()
    return can_stop_early
