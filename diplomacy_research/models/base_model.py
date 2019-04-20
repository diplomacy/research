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
""" Base model
    - Contains the base model, with functions common to all models
"""
from collections import OrderedDict
from functools import reduce
import logging
from operator import mul
from diplomacy_research.utils.model import merge_complex_dicts
from diplomacy_research.settings import DEFAULT_SAVE_DIR, GIT_COMMIT_HASH

# Constants
LOGGER = logging.getLogger(__name__)

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    # Default settings
    args = [
        ('int', 'model_id', -1, 'The model ID.'),
        ('int', 'batch_size', 128, 'Training batch size.'),
        ('str', 'config', 'config.yaml', 'The configuration file to use to set hyparameters (in save_dir)'),
        ('str', 'config_section', '', 'Config section to load from the configuration file. Otherwise root.'),
        ('str', 'save_dir', DEFAULT_SAVE_DIR, 'Save directory'),
        ('bool', 'allow_gpu_growth', False, 'Boolean that indicates to not take the full GPU memory.'),
        ('str', 'gradient_checkpoint', '', 'One of "speed", "memory", "collection" to active gradient checkpointing.'),
        ('bool', 'sync_gradients', True, 'For distributed supervised training, uses SyncReplicasOptimizer'),
        ('bool', 'avg_gradients', False, 'For distributed rl training, uses AvgGradOptimizer'),
        ('bool', 'swap_memory', False, 'If set, reduces memory usage by storing gradients on CPU'),
        ('str', 'grad_aggregation', 'ADD_N', 'One of "ADD_N", "ACCUMULATE_N", "TREE". Gradient aggregation method'),
        ('int', 'nb_param_servers', 3, '(Distributed) The number of parameter servers to start. 0 for non-distributed'),
        ('int', 'min_nb_param_servers', 0, '(Distributed) Launches at least this number of PS to force partitioning.'),
        ('int', 'gpu_id', -1, 'The GPU id for this node. Values >= 100 indicate parameter servers.'),
        ('int', 'grpc_port', 2200, 'The starting port to open for distributed training'),
        ('bool', 'use_partitioner', False, 'Whether to use a partitioner for message and order embeddings'),
        ('bool', 'use_verbs', False, 'Use GRPC+verbs rather than just GRPC for distributed training.'),
        ('bool', 'use_xla', True, 'Use XLA compilation.'),
        ('str', 'profile', '', 'One of "op", "scope", "graph" to activate profiling.'),
        ('str', 'training_mode', 'supervised', 'The current training mode ("supervised" or "reinforcement").'),
        ('bool', 'debug', False, 'Boolean that indicates to load the TensorFlow debugger.'),
        ('bool', 'debug_batch', False, 'Boolean that indicates we want to overfit a mini-batch to debug the model.'),
        ('int', 'debug_nb_cpu_workers', 0, '(Distributed) If set, will start this nb of parallel cpu workers.')
    ]

    # Returning
    return args

class BaseModel():
    """ Base Model"""

    def __init__(self, parent_model, dataset, hparams):
        """ Initialization
            :param parent_model: A `base_model` to which we are adding features
            :param dataset: The dataset that is used to iterate over the data.
            :param hparams: A dictionary of hyper parameters with their values
            :type parent_model: BaseModel
            :type dataset: diplomacy_research.models.datasets.supervised_dataset.SupervisedDataset
            :type dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
        """
        from diplomacy_research.utils.tensorflow import tf
        assert dataset.can_support_iterator, 'The dataset must be able to support an iterator'
        self.placeholders = {}
        self.sess = None
        self.outputs = {}
        self.optimizer = None
        self.nb_optimizers = 0
        self.learning_rate = None
        self.decay_learning_rate = None
        self.build_finalized = False

        # Overriding from parent
        if parent_model:
            self.__dict__.update(parent_model.__dict__)

        # Setting from args
        self.parent_model = parent_model
        self.hparams = hparams
        self.proto_fields = dataset.dataset_builder.get_proto_fields()
        self.cluster_config = dataset.cluster_config
        self.iterator_resource = dataset.iterator._iterator_resource           # pylint: disable=protected-access
        self.features = dataset.output_features

        with tf.device(self.cluster_config.worker_device if self.cluster_config else None):
            self.global_step = tf.train.get_or_create_global_step()

    @property
    def nb_evaluation_loops(self):
        """ Contains the number of different evaluation tags we want to compute
            This also represent the number of loops we should do over the validation set
            Some model wants to calculate different statistics and require multiple pass to do that
        """
        return self.nb_parent_evaluation_loops + self._nb_evaluation_loops - 1

    @property
    def nb_parent_evaluation_loops(self):
        """ Contains the number of different evaluation tags we want to compute
            Note: This only includes models above ourself (self.parent_model) and exclude self._nb_evaluation_loops
        """
        return 1 if not self.parent_model else self.parent_model.nb_evaluation_loops

    @property
    def my_eval_loop_ixs(self):
        """ Contains the eval loop ix that this model uses for evaluation """
        return [0] + list(range(self.nb_parent_evaluation_loops,
                                self.nb_parent_evaluation_loops + self._nb_evaluation_loops - 1))

    def get_evaluation_tags(self):
        """ Returns a list of list of evaluation tags
            Note: There should be a list of tags for every evaluation loop
            e.g. [['Acc_1', 'Acc_5', 'Acc_Tokens'], ['Gr_1', 'Gr_5', 'Gr_Tokens']]
        """
        eval_tags = [] if not self.parent_model else self.parent_model.get_evaluation_tags()
        for _ in range(self.nb_evaluation_loops - len(eval_tags)):
            eval_tags.append([])
        for eval_loop_ix, tags in zip(self.my_eval_loop_ixs, self._evaluation_tags):
            eval_tags[eval_loop_ix] += tags
        return eval_tags

    def get_early_stopping_tags(self):
        """ List of tags to use to detect early stopping
            The tags are a tuple of 1) 'min' or 'max' and 2) the tag's name
            e.g. [('max', '[Gr]Acc_1'), ('min', '[TF]Perplexity')]
        """
        assert isinstance(self._early_stopping_tags, list), 'Expected early stopping tags to be a list of tags'
        early_stopping_tags = [] if not self.parent_model else self.parent_model.get_early_stopping_tags()
        for tag_type, tag_name in self._early_stopping_tags:
            assert tag_type in ('min', 'max'), 'The tag type must be "min" or "max". Got %s' % tag_type
            early_stopping_tags += [(tag_type, tag_name)]
        return early_stopping_tags

    def get_placeholders(self):
        """ Creates and returns TensorFlow placeholders """
        placeholders = {}

        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Building the placeholders
        while models:
            current_model = models.pop(-1)
            placeholders.update(self._placeholders)

        # Returning
        return placeholders

    def get_optimizer(self, learning_rate):
        """ Returns the optimizer to use for this model """
        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Checking if we have an optimizer override
        optimizer = None
        while models:
            current_model = models.pop(-1)
            optimizer = current_model._get_optimizer(learning_rate) or optimizer                                        # pylint: disable=protected-access

        # Default optimizer
        if not optimizer:
            from diplomacy_research.utils.tensorflow import tf
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer

    def make_optimizer(self, learning_rate):
        """ Builds a sync or async Adam optimizer
            :param learning_rate: The learning rate variable
            :return: A sync or async Adam optimizer
        """
        from diplomacy_research.utils.tensorflow import tf
        from diplomacy_research.models.layers.avg_grad_optimizer import AvgGradOptimizer
        assert self.hparams['training_mode'] in ('supervised', 'reinforcement'), 'Invalid training mode.'

        # Getting parameters
        is_distributed = bool(self.cluster_config)
        avg_gradients = self.hparams.get('avg_gradients', False)
        sync_gradients = self.hparams.get('sync_gradients', False)
        training_mode = self.hparams['training_mode']
        max_gradient_norm = self.hparams.get('max_gradient_norm', None)

        # Creating optimizer
        optimizer = self.get_optimizer(learning_rate)

        # Supervised learning - Detecting if we need to sync gradients
        if training_mode == 'supervised' and sync_gradients and is_distributed:
            LOGGER.info('Using SyncReplicasOptimizer for the optimization')
            optimizer = tf.train.SyncReplicasOptimizer(opt=optimizer,
                                                       replicas_to_aggregate=self.cluster_config.num_shards,
                                                       total_num_replicas=self.cluster_config.num_shards)

        # RL - Averaging Gradients
        elif training_mode == 'reinforcement' and avg_gradients:
            LOGGER.info('Using AvgGradOptimizer for the optimization')
            num_workers = 1
            if is_distributed and sync_gradients:
                num_workers = self.cluster_config.count('learner')
            optimizer = AvgGradOptimizer(optimizer,
                                         num_workers=num_workers,
                                         is_chief=not is_distributed or self.cluster_config.is_chief,
                                         max_gradient_norm=max_gradient_norm)

        # RL - Syncing Gradients
        elif training_mode == 'reinforcement' \
                and not avg_gradients \
                and sync_gradients \
                and is_distributed \
                and self.cluster_config.num_shards > 1:
            LOGGER.info('Using SyncReplicasOptimizer for the optimization')
            optimizer = tf.train.SyncReplicasOptimizer(opt=optimizer,
                                                       replicas_to_aggregate=self.cluster_config.num_shards,
                                                       total_num_replicas=self.cluster_config.num_shards)

        # Returning
        return optimizer

    def create_optimizer_op(self, cost_and_scope, ignored_scope=None, max_gradient_norm=None):
        """ Creates an optimizer op to reduce the cost
            :param cost_and_scope: List of tuples (cost, scope, ignored_scope)
                - cost is a tensor representing the cost to minimize
                - scope is either a string, or a list of strings. Contains the scope(s) where the get the vars to update
                - ignored_scope is either None, a string, or a list of strings. Contains scope(s) to ignore.
            :param ignored_scope: A scope or list of scope for which we know we won't compute gradients
            :param max_gradient_norm: Optional. If set, gradients will be clipped to this value.
            :return: The optimizer op

            Note: The ignored scope inside 'cost_and_scope' is local to that cost, while the arg ignored_scope is global
                  for all costs.
        """
        # pylint: disable=too-many-branches
        from diplomacy_research.utils.tensorflow import tf, scope_vars, ensure_finite
        from diplomacy_research.utils import gradient_checkpoint
        assert self.optimizer, 'Optimizer must be defined in self.optimizer before calling this method.'
        if self.cluster_config \
                and self.hparams['training_mode'] == 'supervised' \
                and 'sync_gradients' in self.hparams \
                and self.hparams['sync_gradients']:
            assert isinstance(self.optimizer, tf.train.SyncReplicasOptimizer), 'optimizer must be SyncReplicasOptimizer'
        assert self.hparams['grad_aggregation'].upper() in ['ADD_N', 'ACCUMULATE_N', 'TREE'], 'Invalid aggregation'

        # Warning if more than 1 optimizer is created
        self.nb_optimizers += 1
        if self.nb_optimizers > 1:
            LOGGER.warning('You have created %d optimizers for this model. This is not recommended (High memory usage)',
                           self.nb_optimizers)

        # Determining aggregation_method based on accumulate_n flag
        aggregation_method = None
        if self.hparams['grad_aggregation'].upper() == 'ACCUMULATE_N':
            aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        elif self.hparams['grad_aggregation'].upper() == 'TREE':
            aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE

        # Finding all trainable variables
        all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        ignored_scope_trainable_vars = []
        if isinstance(ignored_scope, list):
            for scope_name in ignored_scope:
                ignored_scope_trainable_vars += scope_vars(scope_name, trainable_only=True)
        elif ignored_scope is not None:
            ignored_scope_trainable_vars = scope_vars(ignored_scope, trainable_only=True)
        ignored_scope_trainable_vars = set(ignored_scope_trainable_vars)

        # Building a list of all trainable vars, and removing them when used by an op
        unused_trainable_vars = set(all_trainable_vars) - set(ignored_scope_trainable_vars)

        # Summing gradients if we are optimizing multiple costs
        global_gradients = {}
        for cost, scope, local_ignored_scope in cost_and_scope:

            local_ignored_vars = []
            if isinstance(local_ignored_scope, list):
                for scope_name in local_ignored_scope:
                    local_ignored_vars += scope_vars(scope_name, trainable_only=True)
            elif local_ignored_scope is not None:
                local_ignored_vars = scope_vars(local_ignored_scope, trainable_only=True)
            local_ignored_vars = set(local_ignored_vars)

            # Computing gradients with respect to all scope vars (except global ignored vars, but incl. local ignored)
            scope_trainable_vars = []
            scope = [scope] if not isinstance(scope, list) else scope
            for scope_name in scope:
                for variable in scope_vars(scope_name, trainable_only=True):
                    if variable not in scope_trainable_vars and variable not in ignored_scope_trainable_vars:
                        scope_trainable_vars += [variable]

            # Computing gradients
            if self.hparams['gradient_checkpoint']:
                LOGGER.info('****** Optimizing graph with gradient checkpointing...')
                gradients = gradient_checkpoint.gradients(cost, scope_trainable_vars,
                                                          checkpoints=self.hparams['gradient_checkpoint'],
                                                          aggregation_method=aggregation_method)
                LOGGER.info('Done optimizing graph with gradient checkpointing...')
            else:
                LOGGER.info('****** Computing gradients with respect to %s...', str(cost))
                gradients = tf.gradients(cost, scope_trainable_vars, aggregation_method=aggregation_method)

            # Storing gradients in global_gradients
            for trainable_var, gradient in zip(scope_trainable_vars, gradients):
                if trainable_var in local_ignored_vars:
                    continue
                if trainable_var in unused_trainable_vars:
                    unused_trainable_vars.remove(trainable_var)
                if gradient is None:
                    LOGGER.warning('Gradient for %s is None. Is the graph disconnected?', str(trainable_var))
                    continue
                if trainable_var.name in global_gradients:
                    global_gradients[str(trainable_var.name)] += [gradient]
                else:
                    global_gradients[str(trainable_var.name)] = [gradient]

        # Warning about missing trainable variables
        for variable in unused_trainable_vars:
            LOGGER.warning('The training variable %s has not been included in the optimizer_op.', str(variable))

        # Warning about ignored training variables
        for variable in ignored_scope_trainable_vars:
            LOGGER.info('Ignoring variable: "%s" (Shape: %s).', str(variable.name), str(variable.shape))

        # Computing and clipping gradients
        gradients = []
        for variable in all_trainable_vars:
            var_gradients = global_gradients.get(str(variable.name), [])
            if not var_gradients:
                gradients += [None]
            elif len(var_gradients) == 1:
                gradients += var_gradients
            else:
                if [1 for grad in var_gradients if isinstance(grad, tf.IndexedSlices)]:
                    LOGGER.info('Adding IndexedSlices for %s', variable)
                gradients += [tf.add_n(var_gradients, name='%s/Add_N' % (variable.name.split(':')[0]))]
        gradients = [ensure_finite(gradient) for gradient in gradients]
        if max_gradient_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # Finding update ops
        update_ops = []
        for _, scope, _ in cost_and_scope:
            if isinstance(scope, list):
                for scope_name in scope:
                    for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name):
                        if update_op not in update_ops:
                            update_ops += [update_op]
            else:
                update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

        # Printing number of variables
        global_vars = tf.global_variables()
        LOGGER.info('Model has %d global vars and %d trainable vars', len(global_vars), len(all_trainable_vars))

        # Computing the number of parameters
        nb_global_params = sum([reduce(mul, variable.shape.as_list(), 1) for variable in global_vars])
        nb_trainable_params = sum([reduce(mul, variable.shape.as_list(), 1) for variable in all_trainable_vars])
        LOGGER.info('Model has %s parameters (%s for trainable vars)',
                    '{:,}'.format(nb_global_params),
                    '{:,}'.format(nb_trainable_params))

        # Creating optimizer op (with dependencies on update for batch norm)
        with tf.control_dependencies(update_ops):
            opt_op = self.optimizer.apply_gradients(zip(gradients, all_trainable_vars), global_step=self.global_step)

        # Returning optimization op
        return opt_op

    def build_policy(self):
        """ Builds a policy model (initial step) """
        if hasattr(self, '_build_policy_initial'):
            return getattr(self, '_build_policy_initial')()
        if self.parent_model is None:
            raise NotImplementedError()

        # Calling our parent, and updating recursively
        ret_val = self.parent_model.build_policy()
        self.__dict__.update(self.parent_model.__dict__)
        return ret_val

    def build_value(self):
        """ Builds a value model (initial step) """
        if hasattr(self, '_build_value_initial'):
            return getattr(self, '_build_value_initial')()
        if self.parent_model is None:
            raise NotImplementedError()

        # Calling our parent, and updating recursively
        ret_val = self.parent_model.build_value()
        self.__dict__.update(self.parent_model.__dict__)
        return ret_val

    def build_draw(self):
        """ Builds a draw model (initial step) """
        if hasattr(self, '_build_draw_initial'):
            return getattr(self, '_build_draw_initial')()
        if self.parent_model is None:
            raise NotImplementedError()

        # Calling our parent, and updating recursively
        ret_val = self.parent_model.build_draw()
        self.__dict__.update(self.parent_model.__dict__)
        return ret_val

    def finalize_build(self):
        """ Builds the policy, value, and draw model (final step) """
        if self.build_finalized:
            LOGGER.warning('Build already finalized. Skipping.')
            return

        if self.parent_model:
            self.parent_model.finalize_build()
            self.outputs.update(self.parent_model.outputs)

        if hasattr(self, '_build_policy_final'):
            getattr(self, '_build_policy_final')()
        if hasattr(self, '_build_value_final'):
            getattr(self, '_build_value_final')()
        if hasattr(self, '_build_draw_final'):
            getattr(self, '_build_draw_final')()

        self.build_finalized = True

    def encode_board(self, board_state, name, reuse=None):
        """ Encodes a board state or prev orders state
            :param board_state: The board state / prev orders state to encode - (batch, NB_NODES, initial_features)
            :param name: The name to use for the encoding
            :param reuse: Whether to reuse or not the weights from another encoding operation
            :return: The encoded board state / prev_orders state
        """
        if hasattr(self, '_encode_board'):
            return getattr(self, '_encode_board')(board_state=board_state, name=name, reuse=reuse)
        if self.parent_model is None:
            raise NotImplementedError()
        return self.parent_model.encode_board(board_state=board_state, name=name, reuse=reuse)

    def get_board_state_conv(self, board_0yr_conv, is_training, prev_ord_conv=None):
        """ Computes the board state conv to use as the attention target (memory)

            :param board_0yr_conv: The board state encoding of the current (present) board state)
            :param is_training: Indicate whether we are doing training or inference
            :param prev_ord_conv: Optional. The encoding of the previous orders state.
            :return: The board state conv to use as the attention target (memory)
        """
        # pylint: disable=too-many-arguments
        if hasattr(self, '_get_board_state_conv'):
            return getattr(self, '_get_board_state_conv')(board_0yr_conv, is_training, prev_ord_conv)
        if self.parent_model is None:
            LOGGER.warning('Unable to find a get_board_state_conv function. Returning the `board_0yr_conv`.')
            return board_0yr_conv
        return self.parent_model.get_board_state_conv(board_0yr_conv, is_training, prev_ord_conv)

    def get_board_value(self, board_state, current_power, name='board_state_value', reuse=None):
        """ Computes the estimated value of a board state
            :param board_state: The board state - (batch, NB_NODES, NB_FEATURES)
            :param current_power: The power for which we want the board value - (batch,)
            :param name: The name to use for the operaton
            :param reuse: Whether to reuse or not the weights from another operation
            :return: The value of the board state for the specified power - (batch,)
        """
        from diplomacy_research.utils.tensorflow import tf
        if hasattr(self, '_get_board_value'):
            return getattr(self, '_get_board_value')(board_state, current_power, name, reuse)
        if self.parent_model is None:
            LOGGER.warning('Unable to find a value function. Returning 0. for `get_board_value`.')
            return tf.zeros_like(current_power, dtype=tf.float32)
        return self.parent_model.get_board_value(board_state, current_power, name, reuse)

    def validate(self):
        """ Validates the built model
            Throws a RuntimeError if the model is not build properly.
        """
        assert self.build_finalized, 'The model has not been finalized. Please call .finalize_build()'
        self._validate()
        if self.parent_model:
            self.parent_model.validate()

    def get_session_args(self, decode=False, eval_loop_ix=None):
        """ Returns a dict of kwargs to feed to session.run
            Expected format: {fetches, feed_dict=None}
        """
        fetches, feed_dict = {}, {}

        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Updates the session args
        while models:
            current_model = models.pop(-1)
            new_session_args = current_model._get_session_args(decode=decode, eval_loop_ix=eval_loop_ix)                # pylint: disable=protected-access
            new_fetches = new_session_args.get('fetches', {})
            new_feed_dict = new_session_args.get('feed_dict', None) or {}
            fetches.update(new_fetches)
            feed_dict.update(new_feed_dict)

        # Returning
        return {'fetches': fetches, 'feed_dict': feed_dict}

    def decode(self, **fetches):
        """ Performs decoding on the output
            :param fetches: A dictionary of fetches from the model.
            :return: A dictionary of decoded results
        """
        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Decoding
        decoded_results = {}
        while models:
            current_model = models.pop(-1)
            new_decoded_results = current_model._decode(**fetches)                                                      # pylint: disable=protected-access
            decoded_results = merge_complex_dicts(decoded_results, new_decoded_results)

        # Returning
        return decoded_results

    def evaluate(self, decoded_results, feed_dict, eval_loop_ix, incl_detailed):
        """ Evaluates the model
            :param decoded_results: The decoded results (output of _decode() function)
            :param feed_dict: The feed dictionary that was given to session.run()
            :param eval_loop_ix: The current evaluation loop index
            :param incl_detailed: is true if training is over, more statistics can be computed
            :return: A tuple consisting of:
                        1) An ordered dictionary with result_name as key and result value as value   (Regular results)
                        2) An ordered dictionary with result_name as key and a list of result values (Detailed results)
        """
        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Evaluating
        regular, detailed = OrderedDict(), OrderedDict()
        while models:
            current_model = models.pop(-1)
            new_regular, new_detailed = current_model._evaluate(decoded_results, feed_dict, eval_loop_ix, incl_detailed)    # pylint: disable=protected-access
            regular = merge_complex_dicts(regular, new_regular)
            detailed = merge_complex_dicts(detailed, new_detailed)

        # Returning
        return regular, detailed

    def post_process_results(self, detailed_results):
        """ Perform post-processing on the detailed results
            :param detailed_results: An dictionary which contains detailed evaluation statistics
            :return: A dictionary with the post-processed statistics.
        """
        # Finding the list of models from the top parent downwards
        current_model = self
        models = [current_model]
        while current_model.parent_model:
            current_model = current_model.parent_model
            models += [current_model]

        # Post-Processing
        while models:
            current_model = models.pop(-1)
            detailed_results = current_model._post_process_results(detailed_results)                                    # pylint: disable=protected-access

        # Returning
        return detailed_results

    def add_meta_information(self, outputs):
        """ Adds features, placeholders, and outputs to the meta-graph
            :param outputs: A dictionary of outputs (e.g. {'training_op': ..., 'training_outputs': ...})
            :return: Nothing, but adds those items to the meta-graph
        """
        from diplomacy_research.utils.tensorflow import tf

        # Storing features
        for feature in self.features:
            cached_features = tf.get_collection('feature_{}'.format(feature))
            if not cached_features:
                tf.add_to_collection('feature_{}'.format(feature), self.features[feature])

        # Storing placeholders
        for placeholder in self.placeholders:
            cached_placeholders = tf.get_collection('placeholder_{}'.format(placeholder))
            if not cached_placeholders:
                tf.add_to_collection('placeholder_{}'.format(placeholder), self.placeholders[placeholder])

        # Storing outputs in model
        for output_name in outputs:
            self.outputs[output_name] = outputs[output_name]
        self.outputs['tag/commit_hash'] = GIT_COMMIT_HASH
        self.outputs['is_trainable'] = True
        self.outputs['iterator_resource'] = self.iterator_resource

        # Labeling outputs on meta-graph
        # Clearing collection if we want to re-add the key
        # Avoiding creating key if already present
        for output_tag in self.outputs:
            if output_tag in outputs and tf.get_collection(output_tag):
                tf.get_default_graph().clear_collection(output_tag)
            if self.outputs[output_tag] is not None \
                    and not output_tag.startswith('_') \
                    and not output_tag.endswith('_ta') \
                    and not tf.get_collection(output_tag):
                tf.add_to_collection(output_tag, self.outputs[output_tag])

        # Storing hparams
        for hparam_name, hparam_value in self.hparams.items():
            if not tf.get_collection('tag/hparam/{}'.format(hparam_name)):
                tf.add_to_collection('tag/hparam/{}'.format(hparam_name), str(hparam_value))

    def add_output(self, output_name, output_value):
        """ Adds an output to all the models """
        current_model = self
        while current_model:
            current_model.outputs[output_name] = output_value
            current_model = current_model.parent_model

    # -------------------------------------------------
    #               Private Methods
    # -------------------------------------------------
    @property
    def _nb_evaluation_loops(self):
        """ Contains the number of different evaluation tags we want to compute
            This also represent the number of loops we should do over the validation set
            Some model wants to calculate different statistics and require multiple pass to do that

            A value of 1 indicates to only run in the main validation loop
            A value > 1 indicates to run additional loops only for this model.
        """
        return 1

    @property
    def _evaluation_tags(self):
        """ List of evaluation tags (1 list of evaluation tag for each evaluation loop)
            e.g. [['Acc_1', 'Acc_5', 'Acc_Tokens'], ['Gr_1', 'Gr_5', 'Gr_Tokens']]
        """
        return []

    @property
    def _early_stopping_tags(self):
        """ List of tags to use to detect early stopping
            The tags are a tuple of 1) 'min' or 'max' and 2) the tag's name
            e.g. [('max', '[Gr]Acc_1'), ('min', '[TF]Perplexity')]
        """
        return []

    @property
    def _placeholders(self):
        """ Return a dictionary of all placeholders needed by the model """
        return {}

    @staticmethod
    def _get_optimizer(learning_rate):
        """ Returns the optimizer to use for this model """
        del learning_rate                       # Unused args

    def _validate(self):
        """ Validates the built model """

    @staticmethod
    def _get_session_args(decode=False, eval_loop_ix=None):
        """ Returns a dict of kwargs to feed to session.run
            Expected format: {fetches, feed_dict=None}
        """
        del decode, eval_loop_ix                # Unused args
        return {}

    @staticmethod
    def _decode(**fetches):
        """ Performs decoding on the output
            :param fetches: A dictionary of fetches from the model.
            :return: A dictionary of decoded results
        """
        del fetches                             # Unused args
        return {}

    @staticmethod
    def _evaluate(decoded_results, feed_dict, eval_loop_ix, incl_detailed):
        """ Evaluates the model
            :param decoded_results: The decoded results (output of _decode() function)
            :param feed_dict: The feed dictionary that was given to session.run()
            :param eval_loop_ix: The current evaluation loop index
            :param incl_detailed: is true if training is over, more statistics can be computed
            :return: A tuple consisting of:
                        1) An ordered dictionary with result_name as key and (weight, value) as value  (Regular results)
                        2) An ordered dictionary with result_name as key and a list of result values  (Detailed results)
        """
        del decoded_results, feed_dict, eval_loop_ix, incl_detailed         # Unused args
        return OrderedDict(), OrderedDict()

    @staticmethod
    def _post_process_results(detailed_results):
        """ Perform post-processing on the detailed results
            :param detailed_results: An dictionary which contains detailed evaluation statistics
            :return: A dictionary with the post-processed statistics.
        """
        return detailed_results
