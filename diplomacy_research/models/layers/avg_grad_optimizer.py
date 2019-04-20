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
""" Average Gradient Optimizer
    - Accumulates a certain number of gradients, average them and then applies them to the variables
"""
import sys
assert 'tensorflow' in sys.modules, 'You need to import TF before importing this module.'

from diplomacy_research.utils.tensorflow import ops
from diplomacy_research.utils.tensorflow import clip_ops
from diplomacy_research.utils.tensorflow import control_flow_ops, gen_control_flow_ops
from diplomacy_research.utils.tensorflow import data_flow_ops
from diplomacy_research.utils.tensorflow import array_ops, gen_array_ops
from diplomacy_research.utils.tensorflow import ensure_finite
from diplomacy_research.utils.tensorflow import dtypes
from diplomacy_research.utils.tensorflow import math_ops, gen_math_ops
from diplomacy_research.utils.tensorflow import optimizer
from diplomacy_research.utils.tensorflow import state_ops
from diplomacy_research.utils.tensorflow import variable_scope


class AvgGradOptimizer(optimizer.Optimizer):
    """ Class to aggregate gradients and then apply them (by passing them to the optimizer.)

        The following accumulators/queue are created:

        * N `gradient accumulators`, one per variable to train. Gradients are pushed to them and will be averaged
            before being applied to variables.

        The optimizer adds nodes to the graph to collect gradients.

        For the Parameter Server job:

        1. An accumulator is created for each variable, and each replica pushes the gradients into the accumulators
            instead of directly applying them to the variables.
        2. Each accumulator averages all gradients when the chief requests the optimizer to apply the gradients.
        3. The global step is increased every time a gradient is pushed on the accumulator

        For the replicas:

        1. Start a step: fetch variables and compute gradients.
        2. Once the gradients have been computed, push them into gradient accumulators.
    """

    def __init__(self, opt, num_workers, is_chief, use_locking=False, max_gradient_norm=None,
                 name='avg_grad_optimizer'):
        """ Constructor
            :param opt: The actual optimizer that will be used to compute and apply the gradients
            :param num_workers: The number of workers sending gradients to the average accumulator.
            :param is_chief: Boolean that indicates if the current worker is the chief.
            :param use_locking: Boolean. If True use locks for update operation.
            :param max_gradient_norm: If set, the average gradients are also clipped by global norm.
            :param average_locally: Boolean. If set, only the workers variables are averaged.
            :param name: Optional name of the returned operation.
            :type opt: optimizer.Optimizer
        """
        super(AvgGradOptimizer, self).__init__(use_locking=use_locking, name=name)
        self._optimizer = opt
        self._num_workers = num_workers
        self._is_chief = is_chief or num_workers == 1
        self._use_locking = use_locking
        self._max_gradient_norm = max_gradient_norm
        self._accumulators = {}                         # var_name: (var, accumulator, device)
        self._finalized = False

        self._local_step = None
        self._global_step = None
        self._sync_token_queue = None

        self.local_step_init_op = None                  # Does local_step = global_step
        self.chief_init_op = None
        self._apply_grad_op = None                      # Chief - Applies gradient

    def compute_gradients(self, *args, **kwargs):
        """ Compute gradients of 'loss' for the variables in 'var_list'

            This simply wraps the compute_gradients() from the real optimizer. The gradients will be aggregated in
            the apply_gradients() so that user can modify the gradients like clipping with per replica global norm
            if needed. The global norm with aggregated gradients can be bad as one replica's huge gradients can hurt
            the gradients from other replicas.

            :return: A list of (gradient, variable) pairs
        """
        # pylint: disable=arguments-differ
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """ Accumulates gradients for the variables and stores them in the accumulator

            Note: This does not update the variables, it just accumulate the gradients

            :param grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients()
            :param global_step: Optional variable to increment by one after the variables have been updated.
            :param name: Optional name for the returned op. Default to name passed to the optimizer constructor.
            :return: The training operation to be run by each replica

            Raises - ValueError if the grads_and_vars is empty
                   - ValueError if global step is not provided
                   - ValueError if update() has already been called
        """
        # Making sure grads_and_var and global_step are provided
        if not grads_and_vars:
            raise ValueError('Must supply at least one variable.')
        if not global_step:
            raise ValueError('You must provide a global_step variable')
        if self._finalized:
            raise ValueError('The optimizer has been finalized. You cannot call this method after update().')

        train_ops = []
        accumulated_grad = []
        var_list = []
        chief_init_ops = []
        self._global_step = global_step

        # Colocating local step to prevent it from being placed on the parameter server
        local_anchor = gen_control_flow_ops.no_op()
        with ops.device(local_anchor.device):
            self._local_step = variable_scope.variable(initial_value=0,
                                                       trainable=False,
                                                       collections=[ops.GraphKeys.LOCAL_VARIABLES],
                                                       dtype=global_step.dtype.base_dtype,
                                                       name='sync_rep_local_step')

        # Setting initial step
        self.local_step_init_op = state_ops.assign(self._local_step, global_step)
        chief_init_ops += [self.local_step_init_op]

        with ops.name_scope(None, self._name):

            # Creating accumulators
            for grad, var in grads_and_vars:
                var_list += [var]
                with ops.device(var.device):

                    # No gradient - Pass-Through
                    if grad is None:
                        accumulated_grad += [None]
                        continue

                    # Sparse Variable - Accumulating over the dense shape
                    elif isinstance(grad, ops.IndexedSlices):
                        grad_accum = self._create_sparse_accumulator(var, grad)
                        train_ops += [grad_accum.apply_indexed_slices_grad(grad, local_step=self._local_step)]
                        accumulated_grad += [self._take_sparse_grad(grad_accum, grad)]
                        chief_init_ops += [grad_accum.set_global_step(global_step, name='SetGlobalStep')]

                    # Dense Variable
                    elif isinstance(grad, ops.Tensor):
                        grad_accum = self._create_dense_accumulator(var, grad)
                        train_ops += [grad_accum.apply_grad(grad, local_step=self._local_step)]
                        accumulated_grad += [self._take_dense_grad(grad_accum, grad)]
                        chief_init_ops += [grad_accum.set_global_step(global_step, name='SetGlobalStep')]

                    # Unknown
                    else:
                        raise RuntimeError('Unsupported gradient type.')

            # Building update_op
            with ops.device(self._global_step.device), ops.name_scope(''):
                accumulated_grad = [ensure_finite(gradient) for gradient in accumulated_grad]
                if self._max_gradient_norm:
                    accumulated_grad, _ = clip_ops.clip_by_global_norm(accumulated_grad, self._max_gradient_norm)
                self._apply_grad_op = self._optimizer.apply_gradients(zip(accumulated_grad, var_list), global_step)

            # Building chief init ops
            self.chief_init_op = control_flow_ops.group(*chief_init_ops)

            # Building train_op
            return control_flow_ops.group(*train_ops)

    def update(self, version_step=None):
        """ Performs the gradient averaging, updates the variables, and the global step
            :param version_step: A variable that represents the model's version
            :return: The update operation to be run by the chief

            Raises - ValueError if apply_gradients has not been calculated.
                   - ValueError if reset() has already been called.
        """
        if not self._accumulators or not self._global_step:
            raise ValueError('Should be called after apply_gradients().')
        if self._finalized:
            raise ValueError('The optimizer has been finalized. You cannot call this method after update().')

        self._finalized = True
        if self._num_workers == 1:
            return self._update_standalone(version_step=version_step)
        if self._is_chief:
            return self._update_distributed_as_chief(version_step=version_step)
        return self._update_distributed_as_worker()

    def init(self):
        """ Returns the operation to run to initialize the avg grad optimizer """
        return self.chief_init_op if self._is_chief else self.local_step_init_op

    def _update_standalone(self, version_step=None):
        """ Performs the gradient averaging, updates the variables, and the global step
            :param version_step: A variable that represents the model's version
            :return: The update operation to run

            Note: This method is called when there are no workers (no synchronization)
        """
        with ops.device(self._global_step.device), ops.name_scope(''):
            with ops.control_dependencies([self._apply_grad_op]):
                update_ops = [state_ops.assign(self._local_step, self._global_step)]

                # Increasing version step
                if version_step is not None:
                    update_ops += [state_ops.assign_add(version_step, 1)]

                # Returning
                return control_flow_ops.group(*update_ops)

    def _update_distributed_as_chief(self, version_step=None):
        """ Performs the gradient averaging, updates the variables, and the global step
            :param version_step: A variable that represents the model's version
            :return: The update operation to run

            Note: This method is called by the chief when synchronization is required.
        """
        # Creating sync_token queue
        with ops.device(self._global_step.device), ops.name_scope(''):
            self._sync_token_queue = data_flow_ops.FIFOQueue(capacity=-1,
                                                             dtypes=self._global_step.dtype.base_dtype,
                                                             shapes=(),
                                                             name='sync_token_q',
                                                             shared_name='sync_token_q')

            # Applying grads, then adding tokens to queue
            with ops.control_dependencies([self._apply_grad_op]):
                tokens = gen_array_ops.fill([self._num_workers], self._global_step)
                sync_op = self._sync_token_queue.enqueue_many((tokens,))

                # Waiting for token in queue (sync point)
                with ops.control_dependencies([sync_op]):
                    token = self._sync_token_queue.dequeue()
                    update_ops = [state_ops.assign(self._local_step, token)]

                    # Increasing version step
                    if version_step is not None:
                        update_ops += [state_ops.assign_add(version_step, 1)]

                    # Returning
                    return control_flow_ops.group(*update_ops)

    def _update_distributed_as_worker(self):
        """ Performs the gradient averaging, updates the variables, and the global step
            :param version_step: A variable that represents the model's version
            :return: The update operation to run

            Note: This method is called by a worker when synchronization is required.
        """
        # Creating sync_token queue
        with ops.device(self._global_step.device), ops.name_scope(''):
            self._sync_token_queue = data_flow_ops.FIFOQueue(capacity=-1,
                                                             dtypes=self._global_step.dtype.base_dtype,
                                                             shapes=(),
                                                             name='sync_token_q',
                                                             shared_name='sync_token_q')

            # Waiting for token in queue (sync point)
            token = self._sync_token_queue.dequeue()
            return state_ops.assign(self._local_step, token)

    def _create_dense_accumulator(self, var, grad):
        """ Creates a dense accumulator for the specified variable """
        assert var.name not in self._accumulators, 'Variable %s has already an accumulator' % var.name
        shared_name = None if self._num_workers == 1 else var.name + '/grad_accum'
        grad_accum = data_flow_ops.ConditionalAccumulator(grad.dtype,
                                                          shape=var.get_shape(),
                                                          shared_name=shared_name)
        self._accumulators[var.name] = (var, grad_accum, var.device)
        return grad_accum

    def _create_sparse_accumulator(self, var, grad):
        """ Creates a sparse accumulator for the specified variable """
        assert var.name not in self._accumulators, 'Variable %s has already an accumulator' % var.name
        shared_name = None if self._num_workers == 1 else var.name + '/grad_accum'
        grad_accum = data_flow_ops.SparseConditionalAccumulator(grad.dtype,
                                                                shape=(),
                                                                shared_name=shared_name)
        self._accumulators[var.name] = (var, grad_accum, var.device)
        return grad_accum

    @staticmethod
    def _take_dense_grad(grad_accum, grad):
        """ Computes the gradient for a ConditionalAccumulator
            :param grad_accum: The gradient accumulator where gradients are stored
            :param grad: An instance of the gradient stored in the accumulator
            :return: The avg gradient to apply (or a zero-like object if no gradients are stored)
            :type grad_accum: data_flow_ops.ConditionalAccumulator
        """
        def _take_grad():
            """ Computes the gradient from the accumulator """
            avg_grad = grad_accum.take_grad(num_required=1)
            with ops.control_dependencies([avg_grad]):
                return array_ops.identity(avg_grad)

        def _zero_grad():
            """ Returns a zeroed-out gradient """
            zero_like_grad = array_ops.zeros_like(grad)
            with ops.control_dependencies([zero_like_grad]):
                return array_ops.identity(zero_like_grad)

        return control_flow_ops.cond(gen_math_ops.equal(grad_accum.num_accumulated(), 0),
                                     true_fn=_zero_grad,
                                     false_fn=_take_grad)

    @staticmethod
    def _take_sparse_grad(grad_accum, grad):
        """ Computes the gradient for a SparseConditionalAccumulator
            :param grad_accum: The gradient accumulator where gradients are stored
            :param grad: An instance of the gradient stored in the accumulator
            :return: The avg gradient to apply (or a zero-like object if no gradients are stored)
            :type grad_accum: data_flow_ops.SparseConditionalAccumulator
        """
        def _take_grad():
            """ Computes the gradient from the accumulator """
            avg_grad = grad_accum.take_indexed_slices_grad(num_required=1)
            with ops.control_dependencies([avg_grad]):
                return ops.IndexedSlices(values=array_ops.identity(avg_grad.values),
                                         indices=avg_grad.indices,
                                         dense_shape=avg_grad.dense_shape)

        def _zero_grad():
            """ Returns a zeroed-out gradient """
            zero_values = array_ops.zeros_like(grad.values)
            with ops.control_dependencies([zero_values]):
                return ops.IndexedSlices(values=array_ops.identity(zero_values),
                                         indices=math_ops.cast(grad.indices, dtypes.int64),
                                         dense_shape=math_ops.cast(grad.dense_shape, dtypes.int64))

        return control_flow_ops.cond(gen_math_ops.equal(grad_accum.num_accumulated(), 0),
                                     true_fn=_zero_grad,
                                     false_fn=_take_grad)

    def get_slot(self, var, name):
        """ Returns a slot named 'name' created by 'var' by the Optimizer """
        return self._optimizer.get_slot(var, name)

    def get_slot_names(self):
        """ Returns a list of the names of slots created by the Optimizer """
        return self._optimizer.get_slot_names()

    def variables(self):
        """ Fetches a list of optimizer variables in the default graph (excluding the local step) """
        return self._optimizer.variables()

    def _apply_dense(self, grad, var):
        """ Add ops to apply dense gradients to `var`. """
        raise NotImplementedError()

    def _apply_sparse(self, grad, var):
        """ Add ops to apply sparse gradients to `var`. """
        raise NotImplementedError()

    def _resource_apply_dense(self, grad, handle):
        """ Add ops to apply dense gradients to the variable `handle`. """
        raise NotImplementedError()

    def _resource_apply_sparse(self, grad, handle, indices):
        """ Add ops to apply sparse gradients to the variable `handle`. """
        raise NotImplementedError()
