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
""" Base Draw model
    - Contains the base draw model, which is used by all draw models.
"""
from collections import OrderedDict
import logging
import re
from diplomacy_research.models.base_model import BaseModel

# Constants
LOGGER = logging.getLogger(__name__)
VALID_TAG = re.compile('^tag/draw/v[0-9]{3}_[a-z0-9_]+$')

def load_args():
    """ Load possible arguments
        :return: A list of tuple (arg_type, arg_name, arg_value, arg_desc)
    """
    return [
        ('int', 'draw_model_id', -1, 'The model ID of the value function.'),
        ('float', 'draw_coeff', 1.0, 'The coefficient to apply to the draw loss')
    ]

class BaseDrawModel(BaseModel):
    """ Base Draw Model"""

    def __init__(self, parent_model, dataset, hparams):
        """ Initialization
            :param parent_model: A `base_model` to which we are adding features
            :param dataset: The dataset that is used to iterate over the data.
            :param hparams: A dictionary of hyper parameters with their values
            :type parent_model: diplomacy_research.models.base_model.BaseModel
            :type dataset: diplomacy_research.models.datasets.supervised_dataset.SupervisedDataset
            :type dataset: diplomacy_research.models.datasets.queue_dataset.QueueDataset
        """
        BaseModel.__init__(self, parent_model, dataset, hparams)
        self.build_draw()

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
        return [['[Draw]L2_Loss']]

    @property
    def _early_stopping_tags(self):
        """ List of tags to use to detect early stopping
            The tags are a tuple of 1) 'min' or 'max' and 2) the tag's name
            e.g. [('max', '[Gr]Acc_1'), ('min', '[TF]Perplexity')]
        """
        return [('min', '[Draw]L2_Loss')]

    @property
    def _placeholders(self):
        """ Return a dictionary of all placeholders needed by the model """
        from diplomacy_research.utils.tensorflow import tf, get_placeholder_with_default
        return {
            'stop_gradient_all': get_placeholder_with_default('stop_gradient_all', False, shape=(), dtype=tf.bool)
        }

    def _build_draw_initial(self):
        """ Builds the draw model (initial step) """
        raise NotImplementedError()

    def _build_draw_final(self):
        """ Builds the draw model (final step) """

    def _get_session_args(self, decode=False, eval_loop_ix=None):
        """ Returns a dict of kwargs to feed to session.run
            Expected format: {fetches, feed_dict=None}
        """
        # Detecting if we are doing validation
        in_validation, our_validation = False, False
        if eval_loop_ix is not None:
            in_validation = True
            our_validation = eval_loop_ix in self.my_eval_loop_ixs

        # --------- Fetches ---------------
        train_fetches = {'optimizer_op': self.outputs['optimizer_op'],
                         'draw_loss': self.outputs['draw_loss']}

        eval_fetches = {'draw_loss': self.outputs['draw_loss']}

        # --------- Feed dict --------------
        # Building feed dict
        feed_dict = {self.placeholders['stop_gradient_all']: False}

        # --------- Validation Loop --------------
        # Validation Loop - Running one of our validation loops
        if our_validation:
            return {'fetches': eval_fetches, 'feed_dict': feed_dict}

        # Validation Loop - Running someone else validation loop
        if in_validation:
            return {'feed_dict': feed_dict}

        # --------- Training Loop --------------
        # Training Loop - We want to decode the specific batch to display stats
        if decode:
            return {'fetches': eval_fetches, 'feed_dict': feed_dict}

        # Training Loop - Training the model
        return {'fetches': train_fetches, 'feed_dict': feed_dict}

    def _validate(self):
        """ Validates the built model """
        # Making sure all the required outputs are present
        assert 'draw_target' in self.features
        assert 'draw_prob' in self.outputs
        assert 'draw_loss' in self.outputs
        assert len(self.outputs['draw_prob'].shape) == 1

        # Making sure we have a name tag
        for tag in self.outputs:
            if VALID_TAG.match(tag):
                break
        else:
            raise RuntimeError('Unable to find a name tag. Format: "tag/draw/v000_xxxxxx".')

    @staticmethod
    def _decode(**fetches):
        """ Performs decoding on the output (draw model)
            :param fetches: A dictionary of fetches from the model.
            :return: A dictionary of decoded results, including various keys for evaluation
        """
        if 'draw_loss' not in fetches:
            return {}
        return {'draw_loss': fetches['draw_loss']}

    def _evaluate(self, decoded_results, feed_dict, eval_loop_ix, incl_detailed):
        """ Calculates the accuracy of the model
            :param decoded_results: The decoded results (output of _decode() function)
            :param feed_dict: The feed dictionary that was given to session.run()
            :param eval_loop_ix: The current evaluation loop index
            :param incl_detailed: is true if training is over, more statistics can be computed
            :return: A tuple consisting of:
                        1) An ordered dictionary with result_name as key and (weight, value) as value  (Regular results)
                        2) An ordered dictionary with result_name as key and a list of result values  (Detailed results)
        """
        # Detecting if it's our evaluation or not
        if eval_loop_ix == -1:
            pass
        else:
            our_validation = eval_loop_ix in self.my_eval_loop_ixs
            if not our_validation:
                return OrderedDict(), OrderedDict()

        # Returning evaluation results
        return OrderedDict({'[Draw]L2_Loss': (1, decoded_results['draw_loss'])}), OrderedDict()
