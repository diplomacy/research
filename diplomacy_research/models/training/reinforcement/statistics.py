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
""" Reinforcement Learning - Statistics
    - Class responsible for for computing and displaying RL statistics
"""
import datetime
import logging
import os
import pickle
import resource
import time
import numpy as np
from diplomacy_research.models.gym.environment import DoneReason
from diplomacy_research.models.training.reinforcement.common import get_version_id, get_opponent_version_id, \
    get_nb_rl_agents
from diplomacy_research.models.state_space import NB_SUPPLY_CENTERS, ALL_POWERS

# Constants
LOGGER = logging.getLogger(__name__)
STATS_FILE_NAME = 'rl_stats.pkl'


def reset_stats(trainer):
    """ Resets the dictionary of statistics to its initial state
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    trainer.stats = {key: {power_name: {'episode_rewards': [],
                                        'eval_rewards': {reward_fn.name: [] for reward_fn in trainer.eval_reward_fns},
                                        'ending_scs': [],
                                        'has_won': [],
                                        'has_most_sc': [],
                                        'rank': [],
                                        'nb_years': [],
                                        'phases_to_reach_sc': {nb_sc: [] for nb_sc in range(4, 20, 2)},
                                        'done_reason': [],
                                        'nb_games': 0}
                           for power_name in ALL_POWERS + ['ALL']}
                     for key in ['train', 'eval']}
    for key in ['train', 'eval']:
        trainer.stats[key].update({'memory_usage': 0,
                                   'version/player': 0,
                                   'version/opponent': 0,
                                   'nb_rl_agents': 0,
                                   'nb_steps': 0,
                                   'nb_games': 0,
                                   'nb_games_last_version': 0})

    # Loading stats from disk
    if os.path.exists(os.path.join(trainer.flags.save_dir, STATS_FILE_NAME)):
        with open(os.path.join(trainer.flags.save_dir, STATS_FILE_NAME), 'rb') as stats_file:
            trainer.stats = pickle.load(stats_file)

    # Overriding starting_version
    trainer.stats['starting_version'] = get_version_id(trainer)

def save_stats(trainer):
    """ Saves statistics to disk """
    with open(os.path.join(trainer.flags.save_dir, STATS_FILE_NAME), 'wb') as stats_file:
        pickle.dump(trainer.stats, stats_file, pickle.HIGHEST_PROTOCOL)

def update_stats(trainer, stats_mode, stats_games_proto, epoch_results=None):
    """ Updates the stats from the latest played games
        :param trainer: A reinforcement learning trainer instance.
        :param stats_mode: One of 'train', or 'eval'.
        :param stats_games_proto: A list of `SavedGameProto` instances to use to update the stats
        :param epoch_results: A dictionary with evaluation tags as key and a list of value for each mini-batch
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert stats_mode in ['train', 'eval'], 'Arg "stats_mode" must be either "train" or "eval".'

    epoch_results = epoch_results or {}
    nb_rl_agents = get_nb_rl_agents(trainer.flags.mode if stats_mode == 'train' else trainer.flags.eval_mode)
    stats = trainer.stats[stats_mode]
    version_id = get_version_id(trainer)

    stats['nb_rl_agents'] = nb_rl_agents
    stats['nb_games'] += len(stats_games_proto)
    stats['nb_games_last_version'] = len(stats_games_proto)

    # Resetting nb of games
    for power_name in ALL_POWERS + ['ALL']:
        stats[power_name]['nb_games'] = 0

    # Computing the total reward, nb of ending sc, if we have won, and if we have the most SC
    for saved_game_proto in stats_games_proto:
        if not saved_game_proto.phases:
            LOGGER.warning('Game %s does not have phases. Skipping stats.', saved_game_proto.id)
            continue
        centers = saved_game_proto.phases[-1].state.centers
        nb_centers = [len(centers[power_name].value) for power_name in saved_game_proto.assigned_powers]
        rank = sorted(nb_centers, reverse=True).index(nb_centers[0]) + 1
        stats['nb_steps'] += len(saved_game_proto.phases) * nb_rl_agents

        # Updating the power and the combined stats
        for rl_power_name in saved_game_proto.assigned_powers[:nb_rl_agents]:
            for power_name in [rl_power_name, 'ALL']:
                stats[power_name]['episode_rewards'] += [sum(saved_game_proto.rewards[rl_power_name].value)]
                for reward_fn in trainer.eval_reward_fns:
                    sum_eval_rewards = sum(reward_fn.get_episode_rewards(saved_game_proto, rl_power_name))
                    stats[power_name]['eval_rewards'][reward_fn.name] += [sum_eval_rewards]
                stats[power_name]['ending_scs'] += [nb_centers[0]]
                stats[power_name]['has_won'] += [int(nb_centers[0] >= (NB_SUPPLY_CENTERS // 2 + 1))]
                stats[power_name]['has_most_sc'] += [int(nb_centers[0] == max(nb_centers))]
                stats[power_name]['rank'] += [rank]
                stats[power_name]['done_reason'] += [saved_game_proto.done_reason]
                stats[power_name]['nb_games'] += 1

    # Computing the first year at which we reach the specified number of sc
    # We use a default of 35 years if we never reach the number of SC
    for saved_game_proto in stats_games_proto:
        year_reached = {nb_sc: 35 for nb_sc in range(0, NB_SUPPLY_CENTERS + 1)}

        for rl_power_name in saved_game_proto.assigned_powers[:nb_rl_agents]:

            # Looping through every phase and setting the number of sc there
            year = 0
            for phase in saved_game_proto.phases:
                nb_sc = len(phase.state.centers[rl_power_name].value)
                if phase.name[3:5].isdigit():
                    year = int(phase.name[3:5])
                    year_reached[nb_sc] = min(year_reached[nb_sc], year)

                # Using the year of the last phase when the game is completed
                elif phase.name == 'COMPLETED':
                    year_reached[nb_sc] = min(year_reached[nb_sc], year)

            # Making sure that if the years reached are in ascending order
            # In case we conquered more than 1 sc in a given year
            for nb_sc in range(NB_SUPPLY_CENTERS - 1, -1, -1):
                year_reached[nb_sc] = min(year_reached[nb_sc], year_reached[nb_sc + 1])

            # Adding to list
            for nb_sc in range(4, 20, 2):
                stats[rl_power_name]['phases_to_reach_sc'][nb_sc] += [year_reached[nb_sc]]
                stats['ALL']['phases_to_reach_sc'][nb_sc] += [year_reached[nb_sc]]

            # Number of years
            # e.g. S1951M - S1901M = 50 yrs
            # Some years might be string (e.g. COMPLETED, FORMING), so we need to skip those
            for phase in reversed(saved_game_proto.phases):
                if phase.name[3:5].isdigit():
                    stats[rl_power_name]['nb_years'] += [int(phase.name[3:5]) - 1]
                    stats['ALL']['nb_years'] += [int(phase.name[3:5]) - 1]
                    break
            else:
                stats[rl_power_name]['nb_years'] += [0]
                stats['ALL']['nb_years'] += [0]

    # Logging memory usage
    stats['memory_usage'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    # Logging versions
    stats['version/player'] = version_id
    stats['version/opponent'] = get_opponent_version_id(trainer) if stats_mode == 'train' else 0

    # Trimming stats dict
    for key in stats:
        if isinstance(stats[key], list):
            stats[key] = stats[key][-1000:]
        elif isinstance(stats[key], dict):
            for sub_key in stats[key]:
                if isinstance(stats[key][sub_key], list):
                    stats[key][sub_key] = stats[key][sub_key][-1000:]

    # Updating algo evaluation tags
    for eval_tag in trainer.algorithm.get_evaluation_tags():
        stats[eval_tag] = epoch_results.get(eval_tag, [])

    # Updating
    trainer.stats[stats_mode] = stats

def compile_stats(trainer, stats_mode):
    """ Compile the stats for the last version update to save for Tensorboard
        :param trainer: A reinforcement learning trainer instance.
        :param stats_mode: One of 'train', or 'eval'.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    assert stats_mode in ['train', 'eval'], 'Arg "training_mode" must be either "train" or "eval".'
    nb_rl_agents = get_nb_rl_agents(trainer.flags.mode if stats_mode == 'train' else trainer.flags.eval_mode)
    stats = trainer.stats[stats_mode]
    version_id = get_version_id(trainer)

    nb_new_games = stats.get('nb_games_last_version', 0)
    avg_nb_new_games_per_agent = int(1 + nb_new_games / nb_rl_agents)
    if not nb_new_games:
        LOGGER.warning('Unable to find games to display stats. Skipping.')
        return
    current_time = int(time.time())

    def compute_avg(result_key, pow_name, sub_key=None, factor=1.):
        """ Computes the avg result for a power """
        nb_new_power_games = stats[pow_name]['nb_games']

        # With a sub-key
        if sub_key:
            if not stats[pow_name][result_key][sub_key]:
                return -1
            if not nb_new_power_games:
                return factor * np.mean(stats[pow_name][result_key][sub_key][-avg_nb_new_games_per_agent:])
            return factor * np.mean(stats[pow_name][result_key][sub_key][-nb_new_power_games:])

        # Without a sub-key
        if not stats[pow_name][result_key]:
            return -1
        if not nb_new_power_games:
            return factor * np.mean(stats[pow_name][result_key][-avg_nb_new_games_per_agent:])
        return factor * np.mean(stats[pow_name][result_key][-nb_new_power_games:])

    # Building a results to convert to a feed dict
    feed_dict = {}
    for power_name in ALL_POWERS + ['ALL']:
        feed_dict['rew_avg/%s' % power_name] = compute_avg('episode_rewards', power_name)
        feed_dict['sc_avg/%s' % power_name] = compute_avg('ending_scs', power_name)
        feed_dict['win_by_sc/%s' % power_name] = compute_avg('has_won', power_name, factor=100.)
        feed_dict['most_sc/%s' % power_name] = compute_avg('has_most_sc', power_name, factor=100.)
        feed_dict['nb_years/%s' % power_name] = compute_avg('nb_years', power_name)
        feed_dict['rank/%s' % power_name] = compute_avg('rank', power_name)

        # Year to reach x
        for nb_sc in range(4, 20, 2):
            feed_dict['year_at_%d_sc/%s' % (nb_sc, power_name)] = compute_avg('phases_to_reach_sc',
                                                                              power_name,
                                                                              sub_key=nb_sc)

        # Done reason
        nb_new_power_games = stats[power_name]['nb_games']
        if not nb_new_power_games:
            done_reasons = stats[power_name]['done_reason'][-avg_nb_new_games_per_agent:]
        else:
            done_reasons = stats[power_name]['done_reason'][-nb_new_power_games:]

        nb_engine = len([1 for reason in done_reasons if reason in ('', DoneReason.GAME_ENGINE.value)])
        nb_auto_draw = len([1 for reason in done_reasons if reason == DoneReason.AUTO_DRAW.value])
        nb_thrashing = len([1 for reason in done_reasons if reason == DoneReason.THRASHED.value])
        nb_phase_limit = len([1 for reason in done_reasons if reason == DoneReason.PHASE_LIMIT.value])

        feed_dict['done_engine/%s' % power_name] = 100. * nb_engine / max(1, len(done_reasons))
        feed_dict['done_auto_draw/%s' % power_name] = 100. * nb_auto_draw / max(1, len(done_reasons))
        feed_dict['done_thrashing/%s' % power_name] = 100. * nb_thrashing / max(1, len(done_reasons))
        feed_dict['done_phase_limit/%s' % power_name] = 100. * nb_phase_limit / max(1, len(done_reasons))

        # Additional reward functions
        for reward_fn in trainer.eval_reward_fns:
            feed_dict['%s/%s' % (reward_fn.name, power_name)] = compute_avg('eval_rewards',
                                                                            power_name,
                                                                            sub_key=reward_fn.name)

    # Average versions per day
    avg_version_per_day = 24 * 3600. * (version_id - trainer.stats['starting_version']) \
                          / float(max(1, current_time - trainer.starting_time))

    # General attributes
    feed_dict['nb_rl_agents'] = stats['nb_rl_agents']
    feed_dict['nb_steps'] = stats['nb_steps']
    feed_dict['nb_games'] = stats['nb_games']
    feed_dict['versions_per_day'] = avg_version_per_day
    feed_dict['mem_usage_mb'] = stats['memory_usage']
    feed_dict['version/player'] = stats['version/player']
    feed_dict['version/opponent'] = stats['version/opponent']

    # Algo evaluation tags
    for eval_tag in trainer.algorithm.get_evaluation_tags():
        feed_dict[eval_tag] = np.mean(stats[eval_tag]) if stats[eval_tag] else -1

    # Tensorboard merge computing
    feed_dict = {trainer.placeholders[key]: value for key, value in feed_dict.items()}
    summary_op = trainer.run_without_hooks(trainer.session, trainer.merge_op, feed_dict=feed_dict)

    # Writing to disk
    writer = trainer.writers[stats_mode]
    writer.add_summary(summary_op, version_id)
    writer.flush()

def display_progress(trainer):
    """ Display the current progress (version id and number of versions / day)
        :param trainer: A reinforcement learning trainer instance.
        :type trainer: diplomacy_research.models.training.reinforcement.trainer.ReinforcementTrainer
    """
    version_id = get_version_id(trainer)
    current_time = int(time.time())
    running_time = datetime.timedelta(seconds=int(current_time - trainer.starting_time))

    # Average versions per day
    avg_version_per_day = 24 * 3600. * (version_id - trainer.stats['starting_version']) \
                          / float(max(1, current_time - trainer.starting_time))
    spot_version_per_day = 24 * 3600 / float(max(1, current_time - trainer.last_version_time))
    trainer.last_version_time = current_time

    # Printing message
    message = []
    message += ['Version {:09}'.format(version_id)]
    message += ['Vers/Day Avg: {:6.2f} - Spot: {:6.2f}'.format(avg_version_per_day, spot_version_per_day)]
    message += ['Time {}'.format(running_time)]
    print(' | '.join(message))
