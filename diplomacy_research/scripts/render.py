#!/usr/bin/env python3
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
""" Renders a same tournament game
    Argument: File path to .json in history folder
"""
import argparse
import os
import multiprocessing
import shutil
from diplomacy import Game
import ujson as json
from diplomacy_research.proto.diplomacy_proto.game_pb2 import SavedGame as SavedGameProto
from diplomacy_research.utils.proto import proto_to_dict, read_next_proto

def render_saved_game(saved_game, output_dir, prefix=''):
    """ Renders a specific saved game
        :param saved_game: The saved game to render
        :param output_dir: The output directory where to save the rendering
        :param prefix: An optional prefix to add before the game id
    """
    if prefix:
        output_dir = os.path.join(output_dir, prefix + '_' + saved_game['id'])
    else:
        output_dir = os.path.join(output_dir, saved_game['id'])
    nb_phases = len(saved_game['phases'])
    svg_count = 0

    # Checking if already generated
    # Otherwise, regenerating completely
    if os.path.exists(output_dir):
        nb_svg = len([os.path.join(output_dir, file) for file in os.listdir(output_dir) if file[-4:] == '.svg'])
        if nb_svg == 2 * nb_phases:
            print('Rendered {} (Skipped)'.format(saved_game['id']))
            return
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Creating a Game to replay all orders, and a new Game object per phase to validate
    entire_game = Game()
    if saved_game['phases']:
        entire_game.set_state(saved_game['phases'][0]['state'])

    # Rendering
    for phase in saved_game['phases']:
        phase_game = Game()

        # Setting state
        state = phase['state']
        phase_game.set_state(state)
        entire_game.note = phase_game.note

        # Setting orders
        phase_game.clear_orders()
        orders = phase['orders']
        for power_name in orders:
            phase_game.set_orders(power_name, orders[power_name])
            entire_game.set_orders(power_name, orders[power_name])

        # Validating that we are at the same place
        for power_name in orders:
            assert sorted(phase_game.get_units(power_name)) == sorted(entire_game.get_units(power_name))
            assert sorted(phase_game.get_centers(power_name)) == sorted(entire_game.get_centers(power_name))

        # Rendering with and without orders
        with open(os.path.join(output_dir, '%03d%s' % (svg_count, '.svg')), 'w') as file:
            file.write(entire_game.render(incl_orders=False))
        svg_count += 1
        with open(os.path.join(output_dir, '%03d%s' % (svg_count, '.svg')), 'w') as file:
            file.write(entire_game.render(incl_orders=True))

        # Processing (for entire game)
        svg_count += 1
        entire_game.process()

    print('Rendered {}'.format(saved_game['id']))

# =========================================
# -------      JSON RENDERING    ----------
# =========================================
def render_json(file_path):
    """ Renders a specific json file
        :param file_path: The full path to the json file
        :return: Nothing, but creates a directory (file_path without '.json') containing the rendered images
    """
    dir_path = os.path.dirname(file_path)

    # Aborting if file doesn't exist
    if not os.path.exists(file_path):
        print('File {} does not exist.'.format(file_path))
        return

    # Loading saved game
    file_content = open(file_path, 'r').read()
    saved_game = json.loads(file_content)

    # Rendering
    render_saved_game(saved_game, dir_path)

def render_multi_json_per_folder(history_dir, nb_json_per_folder):
    """ Finds all subfolders under history and renders 'nb_jsons' games in each subfolder found
        :param history_dir: The full path to the history folder
        :param nb_json_per_folder: The number of jsons to render per subfolder
        :return: Nothing
    """
    jsons_to_render = []

    # Finding files to render
    subfolders = [os.path.join(history_dir, path)
                  for path in os.listdir(history_dir)
                  if os.path.isdir(os.path.join(history_dir, path))]
    for folder in subfolders:
        json_games = sorted([os.path.join(folder, json_filename)
                             for json_filename in os.listdir(folder)
                             if json_filename[-5:] == '.json'])
        json_games = json_games[:nb_json_per_folder]
        for json_path in json_games:
            jsons_to_render += [json_path]

    # Running over multiple processes
    nb_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(nb_cores) as pool:
        pool.map(render_json, jsons_to_render)

# =========================================
# -------     PROTO RENDERING    ----------
# =========================================
def render_saved_game_proto(saved_game_proto, output_dir, prefix='', json_only=False):
    """ Renders a saved game proto
        :param saved_game_proto: A `.proto.game.SavedGame` object
        :param output_dir: The output directory where the save the renderings
        :param prefix: An optional prefix to add before the game id
        :param json_only: Indicates we only want to extract the underlying JSON
    """
    saved_game = proto_to_dict(saved_game_proto)
    if json_only:
        os.makedirs(os.path.join(output_dir, 'json'), exist_ok=True)
        output_path = os.path.join(output_dir, 'json', prefix + '_' + saved_game['id'] + '.json')
        with open(output_path, 'w') as file:
            file.write(json.dumps(saved_game))
        print('Saved JSON for {}'.format(saved_game['id']))
    else:
        render_saved_game(saved_game, output_dir, prefix)

def render_proto_file(file_path, args, compressed=True):
    """ Renders all saved game proto in a proto file
        :param file_path: The path to the proto file
        :param args: The parsed command line arguments
        :param compressed: Boolean that indicates if compression was used.
    """
    dir_path = os.path.dirname(file_path)
    game_count = 0

    # Aborting if file doesn't exist
    if not os.path.exists(file_path):
        print('File {} does not exist.'.format(file_path))
        return

    # Processing filter
    games_to_render = []
    if args.filter:
        for part in args.filter.split(','):
            if '-' in part:
                start, stop = part.split('-')
                games_to_render += list(range(int(start), int(stop) + 1))
            elif ':' in part:
                start, stop, step = part.split(':')
                games_to_render += list(range(int(start), int(stop) + 1, int(step)))
            else:
                games_to_render += [int(part)]

    # Rendering each game in the proto file
    with open(file_path, 'rb') as file:
        while True:
            saved_game_proto = read_next_proto(SavedGameProto, file, compressed)
            if saved_game_proto is None:
                break
            game_count += 1
            if game_count in games_to_render or (not games_to_render and not args.count):
                print('(Game #%d) ' % game_count, end='')
                render_saved_game_proto(saved_game_proto, dir_path, prefix='%05d' % game_count, json_only=args.json)
            if game_count % 100 == 0 and args.count:
                print('... %d games found so far.' % game_count)

    # Printing the number of games in the proto file
    if args.count:
        print('Found %d games in the proto file.' % game_count)


# =========================================


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Render some saved games.')
    PARSER.add_argument('--count', action='store_true', help='Count the number of games in the file')
    PARSER.add_argument('--json', action='store_true', help='Only extract jsons without rendering the games')
    PARSER.add_argument('--filter', help='Only render some games e.g. 1-5,6,8,10:100:2')
    PARSER.add_argument('--nb_per_folder', type=int, default=0, help='The number of games per folder to generate')
    PARSER.add_argument('file_path', help='The file path containing the saved games.')
    ARGS = PARSER.parse_args()

    # Rendering a single JSON
    # Syntax: render.py <json path>
    if ARGS.file_path[-5:] == '.json':
        render_json(ARGS.file_path)
        exit(0)

    # Render a series of game in a .pb file
    # Syntax: render.py <pb path>
    if ARGS.file_path[-3:] == '.pb':
        render_proto_file(ARGS.file_path, ARGS, compressed=False)
        exit(0)
    if ARGS.file_path[-4:] == '.pbz':
        render_proto_file(ARGS.file_path, ARGS, compressed=True)
        exit(0)

    # Rendering a certain number of JSON per folder
    # Syntax: render.py <history/> --nb_per_folder <# of json per folder to generate>
    if os.path.exists(ARGS.file_path) and ARGS.nb_per_folder:
        render_multi_json_per_folder(ARGS.file_path, ARGS.nb_per_folder)
        exit(0)

    # Invalid syntax
    PARSER.print_help()
    exit(-1)
