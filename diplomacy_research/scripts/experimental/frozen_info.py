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
""" Returns information about a frozen graph
    Argument: File path to frozen graph
"""
import sys
from diplomacy_research.utils.checkpoint import load_frozen_graph

def print_frozen_tags(frozen_graph_path):
    """ Returns the model name and commit hash used to generate the frozen graph
        :param frozen_graph_path: The path to the frozen graph
        :return: A tuple consisting of 1) The model name used to generate the frozen graph
                                       2) The commit hash used to generate the frozen graph
    """
    graph, _ = load_frozen_graph(frozen_graph_path)

    # Looping over all tags
    print('-' * 80)
    tags = sorted([key for key in graph.get_all_collection_keys() if 'tag/' in key])
    for tag_name in tags:
        print('%s: %s' % (tag_name, str(graph.get_collection(tag_name)[0])))
    print('-' * 80)


if __name__ == '__main__':
    # Getting tag information from frozen graph
    # Syntax: frozen_info.py <frozen_graph_path>
    if len(sys.argv) == 2 and sys.argv[1][-3:] == '.pb':
        print_frozen_tags(sys.argv[1])
        exit(0)

    # Invalid syntax
    print('Syntax: frozen_info.py <frozen_file_path.pb>')
    exit(-1)
