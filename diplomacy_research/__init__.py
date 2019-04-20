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
""" Diplomacy Research """
# Setting up root logger
import os
import logging
import sys

# Adding path to proto/ dir
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))

LOGGING_LEVEL = {'CRITICAL': logging.CRITICAL,
                 'ERROR': logging.ERROR,
                 'WARNING': logging.WARNING,
                 'INFO': logging.INFO,
                 'DEBUG': logging.DEBUG}.get(os.environ.get('DIPLOMACY_LOGGING', 'INFO'), logging.INFO)

# Defining root logger
ROOT_LOGGER = logging.getLogger('diplomacy_research')
ROOT_LOGGER.setLevel(LOGGING_LEVEL)
ROOT_LOGGER.propagate = False

# Adding output to stdout by default
STREAM_HANDLER = logging.StreamHandler(sys.stdout)
STREAM_HANDLER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
STREAM_HANDLER.setFormatter(FORMATTER)
ROOT_LOGGER.addHandler(STREAM_HANDLER)
