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
""" Memory Buffer - General Ops
    - Class responsible for performing general operations (e.g. get/set keys)
"""
import logging

# Constants
LOGGER = logging.getLogger(__name__)
__VERSION_ID__ = 'version.id'
__LOCK_VERSION_ID__ = 'lock.version.id'


# ------------      Version Id      ------------
def set_version_id(buffer, new_version_id):
    """ Sets the current version id
        :param buffer: An instance of the memory buffer.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    with buffer.redis.lock(__LOCK_VERSION_ID__, timeout=120):
        buffer.redis.set(__VERSION_ID__, int(new_version_id))

def get_version_id(buffer):
    """ Returns the current version id
        :param buffer: An instance of the memory buffer.
        :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
    """
    version_id = buffer.redis.get(__VERSION_ID__)
    if version_id is None:
        return 0
    return int(version_id)
