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
""" Memory Buffer - Queue
    - Class responsible for implementing a queue interface using the memory buffer
"""
import logging

# Constants
LOGGER = logging.getLogger(__name__)

class DistributedQueue():
    """ Distributed queue class"""

    def __init__(self, buffer, name):
        """ Constructor
            :param buffer: An instance of the memory buffer.
            :param name: The name of the queue.
            :type buffer: diplomacy_research.models.training.memory_buffer.MemoryBuffer
        """
        self.buffer = buffer
        self.name = name

    def qsize(self):
        """ Returns the size of the queue on the buffer """
        return self.buffer.redis.llen(self.name)

    def empty(self):
        """ Returns True if the queue is empty. False otherwise. """
        return not self.qsize()

    def put(self, *values):
        """ Puts an item into the queue. """
        self.buffer.redis.rpush(self.name, *values)

    def get(self, decode=True):
        """ Remove and return an item from the queue """
        item = self.buffer.redis.lpop(self.name)
        if isinstance(item, bytes) and decode:
            item = item.decode('utf-8')
        return item

    def put_nowait(self, *values):
        """ Alias for put() """
        return self.put(*values)

    def get_nowait(self):
        """ Alias for get() """
        return self.get()
