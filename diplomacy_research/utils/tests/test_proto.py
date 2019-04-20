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
""" Tests for the utils/proto methods """
from diplomacy_research.proto.diplomacy_proto.game_pb2 import Message
from diplomacy_research.utils.proto import proto_to_dict, proto_to_zlib, proto_to_bytes
from diplomacy_research.utils.proto import dict_to_proto, zlib_to_proto, bytes_to_proto

def _get_message():
    """ Returns a test message proto """
    message_proto = Message()
    message_proto.sender = 'FRANCE'
    message_proto.recipient = 'ENGLAND'
    message_proto.time_sent = 1234
    message_proto.phase = 'S1903M'
    message_proto.message = 'This is the message'
    return message_proto

def _compare_messages(message_proto_1, message_proto_2):
    """ Compares 2 messages """
    assert message_proto_1.sender == message_proto_2.sender
    assert message_proto_1.recipient == message_proto_2.recipient
    assert message_proto_1.time_sent == message_proto_2.time_sent
    assert message_proto_1.phase == message_proto_2.phase
    assert message_proto_1.message == message_proto_2.message

def test_to_from_dict():
    """ Tests proto_to_dict and dict_to_proto """
    message_proto = _get_message()
    message_dict = proto_to_dict(message_proto)
    new_message_proto = dict_to_proto(message_dict, Message)
    _compare_messages(message_proto, new_message_proto)

def test_to_from_bytes():
    """ Tests proto_to_bytes and bytes_to_proto """
    message_proto = _get_message()
    message_bytes = proto_to_bytes(message_proto)
    new_message_proto = bytes_to_proto(message_bytes, Message)
    _compare_messages(message_proto, new_message_proto)

def test_to_from_zlib():
    """ Tests proto_to_zlib and zlib_to_proto """
    message_proto = _get_message()
    message_zlib = proto_to_zlib(message_proto)
    new_message_proto = zlib_to_proto(message_zlib, Message)
    _compare_messages(message_proto, new_message_proto)
