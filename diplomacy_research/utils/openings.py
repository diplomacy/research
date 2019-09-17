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
""" Openings
    - Contains a list of standard openings for each power
"""
from numpy.random import choice

def get_standard_openings(power_name):
    """ Returns a list of standard openings for a given power"""
    return {
        'AUSTRIA': (
            ['A BUD - SER', 'F TRI - ALB', 'A VIE - GAL'],
            ['A BUD - SER', 'F TRI - ALB', 'A VIE - TRI'],
            ['A BUD - SER', 'F TRI - ALB', 'A VIE - BUD'],
            ['A BUD - SER', 'F TRI - VEN', 'A VIE - GAL'],
            ['A BUD - SER', 'F TRI H', 'A VIE - GAL'],
            ['A BUD - SER', 'F TRI H', 'A VIE - BUD'],
            ['A BUD - RUM', 'F TRI - ALB', 'A VIE - GAL'],
            ['A BUD - RUM', 'F TRI - ALB', 'A VIE - TRI'],
            ['A BUD - SER', 'F TRI - ALB', 'A VIE H'],
            ['A BUD - SER', 'F TRI - ALB', 'A VIE - TYR'],
            ['A BUD - SER', 'F TRI - ADR', 'A VIE - TRI'],
            ['A BUD - RUM', 'F TRI - ALB', 'A VIE - BUD'],
            ['A BUD - SER', 'F TRI H', 'A VIE - TYR'],
            ['A BUD - SER', 'F TRI - VEN', 'A VIE - BUD'],
            ['A BUD - SER', 'F TRI H', 'A VIE H'],
            ['A BUD - SER', 'F TRI - VEN', 'A VIE - TYR'],
            ['A BUD - RUM', 'F TRI H', 'A VIE - GAL'],
            ['A BUD - RUM', 'F TRI H', 'A VIE - BUD'],
            ['A BUD - SER', 'F TRI S A VEN', 'A VIE - GAL'],
            ['A BUD - GAL', 'F TRI - ALB', 'A VIE - TRI']),
        'ENGLAND': (
            ['F EDI - NWG', 'F LON - NTH', 'A LVP - YOR'],
            ['F EDI - NWG', 'F LON - NTH', 'A LVP - EDI'],
            ['F EDI - NTH', 'F LON - ENG', 'A LVP - YOR'],
            ['F EDI - NTH', 'F LON - ENG', 'A LVP - WAL'],
            ['F EDI - NWG', 'F LON - NTH', 'A LVP - WAL'],
            ['F EDI - NTH', 'F LON - ENG', 'A LVP - EDI'],
            ['F EDI - NWG', 'F LON - ENG', 'A LVP - WAL'],
            ['F EDI - NTH', 'F LON H', 'A LVP - YOR'],
            ['F EDI - NTH', 'F LON - ENG', 'A LVP H'],
            ['F EDI - NWG', 'F LON - ENG', 'A LVP - YOR']),
        'FRANCE': (
            ['F BRE - MAO', 'A MAR - SPA', 'A PAR - BUR'],
            ['F BRE - MAO', 'A MAR S A PAR - BUR', 'A PAR - BUR'],
            ['F BRE - MAO', 'A MAR - SPA', 'A PAR - PIC'],
            ['F BRE - ENG', 'A MAR - SPA', 'A PAR - PIC'],
            ['F BRE - ENG', 'A MAR - SPA', 'A PAR - BUR'],
            ['F BRE - MAO', 'A MAR - BUR', 'A PAR - PIC'],
            ['F BRE - PIC', 'A MAR - SPA', 'A PAR - BUR'],
            ['F BRE - MAO', 'A MAR H', 'A PAR - PIC'],
            ['F BRE - ENG', 'A MAR S A PAR - BUR', 'A PAR - BUR'],
            ['F BRE - MAO', 'A MAR - SPA', 'A PAR - GAS'],
            ['F BRE - MAO', 'A MAR - BUR', 'A PAR - GAS'],
            ['F BRE - ENG', 'A MAR - SPA', 'A PAR - GAS'],
            ['F BRE - MAO', 'A MAR H', 'A PAR - BUR'],
            ['F BRE - PIC', 'A MAR S A PAR - BUR', 'A PAR - BUR'],
            ['F BRE - ENG', 'A MAR - BUR', 'A PAR - PIC'],
            ['F BRE - MAO', 'A MAR - BUR', 'A PAR - BUR'],
            ['F BRE - PIC', 'A MAR - SPA', 'A PAR - GAS']),
        'GERMANY': (
            ['A BER - KIE', 'F KIE - DEN', 'A MUN - RUH'],
            ['A BER - KIE', 'F KIE - HOL', 'A MUN - RUH'],
            ['A BER - KIE', 'F KIE - DEN', 'A MUN - BUR'],
            ['A BER - KIE', 'F KIE - HOL', 'A MUN - BUR'],
            ['A BER - KIE', 'F KIE - HOL', 'A MUN H'],
            ['A BER - KIE', 'F KIE - DEN', 'A MUN H'],
            ['A BER - SIL', 'F KIE - DEN', 'A MUN - RUH'],
            ['A BER - KIE', 'F KIE - DEN', 'A MUN - TYR'],
            ['A BER - MUN', 'F KIE - DEN', 'A MUN - RUH'],
            ['A BER - PRU', 'F KIE - DEN', 'A MUN - SIL'],
            ['A BER - KIE', 'F KIE - HOL', 'A MUN - TYR'],
            ['A BER H', 'F KIE - DEN', 'A MUN - RUH'],
            ['A BER - PRU', 'F KIE - DEN', 'A MUN - RUH'],
            ['A BER - KIE', 'F KIE - BAL', 'A MUN - RUH'],
            ['A BER - KIE', 'F KIE - DEN', 'A MUN - SIL']),
        'ITALY': (
            ['F NAP - ION', 'A ROM - VEN', 'A VEN - TYR'],
            ['F NAP - ION', 'A ROM - APU', 'A VEN H'],
            ['F NAP - ION', 'A ROM - VEN', 'A VEN - PIE'],
            ['F NAP - ION', 'A ROM - NAP', 'A VEN H'],
            ['F NAP - ION', 'A ROM - APU', 'A VEN - TRI'],
            ['F NAP - ION', 'A ROM - VEN', 'A VEN - TRI'],
            ['F NAP - ION', 'A ROM - APU', 'A VEN S F TRI'],
            ['F NAP - ION', 'A ROM - APU', 'A VEN - PIE'],
            ['F NAP - TYS', 'A ROM - VEN', 'A VEN - PIE'],
            ['F NAP - TYS', 'A ROM - TUS', 'A VEN - PIE'],
            ['F NAP - ION', 'A ROM - APU', 'A VEN - TYR'],
            ['F NAP - TYS', 'A ROM - TUS', 'A VEN H'],
            ['F NAP - ION', 'A ROM - TUS', 'A VEN H'],
            ['F NAP - TYS', 'A ROM - VEN', 'A VEN - TYR'],
            ['F NAP - TYS', 'A ROM H', 'A VEN H'],
            ['F NAP - ION', 'A ROM - NAP', 'A VEN - TRI'],
            ['F NAP - ION', 'A ROM - NAP', 'A VEN - PIE'],
            ['F NAP - ION', 'A ROM H', 'A VEN H'],
            ['F NAP - TYS', 'A ROM H', 'A VEN - PIE'],
            ['F NAP - ION', 'A ROM - VEN', 'A VEN - APU'],
            ['F NAP - ION', 'A ROM - TUS', 'A VEN - PIE']),
        'RUSSIA': (
            ['A MOS - UKR', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - UKR', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - STP', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - UKR'],
            ['A MOS - STP', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - UKR', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR H'],
            ['A MOS - UKR', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR H'],
            ['A MOS - STP', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - UKR'],
            ['A MOS - UKR', 'F SEV - BLA', 'F STP/SC - FIN', 'A WAR - GAL'],
            ['A MOS - STP', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - SEV', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - UKR'],
            ['A MOS - SEV', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - UKR'],
            ['A MOS - UKR', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - SIL'],
            ['A MOS - SEV', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - SEV', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR H'],
            ['A MOS - STP', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR H'],
            ['A MOS - SEV', 'F SEV - ARM', 'F STP/SC - BOT', 'A WAR - UKR'],
            ['A MOS - UKR', 'F SEV - RUM', 'F STP/SC - FIN', 'A WAR - GAL'],
            ['A MOS - UKR', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - SIL'],
            ['A MOS - UKR', 'F SEV H', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - UKR', 'F SEV H', 'F STP/SC - BOT', 'A WAR H'],
            ['A MOS - WAR', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR - GAL'],
            ['A MOS - UKR', 'F SEV - BLA', 'F STP/SC - BOT', 'A WAR - LVN'],
            ['A MOS - LVN', 'F SEV - RUM', 'F STP/SC - BOT', 'A WAR H']),
        'TURKEY': (
            ['F ANK - BLA', 'A CON - BUL', 'A SMY - CON'],
            ['F ANK - BLA', 'A CON - BUL', 'A SMY - ARM'],
            ['F ANK - CON', 'A CON - BUL', 'A SMY - ANK'],
            ['F ANK - CON', 'A CON - BUL', 'A SMY H'],
            ['F ANK H', 'A CON - BUL', 'A SMY - CON'],
            ['F ANK - CON', 'A CON - BUL', 'A SMY - ARM'],
            ['F ANK - BLA', 'A CON - BUL', 'A SMY - ANK'],
            ['F ANK - BLA', 'A CON - BUL', 'A SMY H'],
            ['F ANK - ARM', 'A CON - BUL', 'A SMY - CON'])
    }.get(power_name, ())

def get_orders_from_gunboat_openings(power_name):
    """ Samples openings orders from the gunboat openings
        :param power_name: The name of the power we are playing
        :return: A list of opening orders for that power
    """
    gunboat_openings = {'AUSTRIA': {('A VIE - GAL', 'F TRI - ALB', 'A BUD - SER'): 6868,
                                    ('A VIE - TRI', 'F TRI - ALB', 'A BUD - SER'): 3033,
                                    ('A VIE - BUD', 'F TRI - ALB', 'A BUD - SER'): 1153,
                                    ('A VIE - GAL', 'F TRI - VEN', 'A BUD - SER'): 1969,
                                    ('A VIE - TYR', 'F TRI - VEN', 'A BUD - GAL'): 1615,
                                    ('A VIE - TYR', 'F TRI - ALB', 'A BUD - GAL'): 1513,
                                    ('A VIE - TYR', 'F TRI - ALB', 'A BUD - TRI'): 976,
                                    ('A VIE - GAL', 'F TRI H', 'A BUD - SER', ): 741,
                                    ('A VIE - TYR', 'F TRI - VEN', 'A BUD - RUM'): 535,
                                    ('A VIE - GAL', 'F TRI S A VEN', 'A BUD - SER'): 511},

                        'ENGLAND': {('F EDI - NWG', 'F LON - NTH', 'A LVP - YOR'): 6685,
                                    ('F EDI - NWG', 'F LON - NTH', 'A LVP - EDI'): 5647,
                                    ('F EDI - NTH', 'F LON - ENG', 'A LVP - YOR'): 4683,
                                    ('F EDI - NTH', 'F LON - ENG', 'A LVP - WAL'): 1744,
                                    ('F EDI - NTH', 'F LON - ENG', 'A LVP H'): 85,
                                    ('F EDI - NTH', 'F LON - ENG', 'A LVP - EDI'): 350,
                                    ('F EDI - NWG', 'F LON - NTH', 'A LVP - WAL'): 314,
                                    ('F EDI - NWG', 'F LON - ENG', 'A LVP - WAL'): 114,
                                    ('F EDI - NWG', 'F LON - ENG', 'A LVP - EDI'): 159,
                                    ('F EDI - NWG', 'F LON - ENG', 'A LVP - YOR'): 126},

                        'FRANCE': {('F BRE - MAO', 'A PAR - BUR', 'A MAR - SPA'): 4071,
                                   ('F BRE - MAO', 'A PAR - PIC', 'A MAR - SPA'): 1757,
                                   ('F BRE - MAO', 'A PAR - BUR', 'A MAR S A PAR - BUR'): 5298,
                                   ('F BRE - ENG', 'A PAR - PIC', 'A MAR - SPA'): 1128,
                                   ('F BRE - MAO', 'A PAR - PIC', 'A MAR - BUR'): 2277,
                                   ('F BRE - ENG', 'A PAR - BUR', 'A MAR S A PAR - BUR'): 1163,
                                   ('F BRE - MAO', 'A PAR - GAS', 'A MAR - BUR'): 939,
                                   ('F BRE - MAO', 'A PAR - BUR', 'A MAR - PIE'): 1029,
                                   ('F BRE - ENG', 'A PAR - PIC', 'A MAR - BUR'): 2736,
                                   ('F BRE - ENG', 'A PAR - BUR', 'A MAR - SPA'): 1904,
                                   ('F BRE - ENG', 'A PAR - BUR', 'A MAR - PIE'): 1121},

                        'GERMANY': {('F KIE - DEN', 'A MUN - RUH', 'A BER - KIE'): 8054,
                                    ('F KIE - HOL', 'A MUN - RUH', 'A BER - KIE'): 3607,
                                    ('F KIE - DEN', 'A MUN - BUR', 'A BER - KIE'): 3295,
                                    ('F KIE - HOL', 'A MUN - BUR', 'A BER - KIE'): 1229,
                                    ('F KIE - DEN', 'A MUN - RUH', 'A BER - SIL'): 284,
                                    ('F KIE - HOL', 'A MUN - TYR', 'A BER - SIL'): 2095,
                                    ('F KIE - DEN', 'A MUN H', 'A BER - KIE'): 439,
                                    ('F KIE - HOL', 'A MUN H', 'A BER - KIE'): 326,
                                    ('F KIE - HOL', 'A MUN - BUR', 'A BER - SIL'): 315,
                                    ('F KIE - DEN', 'A MUN - TYR', 'A BER - KIE'): 311},

                        'ITALY': {('F NAP - ION', 'A ROM - VEN', 'A VEN - TYR'): 5459,
                                  ('F NAP - ION', 'A ROM - APU', 'A VEN H'): 3048,
                                  ('F NAP - ION', 'A ROM - VEN', 'A VEN - PIE'): 2095,
                                  ('F NAP - ION', 'A ROM - NAP', 'A VEN H'): 775,
                                  ('F NAP - TYS', 'A ROM - VEN', 'A VEN - PIE'): 592,
                                  ('F NAP - ION', 'A ROM - APU', 'A VEN - TRI'): 865,
                                  ('F NAP - ION', 'A ROM - VEN', 'A VEN - TRI'): 1655,
                                  ('F NAP - ION', 'A ROM - APU', 'A VEN - TYR'): 1712,
                                  ('F NAP - ION', 'A ROM - APU', 'A VEN S F TRI'): 1924,
                                  ('F NAP - ION', 'A ROM - APU', 'A VEN - PIE'): 826},

                        'RUSSIA': {('F STP/SC - BOT', 'A MOS - UKR', 'A WAR - GAL', 'F SEV - BLA'): 9926,
                                   ('F STP/SC - BOT', 'A MOS - UKR', 'A WAR - GAL', 'F SEV - RUM'): 970,
                                   ('F STP/SC - BOT', 'A MOS - STP', 'A WAR - UKR', 'F SEV - BLA'): 1231,
                                   ('F STP/SC - BOT', 'A MOS - UKR', 'A WAR H', 'F SEV - BLA'): 498,
                                   ('F STP/SC - BOT', 'A MOS - STP', 'A WAR - UKR', 'F SEV - RUM'): 280,
                                   ('F STP/SC - BOT', 'A MOS - STP', 'A WAR - GAL', 'F SEV - BLA'): 1260,
                                   ('F STP/SC - BOT', 'A MOS - UKR', 'A WAR H', 'F SEV - RUM'): 414,
                                   ('F STP/SC - BOT', 'A MOS - STP', 'A WAR - GAL', 'F SEV - RUM'): 302,
                                   ('F STP/SC - FIN', 'A MOS - UKR', 'A WAR - GAL', 'F SEV - BLA', ): 934,
                                   ('F STP/SC - BOT', 'A MOS - SEV', 'A WAR - GAL', 'F SEV - RUM'): 363},

                        'TURKEY': {('F ANK - BLA', 'A SMY - CON', 'A CON - BUL'): 12382,
                                   ('F ANK - BLA', 'A SMY - ARM', 'A CON - BUL'): 4927,
                                   ('F ANK - CON', 'A SMY H', 'A CON - BUL'): 450,
                                   ('F ANK - CON', 'A SMY - ANK', 'A CON - BUL'): 1144,
                                   ('F ANK H', 'A SMY - CON', 'A CON - BUL'): 258,
                                   ('F ANK - CON', 'A SMY - ARM', 'A CON - BUL'): 477,
                                   ('F ANK - BLA', 'A SMY - ANK', 'A CON - BUL'): 165,
                                   ('F ANK - BLA', 'A SMY H', 'A CON - BUL'): 138,
                                   ('F ANK - ARM', 'A SMY - CON', 'A CON - BUL'): 133,
                                   ('F ANK S F SEV - BLA', 'A SMY - CON', 'A CON - BUL'): 91}}

    # Sampling for distribution
    if power_name not in gunboat_openings:
        return []
    orders, counts = list(gunboat_openings[power_name]), list(gunboat_openings[power_name].values())
    probs = [float(count) / sum(counts) for count in counts]
    order_ix = choice(range(len(orders)), size=1, p=probs)[0]
    return list(orders[order_ix])
