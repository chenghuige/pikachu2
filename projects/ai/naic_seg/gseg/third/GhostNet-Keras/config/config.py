# -- coding: utf-8 --

import argparse
import json

import os
from bunch import Bunch

def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = Bunch(config_dict)

    return config, config_dict



def get_train_args():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='C',
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser
