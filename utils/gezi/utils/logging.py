#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   logging.py
#        \author   chenghuige  
#          \date   2016-09-24 09:25:56.796006
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# flags = tf.app.flags
# FLAGS = flags.FLAGS

import sys 
import os
import inspect

# import sys
# if 'absl.logging' in sys.modules:
#   import absl.logging
#   absl.logging.set_verbosity('info')
#   absl.logging.set_stderrthreshold('info')
#   # and any other apis you want, if you want

# import coloredlogs
import logging
# coloredlogs.install()
import logging.handlers
from colorama import Fore, Style

from icecream import install
install()
from icecream import ic
ic.configureOutput(includeContext=True)

import gezi
from tqdm.auto import tqdm

# https://github.com/tqdm/tqdm/issues/313
class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)  # , file=sys.stderr) TODO could not use end=''.. not show anything
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
    
# https://github.com/tqdm/tqdm/issues/744        
class DummyTqdmFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len( x.rstrip() ) > 0:
            tqdm.write( x, file=self.file, end='' )

    def flush(self):
        return getattr( self.file, "flush", lambda: None )()

_logger = logging.getLogger('gezi')
_logger2 = logging.getLogger('gezi2')
   
#_handler = logging.StreamHandler()
#_handler.setLevel(logging.INFO)
#_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
#_logger.addHandler(_handler)
#_logger.setLevel(logging.INFO)

log = _logger.log
#debug = _logger.debug
#error = _logger.error
#fatal = _logger.fatal
#info = _logger.info
#warn = _logger.warn
#warning = _logger.warning  

#info2 = _logger2.info

hvd = None

def set_hvd(hvd_):
  global hvd 
  hvd = hvd_

def set_dist(dist):
  set_hvd(dist)

def info(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.info(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def info2(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger2.info(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def sinfo(x, prefix='------------'):
  if not hvd or hvd.rank() == 0:
    try:
      callers_local_vars = inspect.currentframe().f_back.f_locals.items()
      _logger.debug('{}{}:{}'.format(prefix, str([k for k, v in callers_local_vars if v is x][0]), x)) 
    except Exception:
      pass

def sprint(x, prefix='------------'):
  if not hvd or hvd.rank() == 0:
    try:
      callers_local_vars = inspect.currentframe().f_back.f_locals.items()
      _logger.debug('{}{}:{}'.format(prefix, str([k for k, v in callers_local_vars if v is x][0]), x)) 
    except Exception:
      pass

def fatal(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.fatal(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def error(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.error(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def debug(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.debug(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def warn(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.warn(' '.join("{}".format(a) for a in args))
    except Exception:
      pass

def warning(*args):
  if not hvd or hvd.rank() == 0:
    try:
      _logger.warning('WARNING: %s' % (' '.join("{}".format(a) for a in args)))
    except Exception:
      pass

def ice(*args):
  if ic.enabled:
    includeContext = ic.includeContext
    ic.configureOutput(includeContext=False)
    debug(*args)
    ic(args)
    ic.configureOutput(includeContext=includeContext)
  else:
    info(*args)

from datetime import timedelta
import time

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class ElapsedFormatter():
  def __init__(self):
    self.start_time = time.time()

  def format(self, record):
    elapsed_seconds = record.created - self.start_time
    #using timedelta here for convenient default formatting
    elapsed = timedelta(seconds = elapsed_seconds)
    # log_colors[record.levelno]
    # return f'{bcolors.OKBLUE}{gezi.now_time()} {str(elapsed)[:-7]} {record.getMessage()}{bcolors.ENDC}'
    if gezi.get('COLOR_LOGGING'):
      return f'{Fore.BLUE}{gezi.now_time()} {str(elapsed)[:-7]} {record.getMessage()}{Style.RESET_ALL}'
    else:
      return f'{gezi.now_time()} {str(elapsed)[:-7]} {record.getMessage()}'
      
_logging_file = None
_logging_file2 = None
inited = None

def _get_handler(file, formatter, split=True, split_bytime=False, mode = 'a', level=logging.INFO):
  #setting below will set root logger write to _logging_file
  #logging.basicConfig(filename=_logging_file, level=level, format=None)
  #logging.basicConfig(filename=_logging_file, level=level)
  #save one per 1024k/4, save at most 10G 1024
  if split:
    if not split_bytime:
      file_handler = logging.handlers.RotatingFileHandler(file, mode=mode, maxBytes=1024*1024/4, backupCount=10240*4)
    else:
      file_handler = logging.handlers.TimedRotatingFileHandler(file, when='H', interval=1, backupCount=1024)
      file_handler.suffix = "%Y%m%d-%H%M"
  else:
    file_handler = logging.FileHandler(_logging_file, mode=mode)  
  file_handler.setLevel(level)  
  file_handler.setFormatter(formatter)

  # def decorate_emit(fn):
  # # add methods we need to the class
  #     def new(*args):
  #         levelno = args[0].levelno
  #         if(levelno >= logging.CRITICAL):
  #             color = '\x1b[31;1m'
  #         elif(levelno >= logging.ERROR):
  #             color = '\x1b[31;1m'
  #         elif(levelno >= logging.WARNING):
  #             color = '\x1b[33;1m'
  #         elif(levelno >= logging.INFO):
  #             color = '\x1b[32;1m'
  #         elif(levelno >= logging.DEBUG):
  #             color = '\x1b[35;1m'
  #         else:
  #             color = '\x1b[0m'
  #         # add colored *** in the beginning of the message
  #         args[0].msg = "{0}***\x1b[0m {1}".format(color, args[0].msg)

  #         # new feature i like: bolder each args of message 
  #         args[0].args = tuple('\x1b[1m' + arg + '\x1b[0m' for arg in args[0].args)
  #         return fn(*args)
  #     return new
  # file_handler.emit = decorate_emit(file_handler.emit)

  return file_handler

def set_dir(path=None, file='log.html', logtostderr=True, logtofile=True, split=True, split_bytime=False, quiet=False, level=logging.INFO, mode='a'):
  global _logger, _logging_file, inited
  # if _logging_file is None:
  
  #formatter = logging.Formatter("%(asctime)s %(message)s",
  #                          "%Y-%m-%d %H:%M:%S")
  formatter = ElapsedFormatter()

  if logtostderr and not inited:
    # handler = logging.StreamHandler()
    # handler.setLevel(level)
    # handler.setFormatter(formatter)
    # #handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
    handler = TqdmHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    
    handler2 = TqdmHandler()
    handler2.setLevel(level)
    handler2.setFormatter(formatter)
    if quiet:
      handler2.setLevel(logging.ERROR)    
    _logger2.addHandler(handler2)
    
    # some how new tf cause to logg twice.. 
    # https://stackoverflow.com/questions/19561058/duplicate-output-in-simple-python-logging-configuration
    _logger.propagate = False 
    _logger2.propagate = False

    # coloredlogs.install(level=level, logger=_logger)
    # coloredlogs.install(level=level, logger=_logger2)
  
  if not path:
    path = '/tmp/gezi.log'
    if not os.path.isdir(path):
      os.makedirs(path)
  _logging_file = '%s/%s' % (path, file)
  _logging_file2 = '%s/log.txt' % path

  
  # _logger.setLevel(level)
  # _logger2.setLevel(level)
  _logger.setLevel(logging.DEBUG)
  _logger2.setLevel(logging.DEBUG)
  
  if logtofile:
    gezi.try_mkdir(path)
    file_handler = _get_handler(_logging_file, formatter, split, split_bytime, mode, level)
    file_handler2 = _get_handler(_logging_file2, formatter, split, True, mode, level)
    #file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
    file_handler.setLevel(logging.DEBUG)
    file_handler2.setLevel(level)
    _logger.addHandler(file_handler)
    _logger2.addHandler(file_handler2)
    
  inited = True

# TODO logtostderr=False not work correctly with all infos output to screen, now you an set high level to avoid output to screen
def init(path=None, file='log.html', logtostderr=True, logtofile=True, split=True, split_bytime=False, level=logging.INFO, quiet=False, mode='a'):
  set_dir(path=path, mode=mode, file=file, logtostderr=logtostderr, logtofile=logtofile, split=split, split_bytime=split_bytime, quiet=quiet, level=level)

def vlog(level, *args, **kwargs):
  _logger.log(level, ' '.join("{}".format(a) for a in args), **kwargs)

def get_verbosity():
  """Return how much logging output will be produced."""
  return _logger.getEffectiveLevel()

def set_verbosity(verbosity):
  """Sets the threshold for what messages will be logged."""
  _logger.setLevel(verbosity)

def get_logging_file():
  return _logging_file
