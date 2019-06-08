from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np 
from pprint import pprint
from easydict import EasyDict as edict 

class Config(object):
  
  
  def __init__(self, **kwargs):
    self.learning_rate = 0.1

    self.momentum = 0.9

    self.nesterov = True

    self.weight_decay = 1e-4

    self.gamma = 0.1

    self.checkpoint = './checkpoints/'

    self.ckpt_interval = 20

    self.milestones = [150, 225]

    self.max_epoch = 300

    self.data_dir = '../data'
    self.cuda = True
    self.dataset = 'cifar10'

    self.no_aug = False
    self.train_batch = 128
    self.test_batch = 128
    self.workers = 4
    self.manualSeed = 2345

    self.K = 2

    if len(kwargs)>0:
      self._parse(kwargs)
    

  def _parse(self, **kwargs):
    state_dict = self._state_dict()
    for k,v in kwargs._state_dict().items():
      if k not in state_dict:
        raise ValueError('Unknown Option: "--%s"'%k)
      setattr(self, k, v)
    
    print('----- user config -----------')
    # pprint(self._state_dict())
    print('-----------end----------------')

  def _state_dict(self):
    return {k: getattr(self,k) for k,_ in self.__dict__.items() if not k.startswith('__')}

  def __repr__(self):
    s = 'Config object:\n'
    s += '; '.join(['%s: %s'%(k,str(getattr(self, k))) for k,_ in self.__dict__.items() if not k.startswith('__')])
    return s
    # pprint(self._state_dict())

#cfg = Config()  
def _merge_a_into_b(a, b):

  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    
    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
