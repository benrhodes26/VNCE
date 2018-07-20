import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
code_dir_2 = '/home/ben/masters-project/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)
if code_dir_2 not in sys.path:
    sys.path.append(code_dir_2)

import numpy as np
from copy import deepcopy
from numpy import random as rnd

# noinspection PyPep8Naming
class Regulariser(metaclass=ABCMeta):

    def __init__(self, reg_param):
        self.reg_param = reg_param

    @abstractmethod
    def __call__(self, param):
        """Evaluate regularisation term at param"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, param):
        """Returns gradient of regularisation term w.r.t to the parameters"""
        raise NotImplementedError

class L1Regulariser(Regulariser):

    def __call__(self, param):
        return self.reg_param * np.sum(np.abs(param))

    def grad(self, param):
        return self.reg_param * np.sign(param)

