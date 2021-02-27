
import pickle
import time
import os

import random
import copy
import abc
import warnings

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

#from functions import evaluate, evaluate_iter, pytorch_info
import utils
from utils import get_from_dict, ensure_path
from utils_model import build_optimizer

class Optimizer(abc.ABC):
    def __init__(self, dict_=None, load=False):
        self.dict = dict_
        self.verbose = get_from_dict(self.dict, "verbose", default=True, write_default=True)
        #self.device = self.options.device
        #self.options = options
        #self.model = options.model
    def bind_model(self, model):
        if self.model is not None:
            if self.options.verbose:
                print("Optimizer: binding new model. warning: this optimizer has already bound a model, and it will be detached.")
        else:
            self.model = model
    def detach_model(self, model):
        if self.model is None:
            if self.options.verbose:
                print("Optimizer: detaching model. this optimizer hasn't bound a model.")
        else:
            self.model = None
    def bind_model(self, model):
        self.model = model

    #@abc.abstractmethod # must be implemented by child class.
    def train(self):
        return
    def save(self, save_path="./", save_name=None, save_model=False):
        ensure_path(save_path)

        save_name = "optimizer_%s"%self.dict["method"] if save_name is None else save_name
        with open(save_path + save_name, "wb") as f:
            net = self.to(torch.device("cpu"))
            torch.save(net.dict, f)
            net = self.to()

        if save_model:
            if hasattr(self, 'model'):
                self.model.save(save_path, '(model)' + save_name)
            else:
                warnings.warn('')


