import warnings
import re
from utils_ import *
from utils_ import get_best_gpu

def build_optimizer(dict_):
    import Optimizers
    type_ = dict_['type']
    if type_ in ['BP', 'bp']:
        return Optimizers.Optimizer_BP(dict_)
    elif type_ in ['CHL', 'chl']:
        return Optimizers.Optimizer_CHL(dict_)
    elif type_ in ['TP', 'tp']:
        return Optimizers.Optimizer_TP(dict_)
    else:
        raise Exception('Invalid optimizer type: %s'%str(type_))

def build_model(dict_, load=False):
    import Models
    type_ = dict_['type'] # type of the model to be built
    if type_ in ['RSLP', 'RSLP_EI']: # single-layer recurrent perceptron
        return Models.RSLP(dict_, load=load)
    elif type_ in ['RMLP']: # multi-layer recurrent perceptron
        return Models.RMLP_EI(dict_, load=load)
    else:
        raise Exception('Invalid model type: %s'%str(type_))

def build_trainer(dict_, load=False):
    import Trainers
    return Trainers.Trainer(dict_, load=load)

def build_data_loader(dict_, load=False):
    import DataLoaders
    type_ = dict_['type']
    if type_ in ['cifar10']:
        return DataLoaders.DataLoader_cifar10(dict_, load=load)
    else:
        raise Exception('Invalid data loader type: %s'%str(type_))

def get_device(args):
    if hasattr(args, 'device'):
        if args.device not in [None, 'None']:
            print(args.device)
            return args.device
    return get_best_gpu()