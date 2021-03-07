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

def select_file(name, candidate_files, default_file=None, match_prefix='', match_suffix='.py', file_type=''):
    use_default_file = False
    perfect_match = False
    if name is None:
        use_default_file = True
    else:
        matched_count = 0
        matched_files = []
        if match_prefix + name + match_suffix in candidate_files: # perfect match. return this file directly
            perfect_match_name = match_prefix + name + match_suffix
            perfect_match = True
            matched_files.append(perfect_match_name)
            matched_count += 1
        for file_name in candidate_files:
            if name in file_name:
                matched_files.append(file_name)
                matched_count += 1
        if matched_count==1: # only one matched file
            return matched_files[0]
        elif matched_count>1: # multiple files matched
            warning = 'multiple %s files matched: '%file_type
            for file_name in matched_files:
                warning += file_name
                warning += ' '
            warning += '\n'
            if perfect_match:
                warning += 'Using perfectly matched file: %s'%matched_files[0]
            else:
                warning += 'Using first matched file: %s'%matched_files[0]
            warnings.warn(warning)
            
            return matched_files[0]
        else:
            warnings.warn('No file matched name: %s. Trying using default %s file.'%(str(name), file_type))
            use_default_file = True

    if use_default_file:
        if default_file is None:
            if default_file in candidate_files:
                print('Using default %s file: %s'%(str(file_type), default_file))
                return default_file
            else:
                raise Exception('Did not find default %s file: %s'%(file_type, str(default_file)))
        else:
            raise Exception('Plan to use default %s file. But default %s file is not given.'%(file_type, file_type))
    return None