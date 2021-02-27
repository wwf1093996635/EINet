import os
import sys
import re
import argparse
import warnings
import importlib

import config_sys

#from config import Options
from Trainers import Trainer, Evaluator

from utils import build_model, build_optimizer, build_trainer, build_data_loader, get_device, remove_suffix, select_file, ensure_path

import Models
import Optimizers

parser = argparse.ArgumentParser(description='Parse args.')
parser.add_argument('--device', type=str, default=None, help='device')
parser.add_argument('--task', type=str, default=None, help='task to do')
parser.add_argument('--path', type=str, default=None, help='a path to current directory. must be used in some tasks.')
parser.add_argument('--optimizer', type=str, default=None, help='optimizer type. BP, TP, CHL, etc.')
parser.add_argument('--trainer', type=str, default=None, help='trainer type.')
parser.add_argument('--model', type=str, default=None, help='model type. RSLP, RMLP, RSLCNN, RMLCNN, etc.')
parser.add_argument('--data_loader', type=str, default=None, help='data loader type.')
parser.add_argument('--dicts_path', type=str, default=None, help='path to folder that stores param dict files.')
args = parser.parse_args()

def train(args=None, dicts_path=None, **kw):
    if args is None:
        args = kw.get('args')
    
    if dicts_path is None:
        if args.dicts_path is not None:
            dicts_path = args.dicts_path
        else:
            dicts_path = './params/'
    
    sys.path.append(dicts_path)

    model_dict, optimizer_dict, trainer_dict, data_loader_dict = get_param_dicts(args)

    trainer = build_trainer(trainer_dict)
    data_loader = build_data_loader(data_loader_dict)
    # model can be RSLP, RMLP, RCNN ...
    model = build_model(model_dict)
    # optimizer can be BP, TP or CHL optimizer.
    optimizer = build_optimizer(optimizer_dict)
    optimizer.bind_model(model)
    optimizer.bind_trainer(trainer)
    trainer.bind_model(model)
    trainer.bind_optimizer(optimizer)
    trainer.bind_data_loader(data_loader)
    trainer.train()    # the model needs some data from data_loader to get response properties.
    model.analyze(data_loader=data_loader)

def scan_param_files(path):
    files_path = os.listdir(path)
    pattern_model = re.compile(r'dict_model_(.*)\.py')
    pattern_optimizer = re.compile(r'dict_optimizer_(.*)\.py')
    pattern_trainer = re.compile(r'dict_trainer(.*)\.py')
    patern_data_loader = re.compile(r'dict_data_loader_(.*)\.py')
    model_files, optimizer_files, trainer_files, data_loader_files = [], [], [], []
    for file_name in files_path:
        #print(file_name)
        if pattern_model.match(file_name) is not None:
            model_files.append(file_name)
        elif pattern_optimizer.match(file_name) is not None:
            optimizer_files.append(file_name)
        elif pattern_trainer.match(file_name) is not None:
            trainer_files.append(file_name)
        elif patern_data_loader.match(file_name) is not None:
            data_loader_files.append(file_name)
        else:
            #warnings.warn('Unidentifiable param dict: %s'%str(file_name))
            pass

    # remove folders
    for files in [model_files, optimizer_files, trainer_files, data_loader_files]:
        for file in files:
            if os.path.isdir(file):
                warnings.warn('%s is a folder, and will be ignored.'%(path + file))
                files.remove(file)

    return model_files, optimizer_files, trainer_files, data_loader_files

def get_param_dicts(args):
    model_file, optimizer_file, trainer_file, data_loader_file = get_param_files(args)

    Model_Param = importlib.import_module(remove_suffix(model_file))
    model_dict = Model_Param.dict_

    Optimizer_Param = importlib.import_module(remove_suffix(optimizer_file))
    optimizer_dict = Optimizer_Param.dict_

    Trainer_Param = importlib.import_module(remove_suffix(trainer_file))
    trainer_dict = Trainer_Param.dict_

    Data_Loader_Param = importlib.import_module(remove_suffix(data_loader_file))
    data_loader_dict = Data_Loader_Param.dict_

    device = get_device(args)
    print('Using device: %s'%str(device))

    Model_Param.interact(model_dict, optimizer_dict, trainer_dict, data_loader_dict, device=device)
    Optimizer_Param.interact(model_dict, optimizer_dict, trainer_dict, data_loader_dict, device=device)
    Trainer_Param.interact(model_dict, optimizer_dict, trainer_dict, data_loader_dict, device=device)
    Data_Loader_Param.interact(model_dict, optimizer_dict, trainer_dict, data_loader_dict, device=device)

    return model_dict, optimizer_dict, trainer_dict, data_loader_dict

def get_param_files(args, files_path='./params/'):
    if not files_path.endswith('/'):
        files_path += '/'
    model_files, optimizer_files, trainer_files, data_loader_files = scan_param_files(files_path)

    # raise error if did not find any param dict
    if len(model_files)==0:
        raise Exception('No available model param dict in %s'%str(files_path))
    if len(optimizer_files)==0:
        raise Exception('No available model param dict in %s'%str(files_path))
    if len(trainer_files)==0:
        raise Exception('No available model param dict in %s'%str(files_path))
    if len(data_loader_files)==0:
        raise Exception('No available model param dict in %s'%str(files_path)) 

    model_file = select_file(args.model, model_files, default_file='dict_model_RSLP.py', 
        match_prefix='dict_model_', match_suffix='.py', file_type='model')
    #print(model_file)
    #input()

    optimizer_file = select_file(args.optimizer, optimizer_files, default_file='dict_optimizer_BP.py', 
        match_prefix='dict_optimizer_', match_suffix='.py', file_type='optimizer')

    trainer_file = select_file(args.trainer, trainer_files, default_file='dict_trainer.py', 
        match_prefix='dict_trainer_', match_suffix='.py', file_type='trainer')

    data_loader_file = select_file(args.data_loader, data_loader_files, default_file='dict_data_loader_cifar10.py', 
        match_prefix='dict_data_loader_', match_suffix='.py', file_type='data loader')

    return model_file, optimizer_file, trainer_file, data_loader_file

'''
def train():
    options = Options()
    options.build() #build DataLoader, Optimizer, Trainer, etc.    
    options.save(path_="./config/beforeTrain/")
    options.optimizer.scheduler_forward.verbose = True
    options.optimizer.scheduler_out.verbose = True
    options.optimizer.scheduler_rec.verbose = True
    options.trainer.train()
    options.save(path_="./config/afterTrain/")
'''

if __name__=="__main__":
    task = 'train' if args.task is None else args.task
    
    param_dicts = os.listdir('./params/')
    
    
    if task in ['copy', 'copy_files', 'copy_file']: # copy necessary files for training and 
        path = args.path
        ensure_path(args.path)
        file_list = [
            'Models',
            'Optimizers',
            'Trainers.py',
            'DataLoaders.py',
            'params',
            'config_sys.py',
            'utils.py',
            'utils_model.py',
            'main.py',
        ]

        for file in file_list:
            #shutil.copy2(file, path + file)
            if os.path.exists(path+file):
                os.system('rm -r %s'%(path+file))
            os.system('cp -r %s %s'%(file, path+file))
    elif task in ['train']:
        train(args)
    else:
        raise Exception('Invalid task: %s'%str(task))

def scan_models(name, path):
    if not os.path.exists(path):
        raise Exception('Path does not exist: %s'%str(path))
    if not os.path.isdir(path):
        raise Exception('%s is not a folder.'%str(path))
    if not path.endswith('/'):
        path += '/'
    files = os.listdir()
    
    if os.path.exists(path + name): # perfect match
        return name
    
    '''
    # remove folders
    for file in files:
        if os.path.isdir(path + file):
            #print('%s is a folder, and will be ignored.'%(path + file))
            files.remove(file)

    matched_files = []
    matched_count = 0
    for file in files:
        if name in file:
            matched_files.append(file)
            matched_count += 1

    if matched_count > 1:
        warnings.warn('Warning: multiple files match name')
    '''
    return select_file(name, files, default_file=None, match_prefix='', match_suffix='', file_type='saved_model')

'''
def load_and_evaluate():
    options = Options(mode="init", items=["data_loader"])
    options.build()
    model = load(options)
    evaluate(model, options)

def load(options):
    model = load_model(dir_="./saved_models/EINet_epoch=afterTrain", options=options)
    return model

def evaluate(model, options):
    evaluator = Evaluator(options=options)
    evaluator.bind_data_loader(options.data_loader)
    evaluator.bind_model(model)
    evaluator.evaluate()
'''



