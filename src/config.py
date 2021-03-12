import torch
import sys
import re
import os

class Param:
    # empty class. 
    # by setting params as its attributes, instance of this class can serve as holder of params  
    pass

def get_sys_type():
    if re.match(r'win', sys.platform) is not None:
        sys_type = 'windows'
    elif re.match(r"linux", sys.platform) is not None:
        sys_type = 'linux'
    else:
        sys_type = 'unknown'
    return sys_type

def get_libs_path():
    return {
        'WWF-PC': 'A:/Software_Projects/Libs/',
        'srthu2': '/data4/wangweifan/Libs/',
    }

def init():
    # import paths of environment modules
    sys_type = get_sys_type()
    libs_path = get_libs_path()
    if sys_type in ['windows']:
        sys.path.append(libs_path["WWF-PC"])
    elif sys_type in ['linux']:
        sys.path.append(libs_path["srthu2"])
    else:
        raise Exception("Cannot add Libs path. Unknown system type.")
    
    # import paths.
    paths = []
    '''
    paths.append("./Models/")
    paths.append("./Optimizers")
    '''
    for path in paths:
        sys.path.append(path)
init()

import DataLoaders
import Optimizers
import Models
import Trainers

import config_ML
from utils import get_best_gpu, get_sys_type, ensure_path, load_param, set_default_attr

def main():
    options = Options(task="init")

def test_build_model():
    model_param = def_model_param()
    model_dict = build_model_dict(model_param, dict_data_loader={'data_set':'cifar_10'})

def def_model_param(**kw): # define necessary model parameters here.
    model_type = "RSLP_EI"
    model_name = "RSLP_EI"
    data_set = "cifar10"
    batch_size = 64
    optimize_method = "tp"

    E_num = 400
    I_num = 100
    iter_time = 10
    
    init_weight = {
        "i":["input", 1.0e-3],
        "f":["input", 1.0e-3],
        "r":["input", 1.0e-3],
    }

    #loss coeffients
    act_coeff = 0.0e-4
    class_coeff = 1.00
    class_loss_func = "CEL"
    weight_coeff = 1.0e-5
    hebb_coeff = 0.00 #Hebb loss
    stab_coeff = 0.00 #loss that punishes divergence of neural activity along time.
    noise_coeff = 0.0 #noise coefficient

    separate_ei = True
    cons_method = "abs" # methd to limit weight, which is used to implement Dale's Law.
    class_loss_func = "CEL" # CEL: Cross Entropy Loss, MSE: Mean Square Error
    if separate_ei:
        act_func_e = ["relu", 1.0]
        act_func_i = ["relu", 1.0]
        time_const_e = 0.2
        time_const_i = 0.4 # fast-spiking inhibitory neurons
        Dale = ["r", "f"] # weights that conform to Dale's Law  
    else:
        act_func = ["relu", 0.0]
        time_const = 0.2
    bias = False
    return load_param(locals())

def def_data_loader_param(**kw):
    data_set = "cifar10"
    batch_size = 64
    return load_param(locals())

def def_trainer_param(**kw):    
    epoch_num = 120
    save_model = True
    save_model_interval = 10
    save_model_path = "../saved_models/"
    return load_param(locals())

class Options: # An Option consists of 4 major components: model, optimizer, data_loader and trainer. 
    def __init__(self, task="init", dir_=None, items=None):
        #self.task = task
        self.dict = {}
        self.verbose = self.dict["verbose"] = True
        #self.dict_model, self.dict_optimizer, self.dict_data_loader, self.dict_trainer = None, None, None, None

        if task in ["load"]:
            self.load(dir_=dir_, items=items) #build Options from saved files.
        elif task in ["init"]:
            self.init() #build Options from scripts.
        #elif task is None or task in ["minimum", "none"]:
        #    self.init_minimum()
        else:
            raise Exception("Unsupported task:%s"%str(task))

    def build_minimum(self):
        self.set_device()

    def load(self, dir_="./config/", items=["model"]):
        with open(path_+"config", "rb") as f:
            self.dict = torch.load(f, map_location=torch.device("cpu"))
        
        self.model_type = self.dict["model_type"]
        #load model
        if "model" in items:
            with open(path_+"model", "rb") as f:
                self.dict_model = torch.load(f, map_location=torch.device("cpu"))
            self.build_model(load=True)
        # to be implemented
        return
    
    def init(self):
        # build loader dict
        param_data_loader = def_data_loader_param()
        self.dict_data_loader = build_data_loader_dict(param_data_loader)

        # build trainer dict
        param_trainer = def_trainer_param()
        self.dict_trainer = build_trainer_dict(param_trainer)

        # build optimizer dict
        param_optimizer = def_optimizer_param(dict_trainer=self.dict_trainer)
        self.dict_optimizer = build_optimizer_dict(param_optimizer, dict_trainer=self.dict_trainer)
    
        # build model dict
        kw = {"dict_data_loader":self.dict_data_loader}
        param_model = def_model_param(**kw)
        self.dict_model = build_model_dict(param_model, **kw)

    def build(self, items=["model", "optimizer", "data_loader", "trainer"]):
        self.set_device()
        self.build_data_loader()
        self.build_model()
        self.build_optimizer()
        if self.dict_trainer is not None:
            self.build_trainer()
            self.trainer.bind_data_loader(self.data_loader)
        #self.optimizer.bind_model(self.model)
        #self.trainer.bind_optimizer(self.optimizer)
        
    def build_data_loader(self):
        if self.dict_data_loader is not None:
            data_set = self.dict_data_loader["data_set"]
            if data_set in ["cifar10"]:
                self.dict_data_loader["data_dir"] = config_ML.dir_cifar10
                self.data_loader = DataLoaders.DataLoader_cifar10(self.dict_data_loader)
            elif data_set in ["mnist"]:
                self.dict_data_loader["data_dir"] = config_ML.dir_mnist
                self.data_loader = DataLoaders.DataLoader_mnist(self.dict_data_loader)

    def build_optimizer(self, load=False):
        if self.dict_optimizer is not None:
            optimize_method = self.dict_optimizer["method"]
            if optimize_method in ["tp", "TP", "target propagation"]:
                self.optimizer = Optimizers.Optimizer_TP(self.dict_optimizer, load=load, options=self)
            else:
                raise Exception("invalid optimizer method: %s"%(self.optimizer_method))

    def build_trainer(self, load=False):
        if self.dict_trainer is not None:
            self.trainer = Trainers.Trainer(self.dict_trainer, options=self)     
        
    def build_model(self, load=False):
        if self.dict_model is not None:
            model_type = self.dict_model["type"]
            if model_type in ["RNN_EI","EI_RNN", "RSLP_EI"]:
                self.model = Models.RSLP_EI(self.dict_model, options=self, load=load)
            else:
                raise Exception("Invalid model type: "+str(model_type))
    def set_device(self):
        self.sys_type = self.dict["sys_type"] = get_sys_type()
        try:
            self.device_str = get_best_gpu()
        except: # Time-out Exception.
            self.device_str = "cuda:0"
        if self.verbose:
            print("Options: Using device: %s"%(self.device_str))    
        self.dict["device"] = self.device_str
        self.device = torch.device(self.device_str)

    def save(self, save_path=None, items=None):
        path_ = "./config/" if save_path is None else save_path
        ensure_path(path_)
        self.model.save(path_, "model") # save model
        self.trainer.save(path_, "trainer")
        self.optimizer.save(path_, "optimizer") # save optmizer
        self.data_loader.save(path_, "data_loader") # save data_loader
    
def build_model_dict(param, **kw):
    model_type = param.model_type
    if model_type in ["RSLP_EI", "RNN_EI"]:
        return build_model_dict_RSLP_EI(param, **kw)
    else:
        raise Exception("Unknown model type:"+str(model_type))

def build_model_dict_RSLP_EI(param, **kw):
    # calculate full model parameters according to given necessaru parameters. 
    # return model dict that can be received by corresponding model class.
    
    #print("model param: ", end="")
    #print(dict_.keys())

    dict_data_loader = kw.get("dict_data_loader")
    if dict_data_loader is not None:
        data_set = dict_data_loader.get("data_set")
        if data_set is not None:
            if data_set in ["cifar_10", "cifar10"]:
                param.input_num = 32 * 32* 3
                param.output_num = 10

    '''
    # try to automatically create local variables from dict items. 
    # this seems impossible in python 3.
    for key, item in dict_.items():
        if key not in ["kw", "dict_", "item", "key"]:
            print("%s = \"%s\""%(key, item))
            #exec("%s = \"%s\""%(key, item))
            #eval("%s = %s"%(key, item))
            #locals()[key] = item
            #vars()[key] = item
            #__dict__[key] = item
            setattr(locals(), key, item)
    print(locals().keys())
    print(model_name)
    '''

    # prepare model dict
    dict_model = {
        "name": param.model_name,
        "type": param.model_type,
        "input_num": param.input_num,
        "output_num": param.output_num,
        "E_num": param.E_num,
        "I_num": param.I_num,
        "N_num": param.E_num + param.I_num,
        "init_weight": param.init_weight,
        "init_method":"zero",
        "iter_time": param.iter_time,
        "separate_ei": param.separate_ei,
        "cons_method": param.cons_method,
        "bias":param.bias,
        "loss_coeff":{
            "class": param.class_coeff,
            "class_loss_func":param.class_loss_func,
            "act": param.act_coeff,
            "weight": param.weight_coeff,
            "hebb": param.hebb_coeff,
            "stab": param.stab_coeff,
        },
        "N":{
            "E_num": param.E_num,
            "I_num": param.I_num,
            "N_num": param.E_num + param.I_num,
            "output_num": param.output_num,
            "init_weight": param.init_weight,
            "bias": param.bias,
            "noself": True,
            "mask": [],
            "separate_ei": param.separate_ei,
            "cons_method": param.cons_method,
            "noise_coeff": param.noise_coeff,
        },
        "mask": [],
        "input_mode": "endure"
    }

    if param.separate_ei:
        dict_model["Dale"] = param.Dale
        dict_model["N"]["Dale"] = param.Dale
        dict_model["N"]["time_const_e"] = param.time_const_e
        dict_model["N"]["time_const_i"] = param.time_const_i
        dict_model["N"]["act_func_e"] = param.act_func_e
        dict_model["N"]["act_func_i"] = param.act_func_i
    else:
        dict_model["N"]["time_const_e"] = param.time_const
        dict_model["N"]["time_const_i"] = param.time_const
        dict_model["N"]["act_func"] = param.act_func

    return dict_model

def def_optimizer_param(**kw):
    optimizer_type = "tp"
    lr = 1.0e-2
    lr_decoder = 10 * lr
    sub_optimizer_type = "sgd"

    lr_decay_method = "linear"
    milestones = [[0.50, 1.0],[0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]]
    return load_param(locals())

def build_optimizer_dict(param, **kw):
    optimizer_type = param.optimizer_type
    if optimizer_type in ["TP", "tp", "target propagation"]:
        return build_optimizer_dict_TP(param, **kw)

def build_optimizer_dict_TP(param, **kw):    
    dict_optimizer = {
        "method": "tp",
        "decoder":{
            "type":"mlp",
            "act_func":"relu",
            "bias": True,
        },
        "optimizer_forward":{
            "lr":param.lr,
            "type":param.sub_optimizer_type,
        },
        "optimizer_rec":{
            "lr":param.lr_decoder,
            "type":param.sub_optimizer_type,
        },
        "optimizer_out":{
            "lr":param.lr_decoder,
            "type":param.sub_optimizer_type,
        },
    }

    set_default_attr(param, "lr_decay_method", "None")
    if param.lr_decay_method in ["linear", "Linear"]:
        dict_optimizer["lr_decay"] = {
            "method": "linear",
            "milestones": param.milestones,
        }
        if kw.get("dict_trainer") is not None:
            dict_optimizer["lr_decay"]["epoch_num"] = kw.get("dict_trainer")["epoch_num"]
    elif param.lr_decay_method in ["log"]:
        dict_optimizer["lr_decay"] = { #learning rate decay
            "method":"log",
            "milestones":[[1.0, 1.0e-3]],
        }
    elif param.lr_decay_method in ["None"]:
        dict_optimizer["lr_decay"] = {"method":"None"}
    else:
        raise Exception("def_optimizer_param: Invalid lr_decay_method:%s"%(str(param.lr_decay_method)))
    return dict_optimizer

def build_data_loader_dict(param, **kw):
    #prepare data_loader dict
    if param.__dict__.get("pipeline") is None:
        setattr(param, "pipeline", "default")
    dict_data_loader = {
        "pipeline":"default",
        "data_set":param.data_set,
        "type":param.data_set,
        "batch_size":param.batch_size,
    }
    return dict_data_loader

def build_trainer_dict(param, **kw):
    #prepare trainer dict
    set_default_attr(param, "save_before_train", True)
    set_default_attr(param, "save_after_train", True)

    dict_trainer = {
        "epoch_num":param.epoch_num,
        "save_model":param.save_model,
        "save_model_path":param.save_model_path,
        "save_model_interval":param.save_model_interval,
        "save_before_train":param.save_before_train,
        "save_after_train":param.save_after_train,
    }
    return dict_trainer

if __name__=="__main__": # for debug. this script should not be executed in other cases.
    main()