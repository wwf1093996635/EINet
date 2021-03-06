import torch

from Models import *
from utils import get_from_dict

def load_model(dir_, options=None):
    if options is None:
        device_str="cpu"
    else:
        device_str = options.dict["device"]

    with open(dir_, "rb") as f:
        dict_model = torch.load(f, map_location=device_str)
    
    model_type = get_from_dict(dict_model, "type", default="RNN_EI", write_default=True)
    
    if model_type in ["RNN_EI"]:
        model = RNN_EI(dict_=dict_model, load=True, options=options)
    
    return model



    
    