import torch
import torch.nn as nn
import torch.nn.functional as F

#training parameters.
from anal_functions import *
from utils_model import *
import random

from RCNNBlock import RCNNBlock

class RCNN_EI(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(RCNN_EI, self).__init__()
        self.dict = dict_
        self.layers = []
        if load:
            self.dict=torch.load(f, map_location=device) 
            for layer_index in range(self.dict["layers_num"]):
                layer_name = "layer" + str(layer_index)
                layer_type = self.dict[layer_name]["type"]
                layer_dict = self.dict[layer_name]
                if layer_type in ["rconv"]:
                    layer = RCNNBlock(dict_=layer_dict, load=True)
                elif layer_type in ["conv"]:
                    layer = nn.Conv2d(layer_dict["N_num"], layer_dict["output_num"], layer_dict["kernel_size"], layer_dict["stride"], layer_dict["padding"]) #in, out, kernel_size, stride, padding
                    layer.load_state_dict(layer_dict["state_dict"])
                elif layer_type in ["linear", "fc"]:
                    layer = nn.Linear(layer_dict["input_num"], layer_dict["output_num"], layer_dict["bias"])
                    layer.load_state_dict(layer_dict["state_dict"])
                else:
                    layer = get_layer(layer_type, layer_dict)             
                self.layers.append(layer)
                if(layer_type not in ["global_avg"]):
                    self.add_module(layer_name, layer)
        else:
            for layer_index in range(self.dict["layers_num"]):
                layer_name = "layer" + str(layer_index)
                layer_type = self.dict[layer_name]["type"]
                layer_dict = self.dict[layer_name]
                if(layer_type in ["rconv"]):
                    layer = RCNNBlock(dict_=layer_dict, load=False)
                    self.dict[layer_name] = layer.dict
                elif(layer_type in ["conv"]):
                    layer = nn.Conv2d(layer_dict["N_num"], layer_dict["output_num"], layer_dict["kernel_size"], layer_dict["stride"], layer_dict["padding"])
                    layer_dict["state_dict"] = layer.state_dict()
                elif(layer_type in ["linear", "fc"]):
                    layer = nn.Linear(layer_dict["input_num"], layer_dict["output_num"], layer_dict["bias"])
                    layer_dict["state_dict"] = layer.state_dict()
                else:
                    layer = get_layer(layer_type, layer_dict)
                self.layers.append(layer)
                if(layer_type not in ["global_avg"]):
                    self.add_module(layer_name, layer)

        self.layer_types = []
        self.rconvs = []
        for layer_index in range(self.dict["layers_num"]):
            type = self.dict["layer"+str(layer_index)]["type"]
            self.layer_types.append(type)
            if(type=="rconv"):
                self.rconvs.append(layer_index)

        if(self.dict["class_loss_func"]=="CEL"):
            self.class_loss_func = torch.nn.CrossEntropyLoss()
        elif(self.dict["class_loss_func"]=="MSE"):
            self.class_loss_func = torch.nn.MSELoss()
        self.loss_count = 0
        self.loss_list = {"class":0.0, "act":0.0, "weight":0.0}
        if(hebb_coeff != 0.0):
            self.loss_list["hebb"] = 0.0
        
        self.dict["cache"] = {}
        self.cache = self.dict["cache"]
    def get_layer_name(self, index):
        return "layer" + str(index)
    def get_layer_index(self, layer_name):
        pattern = "layer(\d+)"
        result = re.search(r''+pattern, layer_name)
        layer_index = None
        if(result is not None):
            try:
                layer_index = int(result.group(1))
            except Exception:
                print("error in getting layer index from %s."%(layer_name))
        return layer_index
    def forward(self, x): #(batch_size, pixel_num)
        count = 0
        acts = {}
        for layer in self.layers:
            if(self.layer_types[count] in ["rconv"]):
                x, act = layer(x)
                acts[self.get_layer_name(count)] = act #for activation regularization
                x = torch.squeeze(x[:, -1:, :, :, :])
            else:
                x = layer(x)
            count += 1
        return x, acts
    def response(self, x, iter_time=None):
        ress = {}
        count = 0
        for layer in self.layers:
            layer_name = "layer"+str(count)
            if(self.dict[layer_name]["type"] in ["rconv"]):
                x, res = layer.response(x, iter_time=iter_time)
                for key in res.keys():
                    res[key] = res[key].detach().cpu()
                ress[layer_name] = res
            else:
                x = layer(x)
            count += 1
        return ress

    def iter(self, x, iter_time=None):
        ress = {}
        count = 0
        for layer in self.layers:
            layer_name = "layer"+str(count)
            if(self.dict[layer_name]["type"] in ["rconv"]):
                x, res = layer.iter(x, iter_time=iter_time)
                ress[layer_name] = res
            else:
                x = layer(x)
            count += 1
        return ress
    def report_loss(self):
        for key in self.loss_list.keys():
            #print("%s:%s"%(key, str(self.loss_list[key]/self.loss_count)), end=" ")
            print("%s:%.4e"%(key, self.loss_list[key]/self.loss_count), end=" ")
        print("\n")
    def reset_loss(self):
        #print("aaa")
        self.loss_count = 0
        #print(self.loss_count)
        for key in self.loss_list.keys():
            self.loss_list[key] = 0.0
    def get_loss(self, x, y):
        #x:(batch_size, sequence_length, input_num)
        #y:(batch_size, sequence_length, output_num)
        output, acts = self.forward(x)
        #self.dict["act_avg"] = torch.mean(torch.abs(act))
        #print(output.size())
        #input()
        loss_class = class_coeff * self.class_loss_func(output, y)
        if act_cons_coeff==0.0:
            loss_act = 0.0
        else:
            for key in acts.keys():
                loss_act += act_cons_coeff * torch.mean(acts[key] ** 2)
            self.loss_list["act"] = self.loss_list["act"] + loss_act.item()
        
        self.loss_list["class"] = self.loss_list["class"] + loss_class.item()
         
        loss_weight = 0.0
        if(hebb_coeff==0.0):
            loss_hebb = 0.0 
        else:
            loss_hebb = self.get_loss_hebb(act)
        self.loss_count += 1
        return loss_class + loss_act + loss_weight + loss_hebb

    def get_loss_hebb(self, act):
        x = torch.squeeze(act[-1, :, :]) #(batch_size, neuron_num)
        batch_size=x.size(1)
        x = x.detach().cpu().numpy()
        x = torch.from_numpy(x).to(device)
        
        weight=self.N.get_r() #(neuron_num, neuron_num)
        act_var = torch.var(x, dim=0) #(neuron_num)
        act_var = act_var * (batch_size - 1)
        act_mean = torch.mean(x, dim=0).detach() #(neuron_num)
        #convert from tensor and numpy prevents including the process of computing var and mean into the computation graph.
        #act_std = torch.from_numpy(act_var).to(device) ** 0.5
        #act_mean = torch.from_numpy(act_mean).to(device)
        act_std = act_var ** 0.5
        std_divider = torch.mm(torch.unsqueeze(act_std, 1), torch.unsqueeze(act_std, 0)) #(neuron_num, neuron_num)
        
        x_normed = (x - act_mean) #broadcast
        act_dot = torch.mm(x_normed.t(), x_normed)
        try:
            act_corr = act_dot / std_divider
        except Exception:
            abnormal_coords = []
            for i in range(list(act_std.size())[0]):
                for j in range(list(act_std.size()[1])):
                    if(std_divider[i][j] == 0.0):
                        print("act_std[%d]][%d]==0.0"%(i, j))
                        std_divider[i][j] = 1.0
                        abnormal_coords.append([i,j])
            act_corr = act_dot / std_divider
            for coord in abnormal_coords:
                act_corr[coord[0]][coord[1]] = 0.0
        
        act_corr = act_corr.detach().cpu().numpy()
        act_corr = torch.from_numpy(act_corr).to(device)

        self.dict["last_act_corr"] = act_corr.detach().cpu()
        return -hebb_index * torch.mean(torch.tanh(torch.abs(weight)) * act_corr)
    def save(self, net_path):
        for key in self.dict["cache"].keys():
            if(self.dict.get(key) is not None):
                self.dict.pop(key)      
        f = open(net_path+"state_dict.pth", "wb")
        net = self.to(torch.device("cpu"))
        torch.save(net.dict, f)
        net = self.to(device)
        f.close()
    def update_weight_cache(self):
        for layer_index in self.rconvs:
            self.layers[layer_index].update_weight_cache()
    def get_iter_data(self, data_loader, iter_time=None, batch_num=None):
        print("calculating iter_data. batch_num=%d"%(len(data_loader)))
        if(iter_time is None):
            iter_time = self.dict["iter_time"]
        self.update_weight_cache()
        self.eval()
        count=0
        data_loader = list(data_loader)
        if(batch_num is None):
            batch_num = len(data_loader)
        else:
            data_loader = random.sample(data_loader, 50)
        ress = {}
        iter_data = {}

        label_count = 0
        for data in data_loader:
            inputs, labels = data
            inputs=inputs.to(device)
            #labels=labels.to(device)
            count += 1
            label_count += labels.size(0)
            res=self.iter(inputs, iter_time=iter_time) #[key](batch_size, iter_time, unit_num)
            
            for layer in res.keys():
                if(ress.get(layer) is None):
                    ress[layer] = {} 
                for key in res[layer].keys():
                    if(ress[layer].get(key) is None):
                        ress[layer][key] = []
                    ress[layer][key].append(res[layer][key])
                last_layer = layer
        for layer in res.keys():
            #print("layer_name: %s"%(layer))
            iter_data[layer] = {}
            cat_dict(iter_data[layer], ress[layer], dim=0) #cat along batch_size dim.         
        self.train()
        return iter_data
    def get_res_data(self, data_loader, iter_time=None, batch_num=None):
        print("calculating res_data. batch_num=%d"%(len(data_loader)))
        self.update_weight_cache()
        self.eval()
        count=0
        data_loader = list(data_loader)
        if(batch_num is None):
            batch_num = len(data_loader)
        else:
            data_loader = random.sample(data_loader, 50)
        ress = {}
        res_data = {}
        for data in data_loader:
            #print("count=%d"%(count))
            count=count+1
            inputs, labels = data
            inputs=inputs.to(device)
            #labels=labels.to(device)
            res=self.response(inputs, iter_time=iter_time) #[key](batch_size, unit_num)
            for layer in res.keys():
                if(ress.get(layer) is None):
                    ress[layer] = {}
                for key in res[layer].keys():
                    if(ress[layer].get(key) is None):
                        ress[layer][key] = []
                    ress[layer][key].append(res[layer][key])
                    #print("length: %d"%(len(ress[layer][key])))
                last_layer = layer
        for layer in res.keys():
            #print("layer_name: %s"%(layer))
            res_data[layer] = {}
            cat_dict(res_data[layer], ress[layer], dim=0) #cat along batch_size dim.         
        self.train()
        return res_data
    def print_weight_info(self):
        ei = self.get_weight("E->I", positive=True)
        ie = self.get_weight("I->E", positive=True)
        er = self.get_weight("E->E", positive=True)
        ir = self.get_weight("I->I", positive=True)
        weights = [ei, ie, er, ir]
        for w in weights:
            print(w)
            print(torch.mean(w))
            print(torch.min(w))
            print(torch.max(w))
            print(list(w.size()))
