import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import heapq

#training parameters.
from param_config import *
from anal_functions import *
from utils_model import *
import random

import sys

from neurons_LIF import neurons_LIF

class RNN_EI(nn.Module):
    def __init__(self, dict_=None, load=False, f=None):
        super(RNN_EI, self).__init__()
        if load:
            self.dict=torch.load(f, map_location=device) 
            self.N = neurons_LIF(load=True, dict_=self.dict["N"]) #neurons
            self.i = self.dict["i"] #input weight
            self.b_0 = self.dict["b_0"]
        else:
            self.dict = dict_
            self.i = torch.nn.Parameter(torch.zeros((self.dict["input_num"], self.dict["N_num"]), device=device))
            init_weight(self.i, self.dict["init"]["i"])
            
            self.dict["i"] = self.i
            
            self.N = neurons_LIF(dict_ = self.dict["N"])
            self.dict["N"] = self.N.dict

            if(self.dict["bias"]==True):
                self.b_0 = torch.nn.Parameter(torch.zeros((self.dict["N_num"]), device=device))            
            else:
                self.b_0 = 0.0
            self.dict["b_0"] = self.b_0

        self.get_i = lambda :self.i
        self.get_f = self.N.get_f
        self.get_r = self.N.get_r

        input_mode = get_name(self.dict["input_mode"])
        if input_mode=="endure" or input_mode is None: #default
            self.prepare_input = self.prepare_input_endure
            self.get_input = self.get_input_endure
        
        if(self.dict["class_loss_func"]=="CEL"):
            self.class_loss_func = torch.nn.CrossEntropyLoss()
        elif(self.dict["class_loss_func"]=="MSE"):
            self.class_loss_func = torch.nn.MSELoss()
        self.loss_list = {"class":0.0, "act":0.0, "weight":0.0}
        if(hebb_index != 0.0):
            self.loss_list["hebb"] = 0.0
        self.loss_count = 0
        self.iter_time = self.dict["iter_time"]

        if self.dict["separate_ei"]:
            self.get_weight = self.get_weight_ei
            self.update_weight_cache = self.update_weight_cache_ei
            self.response_keys = ["E.u","E.x","I.u","I.x","E->E","E->I","I->E","I->I","E->Y","I->Y", "X->E", "X->I", "N->Y", "N->N", "u"]
            self.weight_names = ["X->E", "X->I", "i"]
        else:
            self.response_keys = ["f","r","u"]
            self.get_weight = self.get_weight_uni
        self.cache = {}
        self.E_num = self.N.dict["E_num"]
        self.I_num = self.N.dict["I_num"]
        self.N_num = self.N.dict["N_num"]
        self.dict["noself"] = self.N.dict["noself"]
    def prepare_input_once(self, i_):
        return torch.mm(i_, self.get_i()) + self.b_0
    def prepare_input_endure(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, neuron_num)
        self.cache["input"] = i_
    def prepare_input_full(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, neuron_num)
        i_unsqueezed = torch.unsqueeze(i_, 1)
        return torch.cat([i_unsqueezed for _ in range(self.iter_time)], dim=1) #(batch_size, iter_time, neuron_num)        
    def get_input_endure(self, time=None):
        return self.cache["input"]
    def forward(self, x, iter_time=None): #(batch_size, pixel_num)
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prepare_input(x) #(iter_time, batch_size, neuron_num)
        act_list = []
        output_list = []
        r = 0.0
        for time in range(iter_time):
            f, r, u = self.N.forward(torch.squeeze(self.get_input(time)) + r)
            act_list.append(u) #(batch_size, neuron_num)
            output_list.append(f) #(batch_size, output_num)
        output_list = list(map(lambda x:torch.unsqueeze(x, 1), output_list))
        act_list = list(map(lambda x:torch.unsqueeze(x, 1), act_list))
        output = torch.cat(output_list, dim=1) #(batch_size, iter_time, output_num)
        act = torch.cat(act_list, dim=1) #(batch_size, iter_time, neuron_num)
        return output, act
    def forward_init(self, batch_size):
        self.N.reset_x(batch_size=batch_size)
    def forward_once(self, i_, output_type="r"):
        r_ = i_[:, 0:self.N_num]
        i_raw = i_[:, self.N_num:]
        i_processed = self.prepare_input_once(i_raw)
        f, r, u = self.N.forward(i_processed + r_)
        if output_type=="r":
            o_ = r
        elif output_type=="f":
            o_ = f
        else:
            o_ = None
        i_tot = torch.cat([r_, i_processed], dim=1)
        return i_raw, i_processed, i_tot, o_
    def response(self, x, iter_time=None):
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prepare_input_full(x) #(batch_size, iter_time, neuron_num)
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
        if self.dict["separate_ei"]:
            res_X = torch.squeeze(i_[:, -1, :])
            res["X->E"] = res_X[:, 0:self.E_num]
            res["X->I"] = res_X[:, self.E_num:self.N_num]
        else:
            res["X->N"] = torch.squeeze(i_[:, -1, :])#(batch_size, neuron_num)
        return res
    def iter(self, x, iter_time=None, to_cpu_interval=10):
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prepare_input_full(x) #(batch_size, iter_time, neuron_num)
        ress = {} #responses
        ress_cat = {}
        keys = self.response_keys
        for key in keys:
            ress[key] = []
            ress_cat[key] = None
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
            for key in res.keys():
                ress[key].append(res[key]) #[key](batch_size, unit_num)
            if((time+1)%to_cpu_interval == 0): #avoid GPU OOM.
                cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1)
        cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1) #cat along iter_time dim.
        if self.dict["separate_ei"]:
            ress_cat["X->E"] = i_[:, :, 0:self.E_num]
            ress_cat["X->I"] = i_[:, :, self.E_num:self.N_num]
        else:
            ress_cat["X->N"] = i_
        return ress_cat #(batch_size, iter_time, neuron_num)
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
        output, act = self.forward(x)
        #self.dict["act_avg"] = torch.mean(torch.abs(act))
        #print(output.size())
        #input()
        loss_class = class_index * self.class_loss_func( torch.squeeze(output[:,-1,:]), y)
        loss_act = act_cons_index * torch.mean(act ** 2)
        loss_weight = weight_cons_index * ( torch.mean(self.N.get_r() ** 2) )
        self.loss_list["weight"] = self.loss_list["weight"] + loss_weight.item()
        self.loss_list["act"] = self.loss_list["act"] + loss_act.item()
        self.loss_list["class"] = self.loss_list["class"] + loss_class.item()
        self.loss_count += 1
        #print(self.loss_count)
        if(hebb_index==0.0):
            loss_hebb = 0.0 
        else:
            loss_hebb = self.get_loss_hebb(act)
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
        
        #print(weight.device)
        #print(act_corr.device)
        #if(self.training == True):
            #print(act_corr)
            #print("act_corr")
            #input()
        self.dict["last_act_corr"] = act_corr.detach().cpu()
        return -hebb_index * torch.mean(torch.tanh(torch.abs(weight)) * act_corr)
    def save(self, net_path):
        cached_keys = ["weight_cache"]
        for key in cached_keys:
            if(self.N.dict.get(key) is not None):
                self.N.dict.pop(key)
        for key in self.dict["cache"].keys():
            if(self.dict.get(key) is not None):
                self.dict.pop(key)      

        f = open(net_path+"state_dict.pth", "wb")
        net = self.to(torch.device("cpu"))
        torch.save(net.dict, f)
        net = self.to(device)
        f.close()
    def get_weight_ei(self, name, detach=False, positive=True):
        if(name in self.N.weight_names):
            return self.N.get_weight(name=name, detach=detach, positive=positive)
        elif(name in self.weight_names):
            if(name=="b_0"):
                w = self.b_0
            elif(name=="i"):
                w = self.get_i()
            elif(name=="X->E"):
                w = self.get_i()[:, 0:self.E_num]
            elif(name=="X->I"):
                w = self.get_i()[:, self.E_num:self.N_num]
        else:
            print("invalid weight name:%s"%(name))
        if detach:
            w = w.detach()
        return w
    def update_weight_cache_ei(self):
        self.N.update_weight_cache_ei()
        self.cache["weight_cache"] = self.N.cache["weight_cache"]
        #print("E->E: ")
        #print(self.cache["weight_cache"]['E->E'])
        #print(self.N.get_weight('E->E'))
        #input()
    def get_iter_data(self, data_loader, iter_time=None, batch_num=None):
        print("calculating iter_data. batch_num=%d"%(len(data_loader)))
        if(iter_time is None):
            iter_time = self.iter_time
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
        keys = self.response_keys
        for key in keys:
            iter_data[key] = None
            ress[key] = []

        iter_data["acc"] = [0.0 for _ in range(iter_time)]
        iter_data["loss"] = [0.0 for _ in range(iter_time)]
        #print("aaa")
        #print(len(iter_data["acc"]))
        #print(len(iter_data["loss"])) 
        label_count = 0
        for data in data_loader:
            inputs, labels = data
            inputs=inputs.to(device)
            #labels=labels.to(device)
            count += 1
            label_count += labels.size(0)
            res=self.iter(inputs) #[key](batch_size, iter_time, unit_num)
            for key in ress.keys():
                ress[key].append(res[key])
            
            for time in range(iter_time):
                iter_data["loss"][time] += self.class_loss_func( torch.squeeze(res["N->Y"][:,time,:]), labels)
                iter_data["acc"][time] += ( torch.max( torch.squeeze(res["N->Y"][:,time,:] ), 1)[1]==labels).sum().item()

        for time in range(iter_time):
            iter_data["loss"][time] = iter_data["loss"][time] / count
            iter_data["acc"][time] = iter_data["acc"][time] / label_count
        
        cat_dict(iter_data, ress, dim=0) #cat along batch_size dim.         
        self.train()

        #print(len(iter_data["loss"]))
        #input()

        return iter_data
    def get_res_data(self, data_loader, iter_time=None, batch_num=None):
        print("calculating res_data. batch_num=%d"%(len(data_loader)))
        if(iter_time is None):
            iter_time = self.iter_time
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
        keys = self.response_keys
        for key in keys:
            res_data[key] = None
            ress[key] = []
        for data in data_loader:
            count=count+1
            inputs, labels = data
            inputs=inputs.to(device)
            #labels=labels.to(device)
            res=self.response(inputs) #[key](batch_size, unit_num)
            for key in res.keys():
                ress[key].append(res[key])
        cat_dict(res_data, ress, dim=0) #cat along batch_size dim.         
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

    def cache_weight(self):
        self.cache["i"] = self.get_i().detach().cpu().numpy()
        self.cache["r"] = self.get_r().detach().cpu().numpy()
        self.cache["f"] = self.get_f().detach().cpu().numpy()
    def report_weight_update(self):
        i = self.get_i().detach().cpu().numpy()
        r = self.get_r().detach().cpu().numpy()
        f = self.get_f().detach().cpu().numpy()

        i_delta = np.sum(np.abs(i - self.cache["i"])) / np.sum(np.abs(self.cache["i"]))
        r_delta = np.sum(np.abs(r - self.cache["r"])) / np.sum(np.abs(self.cache["r"]))
        f_delta = np.sum(np.abs(f - self.cache["f"])) / np.sum(np.abs(self.cache["f"]))
        print("weight update rate: i:%.4e r:%.4e f:%.4e"%(i_delta, r_delta, f_delta))