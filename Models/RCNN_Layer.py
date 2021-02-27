import torch
import torch.nn as nn
import torch.nn.functional as F

from anal_functions import *
from utils_model import *
import random

def init_kernel(weight, params):
    name = get_name(params)
    coeff = get_arg(params)
    if(name=="output"):
        divider = weight.size(0) * weight.size(2) * weight.size(3) #output_num * kernel_length * kernel_width
    elif(name=="input"):
        divider = weight.size(1) * weight.size(2) * weight.size(3) #input_num * kernel_length * kernel_width

    lim = coeff / divider
    #print("coeff=%.4e"%(coeff))
    #print("lim=%4e"%(lim))
    if(constraint_method=="force"):
        torch.nn.init.uniform_(weight, 0.0, 2 * lim)
    else:
        torch.nn.init.uniform_(weight, -lim, lim)
    
def get_kernel_ei_mask(E_num, N_num):
    ei_mask = torch.ones((1, N_num, 1, 1), device=device, requires_grad=False)
    for i in range(E_num, N_num):
        ei_mask[0][i][0][0] = -1.0
    return ei_mask

class RCNNBlock(nn.Module):
    def __init__(self, dict_=None, load=False):
        super(RCNNBlock, self).__init__()
        if load:
            self.dict=dict_
            self.f = self.dict["f"]
            self.r = self.dict["r"]
            self.b = self.dict["b"]
        else:
            self.dict = dict_
            if self.dict["bias"]:
                self.b = torch.nn.Parameter(torch.zeros((self.dict["N_num"], 1, 1), device=device))
            else:
                self.b = 0.0
            self.dict["b"] = self.b

            self.f = torch.nn.Parameter(torch.zeros((self.dict["output_num"], self.dict["N_num"], self.dict["kernel_size"], self.dict["kernel_size"]), device=device, requires_grad=True))
            self.r = torch.nn.Parameter(torch.zeros((self.dict["output_num"], self.dict["N_num"], self.dict["r_kernel_size"], self.dict["r_kernel_size"]), device=device, requires_grad=True))
            self.dict["f"] = self.f
            self.dict["r"] = self.r
            init_kernel(self.r, self.dict["init"]["r"])
            init_kernel(self.f, self.dict["init"]["f"])

        self.kernel_size = self.dict["kernel_size"]
        self.f_stride = self.dict["stride"]
        self.f_padding = self.dict["padding"]
        self.r_kernel_size = self.dict["r_kernel_size"]
        self.r_stride = self.dict["r_stride"]
        self.r_padding = self.dict["r_padding"]
        self.output_num = self.dict["output_num"]
        self.iter_time = self.dict["iter_time"]

        self.N_num = self.dict["N_num"]
        if self.dict["separate_ei"]:
            self.time_const_e = self.dict["time_const_e"]
            self.time_const_i = self.dict["time_const_i"]
            self.act_func = self.act_func_ei
            self.act_func_e = get_act_func(self.dict["act_func_e"])
            self.act_func_i = get_act_func(self.dict["act_func_i"])
            self.E_num = self.dict["E_num"]
            self.I_num = self.dict["I_num"]
            self.cal_x = self.cal_x_ei
            self.get_weight = self.get_weight_ei
            self.response_once = self.response_ei_once
            self.update_weight_cache = self.update_weight_cache_ei
            self.weight_names = ["E->E","E->I","I->E","I->I","E->Y","I->Y","N->N","E.r","E.l","I.r","I.l","E.b","I.b","r","E.f","I.f","b"]
        else:
            self.time_const = self.dict["time_const"]
            self.act_func = get_act_func(self.dict["act_func"])
            self.cal_x = self.cal_x_uni
            self.get_weight = self.get_weight_uni
            self.response_once = self.response_uni_once
            self.update_weight_cache = self.update_weight_cache_uni
            self.weight_names = ["N->Y","N->N","N.f","f","b","r"]
        self.constraint_func = get_constraint_func(constraint_method)
        #set recurrent weight
        r_shape = (self.dict["N_num"], self.dict["N_num"], self.r_kernel_size, self.r_kernel_size)
        if self.dict["noself"]:
            self.r_self_mask = torch.ones(r_shape, device=device, requires_grad=False)
            for i in range(self.dict["N_num"]):
                for j in range(self.r_kernel_size):
                    for k in range(self.r_kernel_size):
                        self.r_self_mask[i][i][j][k] = 0.0
            self.get_r_noself = lambda :self.r * self.r_self_mask
        else:
            self.get_r_noself = lambda :self.r
        if(self.dict["separate_ei"] and "r" in self.dict["Dale"]):
            self.ei_mask = get_kernel_ei_mask(E_num=self.dict["E_num"], N_num=self.dict["N_num"])
            self.get_r_ei = lambda :self.ei_mask * self.constraint_func(self.get_r_noself())
        else:
            self.get_r_ei = self.get_r_noself
        if("r" in self.dict["mask"]):
            self.r_mask = get_mask_from_tuple(r_shape)
            self.get_r_mask = lambda :self.r_mask * self.get_r_ei()
        else:
            self.get_r_mask = self.get_r_ei
        self.get_r = self.get_r_mask

        #set forward weight
        f_shape = (self.dict["N_num"], self.dict["output_num"], self.kernel_size, self.kernel_size)
        if("f" in self.dict["Dale"]): #set mask for EI separation
            if(self.ei_mask is None):
                self.ei_mask = get_kernel_ei_mask(E_num=self.dict["E_num"], N_num=self.dict["N_num"])
            self.get_f_ei = lambda :self.ei_mask * self.constraint_func(self.f)
        else:
            self.get_f_ei = lambda :self.f
        if("f" in self.dict["mask"]): #set mask for connection pruning
            self.f_mask = get_mask_from_tuple(f_shape)
            self.get_f_mask = lambda :self.f_mask * self.get_f_ei()
        else:
            self.get_f_mask = self.get_f_ei            
        self.get_f = self.get_f_mask

        input_mode = self.dict.get("input_mode")
        if input_mode is None or input_mode=="endure": #default
            self.prepare_input = self.prepare_input_endure
            self.get_input = self.get_input_endure

        if(index["noise"]==0.0):
            self.get_noise = lambda shape:0.0
        else:
            self.get_noise = self.get_noise_gaussian

        self.dict["cache"] = {}
        self.cache = self.dict["cache"]
        self.dict["weight_cache"] = {}
        self.reset_x = self.reset_x_zero
        self.x = None
    def prepare_input_endure(self, i_):
        self.cache["input"] = i_
    def get_input_endure(self, time):
        return self.cache["input"]
    def act_func_ei(self, x):
        return torch.cat( [self.act_func_e(x[:, 0:self.E_num]), self.act_func_i(x[:, self.E_num:self.N_num])], dim=1)
    def cal_x_uni(self, dx): #(batch_size, N_num, width, height)
        return (1.0 - self.time_const) * (self.x + self.get_noise( (dx.size(0), dx.size(1), dx.size(2), dx.size(3)) ) ) + self.time_const * dx #x:(batch_size, neuron_num)
    def cal_x_ei(self, dx):
        x_e = (1.0 - self.time_const_e) * (self.x[:, 0:self.E_num] + self.get_noise( (dx.size(0), self.E_num, dx.size(2), dx.size(3)) ) ) + self.time_const_e * dx[:, 0:self.E_num] #x:(batch_size, E_num)
        x_i = (1.0 - self.time_const_i) * (self.x[:, self.E_num:self.N_num] + self.get_noise( (dx.size(0), self.I_num, dx.size(2), dx.size(3)) ) ) + self.time_const_i * dx[:, self.E_num:self.N_num] #x:(batch_size, I_num)        
        return torch.cat([x_e, x_i], dim=1)
    def get_noise_gaussian(self, shape):
        noise = torch.zeros(shape, device=device)
        torch.nn.init.normal_(noise, 0.0, index["noise"])
        return noise
    def reset_x_zero(self, shape):
        self.x = torch.zeros(shape, device=device)
    def forward_once(self, i_): #(batch_size, N_num, width, height)
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x:(batch_size, N_num, width, height)
        u = self.act_func(self.x)
        r = F.conv2d(input=i_, weight=self.get_r(), bias=None, stride=self.r_stride, padding=self.r_padding)
        f = F.conv2d(input=i_, weight=self.get_f(), bias=None, stride=self.f_stride, padding=self.f_padding)
        return f, r, u
    def forward(self, i_): #(batch_size, N_num, width, height)
        self.reset_x(shape=(i_.size(0), i_.size(1), i_.size(2), i_.size(3)))
        act_list = []
        output_list = []
        self.prepare_input(i_)
        r = 0.0
        for time in range(self.iter_time):
            f, r, u = self.forward_once(self.get_input(time) + r)
            output_list.append(torch.unsqueeze(f, 1))
            act_list.append(torch.unsqueeze(u, 1))
        act = torch.cat(act_list, dim=1) #cat along iter_time
        output = torch.cat(output_list, dim=1)
        return output, act
    def response_uni_once(self, i_): #(batch_size, N_num, width, height)
        res = {}
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x:(batch_size, N_num, width, height)
        u = self.act_func(self.x)
        r = F.conv2d(u, self.get_r(), None, self.r_stride, self.r_padding)
        f = F.conv2d(u, self.get_f(), None, self.f_stride, self.f_padding)
        res["x"] = self.x
        res["u"] = u
        res["f"] = f
        res["r"] = r
        return f, r, u, res
    def response_ei_once(self, i_): #(batch_size, N_num, width, height)
        res = {}
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x:(batch_size, neuron_num)
        u = self.act_func(self.x)
        r = F.conv2d(u, self.get_r(), None, self.r_stride, self.r_padding)
        f = F.conv2d(u, self.get_f(), None, self.f_stride, self.f_padding)
        res["x"] = self.x
        res["u"] = u
        res["f"] = f
        res["r"] = r

        res["E.x"] = self.x[:, 0:self.E_num, :, :]
        res["I.x"] = self.x[:, self.E_num:self.N_num, :, :]
        res["E.u"] = u[:, 0:self.E_num, :, :]
        res["I.u"] = u[:, self.E_num:self.N_num, :, :]        

        #print(self.cache["weight_cache"]["E->E"].size())
        res["E->E"] = F.conv2d(res["E.u"], self.cache["weight_cache"]["E->E"], None, 1, 0) #input, weight, bias, stride, padding
        res["E->I"] = F.conv2d(res["E.u"], self.cache["weight_cache"]["E->I"], None, 1, 0)
        res["I->E"] = F.conv2d(res["I.u"], self.cache["weight_cache"]["I->E"], None, 1, 0)
        res["I->I"] = F.conv2d(res["I.u"], self.cache["weight_cache"]["I->I"], None, 1, 0)

        res["E->Y"] = F.conv2d(res["E.u"], self.cache["weight_cache"]["E->Y"], None, self.f_stride, self.f_padding) #input, weight, bias, stride, padding
        res["I->Y"] = F.conv2d(res["I.u"], self.cache["weight_cache"]["I->Y"], None, self.f_stride, self.f_padding)

        return f, r, u, res
    def response(self, i_, iter_time=None):
        if iter_time is None:
            iter_time = self.iter_time
        res = {}
        self.reset_x(shape=(i_.size(0), i_.size(1), i_.size(2), i_.size(3)))
        self.prepare_input(i_)
        act_list = []
        output_list = []
        r = 0.0
        for time in range(iter_time):
            f, r, u, res = self.response_once(self.get_input(time) + r)
        return f, res
    def iter(self, i_, iter_time=None, to_cpu_interval=10):
        if(iter_time is None):
            iter_time = self.iter_time
        self.reset_x(shape=(i_.size(0), i_.size(1), i_.size(2), i_.size(3)))
        self.prepare_input(i_)
        ress = {} #responses
        ress_cat = {}
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(self.get_input(time)) + r
            f, r, u, res = self.response_once(i_tot)
            for key in res.keys():
                if(ress.get(key) is None):
                    ress[key] = [] 
                ress[key].append(res[key]) #[key](batch_size, unit_num)
            if((time+1)%to_cpu_interval == 0): #avoid GPU OOM.
                cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1)
        cat_dict(ress_cat, ress, dim_unsqueeze=1, dim=1) #cat along iter_time dim.
        x2n = torch.cat([torch.unsqueeze(self.get_input(time), 1) for time in range(iter_time)], dim=1).detach().cpu()
        if self.dict["separate_ei"]:
            ress_cat["X->E"] = x2n[:, :, 0:self.E_num, :, :]
            ress_cat["X->I"] = x2n[:, :, self.E_num:self.N_num, :, :]
        else:
            ress_cat["X->N"] = x2n
        #print("x2n:", end='')
        #print(x2n.size())
        return f, ress_cat #(batch_size, iter_time, neuron_num)
    def response_ei(self, i_):
        res = {}
        dx = i_ + self.b
        self.x = self.cal_x(dx) #x:(batch_size, neuron_num, width, height)
        u = self.act_func(self.x)
        res["u"] = u

        res["E.x"] = self.x[:, 0:self.E_num, :, :]
        res["I.x"] = self.x[:, self.E_num:self.N_num, :, :]
        res["E.u"] = u[:, 0:self.E_num, :, :]
        res["I.u"] = u[:, self.E_num:self.N_num, :, :]        

        res["E->E"] = torch.mm(res["E.u"], self.dict["weight_cache"]["E->E"])
        res["E->I"] = torch.mm(res["E.u"], self.dict["weight_cache"]["E->I"])
        res["I->E"] = torch.mm(res["I.u"], self.dict["weight_cache"]["I->E"])
        res["I->I"] = torch.mm(res["I.u"], self.dict["weight_cache"]["I->I"])
        
        f = F.conv2d(self.x, self.get_f(), stride=self.f_stride, padding=self.f_padding)
        r = u.mm(self.x, self.get_r(), stride=self.r_stride, padding=self.r_padding)
        res["N->Y"] = f
        res["N->N"] = r

        res["E->Y"] = F.conv2d(res["E.u"], self.dict["weight_cache"]["E->Y"], stride=self.f_stride, padding=self.f_padding)
        res["I->Y"] = F.conv2d(res["I.u"], self.dict["weight_cache"]["I->Y"], stride=self.f_stride, padding=self.f_padding)
        return f, r, u, res
    def get_weight_uni(self, name, positive=None):
        if(name in ["r", "N->N"]):
            return self.get_r()
        elif(name=="b"):
            return self.get_b
    def update_weight_cache_ei(self):
        self.cache["weight_cache"] = {}
        weight_cache = self.cache["weight_cache"]
        N_r = self.get_weight("r", positive=True)
        weight_cache["E->E"] = N_r[0:self.E_num, 0:self.E_num, 0:1, 0:1] #(output, input, kernel_width, kernel_height)
        weight_cache["I->I"] = N_r[self.E_num:self.N_num, self.E_num:self.N_num, 0:1, 0:1]
        weight_cache["E->I"] = N_r[self.E_num:self.N_num, 0:self.E_num, 0:1, 0:1]
        weight_cache["I->E"] = N_r[0:self.E_num, self.E_num:self.N_num, 0:1, 0:1]
        N_f = self.get_weight("f")
        weight_cache["E->Y"] = N_f[:, 0:self.E_num, :, :]
        weight_cache["I->Y"] = N_f[:, self.E_num:self.N_num, :, :]
    def update_weight_cache_uni(self):
        a = 1.0
    def get_weight_ei(self, name, positive=True, detach=False):
        sig_r = False
        sig_f = False
        if(name in ["E.r", "E->E"]):
            w = self.get_r()[0:self.E_num, 0:self.E_num, :, :]
        elif(name in ["I.r", "I->I"]):
            w = self.get_r()[self.E_num:self.N_num, self.E_num:self.N_num, :, :]
            sig_r = True
        elif(name in ["E.l", "E->I"]):
            w = self.get_r()[self.E_num:self.N_num, 0:self.E_num, :, :]
        elif(name in ["I.l", "I->E"]):
            w = self.get_r()[0:self.E_num, self.E_num:self.N_num, :, :]
            sig_r = True
        elif(name in ["E.f", "E->Y"]):
            w = self.get_f()[:, 0:self.E_num, :, :]
        elif(name in ["I.f", "I->Y"]):
            w = self.get_f()[:, self.E_num:self.N_num, :, :]
            sig_f = True
        elif(name in ["b"]):
            w = self.b
        elif(name=="E.b"):
            if(isinstance(self.b, float)):
                return self.b
            else:
                w = self.b[0:self.E_num]
        elif(name=="I.b"):
            if(isinstance(self.b, float)):
                return self.b
            else:
                w = self.b[self.E_num:self.N_num]
        elif(name in ["r", "N->N"]):
            w = self.get_r()
            if positive:
                w = torch.abs(w)
        elif(name in ["f"]):
            w = self.get_f()
        else:
            return "invalid weight name:%s"%(name)
        if detach:
            w = w.detach()
        if(positive and sig_r and ("r" in self.dict["Dale"])):
            w = - w
        elif(positive and sig_f and ("f" in self.dict["Dale"])):
            w = - w
        return w
    def plot():
 
        Painter.plot(ax)
        plt.savefig()