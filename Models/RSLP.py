import pickle

import random
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models import Neurons_LIF
from utils import *
from utils import ensure_path, get_from_dict, search_dict
from utils_model import *
from utils_model import cat_dict

class RSLP(nn.Module):
    def __init__(self, dict_=None, load=False):
        super(RSLP, self).__init__()
        #if options is not None:
        #    self.receive_options(options)
        #else: print('RSLP: warning: options is None.')
        
        print('RSLP: initializing... load=%s'%str(load))
        
        self.dict = dict_

        self.separate_ei = self.dict['separate_ei']

        self.device = 'cpu' if self.dict.get('device') is None else self.dict['device']

        self.N = Neurons_LIF(dict_=self.dict['N'], load=load)
        
        if load:
            self.i = nn.Parameter(self.dict['i']) #input weight
            self.b_0 = self.dict['b_0']
        else:
            self.dict = dict_
            self.i = torch.nn.Parameter(torch.zeros((self.dict['input_num'], self.dict['N_num']), device=self.device))
            init_weight(self.i, self.dict['init_weight']['i'], cons_method=self.dict['cons_method'])
            
            self.dict['i'] = self.i
            self.dict['N'] = self.N.dict

            if self.dict['bias']:
                self.b_0 = torch.nn.Parameter(torch.zeros((self.dict['N_num']), device=self.device))            
            else:
                self.b_0 = 0.0
            self.dict['b_0'] = self.b_0

        self.name = get_from_dict(self.dict, 'name', default='unnamed_RNN_EI', write_default=True)
        
        self.main_loss_coeff = get_from_dict(self.dict['loss'], 'main_loss_coeff', default=0.0, write_default=True)
        self.act_coeff = get_from_dict(self.dict['loss'], 'act_coeff', default=0.0, write_default=True)
        self.weight_coeff = get_from_dict(self.dict['loss'], 'weight_coeff', default=0.0, write_default=True)
        self.hebb_coeff = get_from_dict(self.dict['loss'], 'hebb_coeff', default=0.0, write_default=True)
        
        self.input_mode = get_from_dict(self.dict, 'input_mode', default='endure', write_default=True)
        if self.input_mode in ['endure']: # default input mode
            self.prepare_input = self.prepare_input_endure
            self.get_input = self.get_input_endure
        
        self.loss_dict = search_dict(self.dict, ['loss', 'loss_coeff'])
        self.main_loss_func_str = search_dict(self.loss_dict, ['main_loss_func'], default='CEL', write_default=True)
        if self.main_loss_func_str in ['CEL', 'cel']:
            self.main_loss_func = nn.CrossEntropyLoss()
        elif self.main_loss_func_str in ['MSE', 'mse']:
            self.main_loss_func = nn.MSELoss()
        self.iter_time = self.dict['iter_time']

        if self.separate_ei:
            self.get_weight = self.get_weight_ei
            self.update_weight_cache = self.update_weight_cache_ei
            self.response_keys = ['E.u','E.x','I.u','I.x','E->E','E->I','I->E','I->I','E->Y','I->Y', 'X->E', 'X->I', 'N->Y', 'N->N', 'u']
            self.weight_names = ['X->E', 'X->I', 'i']
        else:
            self.response_keys = ['f','r','u']
            self.get_weight = self.get_weight_uni
        self.cache = {}

        self.get_i = lambda :self.i
        self.get_f = self.N.get_f
        self.get_r = self.N.get_r
        self.noself = self.N.dict['noself']
        self.E_num = self.N.dict['E_num']
        self.I_num = self.N.dict['I_num']
        self.N_num = self.N.dict['N_num']
        self.init_perform()
    def receive_args(self, args):
        if isinstance(args, dict):
            for key, item in args.items():
                if key in ['device']:
                    self.device = self.dict_['device'] = item
                    self.N.device = self.N.dict_['device'] = item
                    self.to(self.device)
        else:
            raise Exception('Invalid arg type: '%type(args))    
    def forward_free(self, x, step_num=None):
        if step_num is None:
            step_num = self.step_num
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))

        r = 0.0
        for time in range(step_num):
            f, r, u = self.N.forward(torch.squeeze(self.get_input(time)) + h)

            # to be implemented
        return
    def forward_clamp(self, y, iter_time=None):
        if iter_time is None:
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

    def receive_options(self, options):
        self.options = options
        self.device = self.options.device
    def prepare_input_once(self, i_):
        #print(i_.size())
        #print(self.get_i().size())
        #print(self.b_0)
        return torch.mm(i_, self.get_i()) + self.b_0
    def prepare_input_endure(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, neuron_num)
        self.cache['input'] = i_
    def prepare_input_full(self, i_):
        i_ = torch.mm(i_, self.get_i()) + self.b_0 #(batch_size, neuron_num)
        i_unsqueezed = torch.unsqueeze(i_, 1)
        return torch.cat([i_unsqueezed for _ in range(self.iter_time)], dim=1) #(batch_size, iter_time, neuron_num)        
    def get_input_endure(self, time=None):
        return self.cache['input']
    def forward(self, x, iter_time=None):
        # x: input image. in shape of [batch_size, pixel_num]
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
    def forward_init(self, x_):
        self.N.reset_x(batch_size=x_.size(0))
    def forward_once(self, x_, r_in, detach_i=False, detach_u=False): #x_:[batch_size, input_num]. r:[batch_size, N_num]
        x_ = x_.view(x_.size(0), -1)
        i_ = self.prepare_input_once(x_) # [batch_size, N_num]
        if detach_i:
            i_1 = i_.detach()
        else:
            i_1 = i_
        if detach_u:
            f, r_, u, u_detach = self.N.forward_once(i_1 + r_in, detach_u=detach_u)
        else:
            f, r_, u = self.N.forward_once(i_1 + r_in, detach_u=detach_u)
        results = {'i_':i_, 'o_':f, 'r_':r_, 'u':u}
        if detach_i:
            results['i_detach'] = i_1
        if detach_u:
            results['u_detach'] = u_detach
        return results
    def response(self, x, iter_time=None):
        if(iter_time is None):
            iter_time = self.iter_time
        x = x.view(x.size(0), -1)
        self.N.reset_x(batch_size=x.size(0))
        i_ = self.prepare_input_full(x) # [batch_size, iter_time, neuron_num]
        r = 0.0
        for time in range(iter_time):
            i_tot = torch.squeeze(i_[:, time, :]) + r
            f, r, u, res = self.N.response(i_tot)
        if self.dict['separate_ei']:
            res_X = torch.squeeze(i_[:, -1, :])
            res['X->E'] = res_X[:, 0:self.E_num]
            res['X->I'] = res_X[:, self.E_num:self.N_num]
        else:
            res['X->N'] = torch.squeeze(i_[:, -1, :])#(batch_size, neuron_num)
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
        if self.separate_ei:
            ress_cat['X->E'] = i_[:, :, 0:self.E_num]
            ress_cat['X->I'] = i_[:, :, self.E_num:self.N_num]
        else:
            ress_cat['X->N'] = i_
        return ress_cat #(batch_size, iter_time, neuron_num)
    def init_perform(self):
        self.sample_count = 0
        self.perform_list = {'main':0.0, 'act':0.0, 'weight':0.0}
        if self.hebb_coeff != 0.0:
            self.perform_list['hebb'] = 0.0        
        self.perform_list['acc'] = 0.0
    def get_perform(self, end='\n', prefix=None, verbose=True):
        perform = {}
        if print:
            if prefix is not None:
                print(prefix, end=' ')
            for key in self.perform_list.keys():
                if key in ['acc']:
                    perform[key] = self.perform_list[key]/self.sample_count
                    print('%s:%s'%(key, perform[key]), end=' ')
                else:
                    perform[key] = self.perform_list[key]/self.batch_count
                    print('%s:%.3e'%(key, perform[key]), end=' ')
            print('%s'%end, end='')
        return perform
    def reset_perform(self):
        self.sample_count = 0
        self.batch_count = 0
        for key in self.perform_list.keys():
            self.perform_list[key] = 0.0
    def cal_perform(self, x, y):
        #x:(batch_size, sequence_length, input_num)
        #y:(batch_size, sequence_length). correct label.
        output, act = self.forward(x) #output:[batch_size, sequence_length, output_num]
        #self.dict['act_avg'] = torch.mean(torch.abs(act))
        #print(output.size())
        #input()
        return self.cal_perform_from_outputs(torch.squeeze(output[:,-1,:]), act, y)
    def cal_perform_from_outputs(self, output, act=None, y=None):
        if isinstance(output, list): # [time][batch_size, output_num]
            output = torch.cat(list(map(lambda x:torch.unsqueeze(x, 1), output)), dim=1) #[batch_size, time, output_num]
            #print(output.size())
            output = torch.squeeze(output[:,-1,:])
        
        #print(output.size())
        #print(y.size())

        loss_main = self.main_loss_coeff * self.main_loss_func(output, y)
        self.perform_list['main'] += loss_main.item()
        
        if act is None:
            loss_act = 0.0
        else:
            loss_act = self.act_coeff * torch.mean(act ** 2)
            self.perform_list['act'] += loss_act.item()
    
        loss_weight = self.weight_coeff * ( torch.mean(self.N.get_r() ** 2) )
        self.perform_list['weight'] += loss_weight.item()        
        
        #print(output[:,-1,:].size())
        #print(torch.max(output[:,-1,:], dim=1)[1].size())
        #print(y.size())
        correct_num = (torch.max(output, dim=1)[1]==y).sum().item()
        self.perform_list['acc'] += correct_num
        batch_size = y.size(0)
        acc = correct_num / batch_size
        
        self.sample_count += batch_size
        self.batch_count += 1
        #print('sample_count: %d  batch_count: %d'%(self.sample_count, self.batch_count))

        if self.hebb_coeff==0.0:
            loss_hebb = 0.0 
        else:
            loss_hebb = self.get_loss_hebb(act)
        return {
            'loss_total': loss_main + loss_act + loss_weight + loss_hebb,
            'loss_class': loss_main,
            'loss_act': loss_act,
            'loss_weight': loss_weight,
            'loss_hebb': loss_hebb,
            'acc': acc
        }
    
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
                        print('act_std[%d]][%d]==0.0'%(i, j))
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
            #print('act_corr')
            #input()
        self.dict['last_act_corr'] = act_corr.detach().cpu()
        return - hebb_coeff * torch.mean(torch.tanh(torch.abs(weight)) * act_corr)
    def save(self, save_path='./', save_name=None):
        '''
        for key in self.dict['cache'].keys():
            if(self.dict.get(key) is not None):
                self.dict.pop(key)      
        '''
        ensure_path(save_path)
        save_name = self.dict['name'] if save_name is None else save_name
        with open(save_path + save_name, 'wb') as f:
            self.to(torch.device('cpu'))
            torch.save(self.dict, f)
            self.to(self.device)
    def get_weight_ei(self, name, detach=False, positive=True):
        if name in self.N.weight_names:
            return self.N.get_weight(name=name, detach=detach, positive=positive)
        elif name in self.weight_names:
            if name in ['b_0']:
                w = self.b_0
            elif name in ['i']:
                w = self.get_i()
            elif name in ['X->E']:
                w = self.get_i()[:, 0:self.E_num]
            elif name in ['X->I']:
                w = self.get_i()[:, self.E_num:self.N_num]
            else:
                raise Exception('Invalid weight name:%s'%(str(name)))
        else:
            raise Exception('Invalid weight name:%s'%(str(name)))
        if detach:
            w = w.detach()
        return w
    def update_weight_cache_ei(self):
        self.N.update_weight_cache_ei()
        self.cache['weight_cache'] = self.N.cache['weight_cache']
        #print('E->E: ')
        #print(self.cache['weight_cache']['E->E'])
        #print(self.N.get_weight('E->E'))
        #input()
    def get_iter_data(self, input_data, batch_num=None, iter_time=None): #return a list of state dicts in shape of [time_step][key][batch_size, unit_num].
        print('calculating iter_data. sample_num=%d'%(data.size(0)))
        data_loader = split_data_into_batches(data, batch_size=128)
        if(iter_time is None):
            iter_time = self.iter_time
        self.update_weight_cache()
        self.eval()
        count = 0
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

        iter_data['acc'] = [0.0 for _ in range(iter_time)]
        iter_data['loss'] = [0.0 for _ in range(iter_time)]
        #print('aaa')
        #print(len(iter_data['acc']))
        #print(len(iter_data['loss'])) 
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
                iter_data['loss'][time] += self.main_loss_func( torch.squeeze(res['N->Y'][:,time,:]), labels)
                iter_data['acc'][time] += ( torch.max( torch.squeeze(res['N->Y'][:,time,:] ), 1)[1]==labels).sum().item()

        for time in range(iter_time):
            iter_data['loss'][time] = iter_data['loss'][time] / count
            iter_data['acc'][time] = iter_data['acc'][time] / label_count
        cat_dict(iter_data, ress, dim=0) #cat along batch_size dim.         
        self.train()
        #print(len(iter_data['loss']))
        #input()
        return iter_data
    def get_res_data(self, input_data, iter_time=None, batch_num=None):
        print('calculating res_data. batch_num=%d'%(data.size(0)))
        data_loader = split_data_into_batches(data, batch_size=128)
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
        ei = self.get_weight('E->I', positive=True)
        ie = self.get_weight('I->E', positive=True)
        er = self.get_weight('E->E', positive=True)
        ir = self.get_weight('I->I', positive=True)
        weights = [ei, ie, er, ir]
        for w in weights:
            print(w)
            print(torch.mean(w))
            print(torch.min(w))
            print(torch.max(w))
            print(list(w.size()))
    def cache_weight(self):
        self.cache['i'] = self.get_i().detach().cpu().numpy()
        self.cache['r'] = self.get_r().detach().cpu().numpy()
        self.cache['f'] = self.get_f().detach().cpu().numpy()
    def report_weight_update(self):
        i = self.get_i().detach().cpu().numpy()
        r = self.get_r().detach().cpu().numpy()
        f = self.get_f().detach().cpu().numpy()

        i_delta = np.sum(np.abs(i - self.cache['i'])) / np.sum(np.abs(self.cache['i']))
        r_delta = np.sum(np.abs(r - self.cache['r'])) / np.sum(np.abs(self.cache['r']))
        f_delta = np.sum(np.abs(f - self.cache['f'])) / np.sum(np.abs(self.cache['f']))
        print('weight update rate: i:%.4e r:%.4e f:%.4e'%(i_delta, r_delta, f_delta))
    
    def get_weight_corr(self, input_data, save_dir='./data', save_dat=False, load_dat=False, res_data_num=None, iter_data_num=None):
        #save_dat: save computed res_data and iter_data in save_dir.
        #load_dat: load res_data and iter_data if they exist in save_dir.
        net=self
        if save_dat:
            ensure_dir(save_dir)
            ensure_dir(save_dir + 'data/')
        if load_dat:
            res_data, iter_data = read_data(save_dir + 'data/res_data'), read_data(save_dir + 'data/iter_data')
        else:
            res_data, iter_data = None, None
        if res_data is None:
            res_data = net.get_res_data(input_data, iter_time=None)
            if save_dat:
                save_data(res_data, save_dir+'data/'+'res_data')
        if iter_data is None:
            iter_data = net.get_iter_data(input_data, iter_time=None)
            if save_dat:
                save_data(iter_data, save_dir+'data/'+'iter_data')
        #corr_method=['pearson', 'kendall', 'spearman']
        corr_method = ['pearson']

        res_corr = cal_res_corr_pearson(net, res_data, separate_ei=net.dict['separate_ei'], net_dict=net.dict)
        rf_corr = cal_rf_corr_pearson(net, rf=net.get_weight('i', detach=True), separate_ei=net.dict['separate_ei'], net_dict=net.dict)

        weight = net.get_weight('r').detach().clone()
        
        '''
        plot_res_weight_corr(net=net, save_dir=save_dir+'response-weight correlation/', res_corr=res_corr)
        plot_rf_weight_corr(net=net, save_dir=save_dir+'rf-weight correlation/', rf_corr=rf_corr)
        plot_rf_res_corr(net=net, save_dir=save_dir+'rf-res correlation/', rf_corr=rf_corr, res_corr=res_corr) 

        plot_weight(net=net, save_dir=save_dir + 'weight/')
        visualize_weight(net=net, name='r', save_dir=save_dir + 'weight/plot/')

        ensure_dir(save_dir + 'response_analysis/')
        for key in res_data.keys():
            #print(key)
            plot_res(res_data[key], name=key, save_dir=save_dir+'response_analysis/'+key+'/', is_act=('.u' in key))
        
        plot_iter(net, iter_data, save_dir=save_dir+'iter/')
        anal_stability(net=net, iter_data=iter_data, save_dir=save_dir + 'stability/')
        #rf_corr
        for key in rf_corr.keys():
            plot_dist(data=rf_corr[key], name=key, bins=50, save_dir=save_dir + 'rf/' + key + '/')
        plot_rf(net, save_dir = save_dir + 'rf/' + 'plot/')

        for key in res_corr.keys():
            plot_dist(data=res_corr[key], name=key, bins=50, save_dir=save_dir + 'responses/' + key + '/')
        print('quick analysis finished.')
        '''
        return res_corr, rf_corr, weight
                
        '''
        corr:{
            'res_corr':{
                'E->E':(E_num, E_num)
                'E->I':(E_num, I_num)
                'I->E':(I_num, E_num)
                'I->I':(I_num, I_num)
            },
            'rf_corr':{
                'E->E':(E_num, E_num)
                'E->I':(E_num, I_num)
                'I->E':(I_num, E_num)
                'I->I':(I_num, I_num)                
            }
        }
        weight:{
            'E->E':(E_num, E_num)
            'E->I':(E_num, I_num)
            'I->E':(I_num, E_num)
            'I->I':(I_num, I_num)
        }
        '''

    def get_E_num(self):
        return self.dict['E_num']
    def get_I_num(self):
        return self.dict['I_num']
    def get_N_num(self):
        return self.dict['N_num']
    def plot_weight_resp_corr(self, ax, save=False, save_path='./', save_name='RSLP_weight_resp_corr.png'):
        fig, axes = plt.subplots()

        # to be implemented

        return
    
    
    
        

        




