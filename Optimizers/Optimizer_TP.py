import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim

from LRSchedulers import LinearLR
from Optimizers import Optimizer

import utils
from utils import get_from_dict, search_dict, ensure_path
import utils_model
from utils_model import build_mlp, build_optimizer

class Optimizer_TP(Optimizer):   
    def __init__(self, dict_=None, load=False):
        super().__init__(dict_, load)
        #self.modules += [self.optimizer_forward, self.optimizer_rec, self.optimizer_out] # torch.Optimizer objects don't have to() method.
        self.load = load
        self.device = 'cpu' if self.dict.get('device') is None else self.dict['device']

        self.mode = search_dict(self.dict, ['mode'], default='train_on_u', write_default=True)

        if self.mode in ['train_on_r']:
            self.train = self.train_on_r
        elif self.mode in ['train_on_u']:
            self.train = self.train_on_u
        else:
            raise Exception('Invalid mode: %s'%str(self.mode))

        self.get_target_method = search_dict(self.dict, ['get_target_method', 'get_target'], default='gradient', write_default=True)
        self.loss_func_str = search_dict(self.dict, ['loss_func', 'main_loss_func'], default='CEL', write_default=True)

        if self.get_target_method in ['naive', 'force']:
            self.get_target = self.get_target_naive
        elif self.get_target_method in ['gradient', 'gradient-based']:
            if self.loss_func in ['CEL']:
                self.loss_func = nn.CrossEntropyLoss()
                self.get_target = self.get_target_single_class_cel
            else:
                pass
                # to be implemented

    def bind_model(self, model):
        self.model = model
        self.build_decoder(load=self.load)
        self.build_optimizer(load=self.load)
    def bind_trainer(self, trainer):
        self.trainer = trainer
        self.update_epoch_init()
    def build_optimizer(self, load=False):
        self.optimizer_forward = utils_model.build_optimizer(self.dict['optimizer_forward'], model=self.model, load=load)
        self.optimizer_rec = utils_model.build_optimizer(self.dict['optimizer_rec'], model=self.decoder_rec, load=load)
        self.optimizer_out = utils_model.build_optimizer(self.dict['optimizer_out'], model=self.decoder_out, load=load)
    def receive_args(self, args):
        if isinstance(args, dict):
            for key, item in args.items():
                if key in ['device']:
                    self.device = self.dict_['device'] = item
        else:
            raise Exception('Invalid arg type: '%type(args)) 
    def build_decoder(self, load=False):
        if self.model is None:
            raise Exception('Optimizer_TP: no model bound yet.')

        #print(self.model.dict['output_num'])
        for decoder_type in ['rec', 'out']: # decoder_in is not needed in target_propagation.
            if decoder_type in ['rec']:
                dict_ = get_from_dict(self.dict, 'decoder_rec', default={}, write_default=True)
            elif decoder_type in ['out', 'output']:
                dict_ = get_from_dict(self.dict, 'decoder_out', default={}, write_default=True)
            #elif decoder_type in ['in', 'input']:
            #    dict_ = get_from_dict(self.dict, 'decoder_in', default={}, write_default=True)
            else:
                raise Exception('invalid decoder_type %s'%decoder_type) 

            dict_['act_func'] = get_from_dict(self.dict['decoder'], 'act_func', default='softplus', write_default=True)
            dict_['type'] = get_from_dict(self.dict['decoder'], 'type', default='mlp', write_default=True)

            if dict_['type'] in ['mlp', 'MLP']:
                dict_['bias'] = get_from_dict(self.dict['decoder'], 'bias', default=True, write_default=True)
                if decoder_type in ['rec']:
                    #dict_['unit_nums'] = [self.model.dict['N_num'], 2 * self.model.dict['N_num']] #r_out -> i_ : r_in
                    dict_['unit_nums'] = [self.model.dict['N_num'], self.model.dict['N_num']] #r_out -> i_ : r_in
                elif decoder_type in ['out', 'output']:
                    #dict_['unit_nums'] = [self.model.dict['output_num'], 2 * self.model.dict['N_num']] #o_ -> i_ : r_in
                    dict_['unit_nums'] = [self.model.dict['output_num'], self.model.dict['N_num']] #o_ -> i_ : r_in
                #elif decoder_type in ['in', 'input']:
                #    dict_['unit_nums'] = [self.model.dict['N_num'], 2 * self.model.dict['input_num']] #i_ -> x
                else:
                    raise Exception('invalid decoder_type %s'%decoder_type) 
                decoder = build_mlp(dict_, load=load)
                decoder.to(self.device)
            else:
                raise Exception('unsupported decoder type: %s'%str(decoder_type))
            
            if decoder_type in ['rec']:
                self.decoder_rec = decoder
            elif decoder_type in ['out', 'output']:
                self.decoder_out = decoder
            elif decoder_type in ['in', 'input']:
                self.decoder_in = decoder
            else:
                raise Exception('invalid decoder_type %s'%decoder_type)
        self.modules = [self.decoder_rec, self.decoder_out]
    def update_epoch_init(self):
        self.update_func_list = []
        self.update_lr_init()
        self.update_func_list.append(self.update_lr_init())
    def update_batch(self, **kw):
        return
    def update_epoch(self, **kw): #epoch_ratio: epoch_current / epoch_num_in_total
        for func in self.update_func_list:
            func(**kw)
    def update_lr_init(self):
        self.lr_decay = get_from_dict(self.dict, 'lr_decay', default={'method':'none'}, write_default=True)
        lr_decay_method = self.lr_decay.get('method')
        if lr_decay_method in ['None', 'none'] or lr_decay_method is None:
            pass
        elif lr_decay_method in ['exp']:
            decay = search_dict(self.lr_decay, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
            self.scheduler_forward = torch.optim.lr_scheduler.ExponentialLR(optimizer_forward, gamma=decay)
            self.scheduler_rec = torch.optim.lr_scheduler.ExponentialLR(optimizer_rec, gamma=decay)
            self.scheduler_out = torch.optim.lr_scheduler.ExponentialLR(optimizer_out, gamma=decay)
            return self.update_lr
        elif lr_decay_method in ['linear']:
            milestones = search_dict(self.lr_decay, ['milestones'], write_default=False)
            epoch_num = self.trainer.dict['epoch_num']
            self.scheduler_forward = LinearLR(self.optimizer_forward, milestones=milestones, epoch_num=epoch_num)
            self.scheduler_rec = LinearLR(self.optimizer_rec, milestones=milestones, epoch_num=epoch_num)
            self.scheduler_out = LinearLR(self.optimizer_out, milestones=milestones, epoch_num=epoch_num)
            return self.update_lr
        else:
            raise Exception('Invalid lr decay method: '+str(lr_decay_method))
    def update_lr(self, **kw):
        self.scheduler_forward.step()
        self.scheduler_rec.step()
        self.scheduler_out.step()
    def evaluate(self, x, y): #x_: input_data, y_:output_target.
        self.model.get_perform(x, y)
    def train_on_r(self, x, y, iter_time_=None): #x_: input_data, y_:output_target.
        if iter_time_ is None:
            iter_time = self.model.dict['iter_time']
        else:
            iter_time = iter_time_
        results = {}
        self.optimizer_forward.zero_grad()
        self.optimizer_out.zero_grad()
        self.optimizer_rec.zero_grad()
        r_s = []
        r_detach = []
        r_out_detach = []
        self.model.forward_init(x)
        r_ = r_init = torch.zeros([x.size(0), self.model.dict['N_num']], device=self.device)
        for time in range(iter_time):
            results = self.model.forward_once(x_=x, r_in=r_, detach_i=False) # i_detach=False so that gradient can propagate through i.
            #i_, i_detach, r_out, o_ = results['i_'], results['i_detach'], results['r_'], results['o_']
            i_, r_, o_ = results['i_'], results['r_'], results['o_']
            #r_out_detach.append(r_out.detach())
            #if time==(iter_time-1):
            #    outputs.append(o_)
            #else:
            #    outputs.append(r_)
            #i_r_detach.append(torch.cat([r_in, i_detach], dim=1)) 
            r_s.append(r_)
            r_ = r_.detach()
            r_detach.append(r_)
            #i_s.append(i_)
            #o_s.append(o_) #for evaluation of performance
        output = o_
        self.model.cal_perform_from_outputs(output, act=None, y=y)

        # train decoder
        '''
        for time in range(iter_time):
            if time==(iter_time-1):
                output = outputs[time].detach() #block gradient flow.
                input_pred = self.decoder_out(output)
                loss_o = F.mse_loss(i_r_detach[time], input_pred)
                loss_o.backward(retain_graph=True)
            else:
                output = outputs[time].detach()
                input_pred = self.decoder_rec(output)
                loss_r = F.mse_loss(i_r_detach[time], input_pred)
                loss_r.backward(retain_graph=True) 
        '''
        for time in range(iter_time):
            if time==(iter_time-1):
                #output = outputs[time].detach() #block gradient flow.
                #input_pred = self.decoder_out(output)
                output_detach = output.detach()
                r_in_pred = self.decoder_out(output_detach)
                loss_o = F.mse_loss(r_detach[time-1], r_in_pred)
                loss_o.backward(retain_graph=True)
            elif time > 0:
                #output = outputs[time].detach()
                #input_pred = self.decoder_rec(output)
                r_in_pred = self.decoder_rec(r_detach[time])
                loss_r = F.mse_loss(r_detach[time-1], r_in_pred)
                loss_r.backward(retain_graph=True) 
        '''
        for time in range(iter_time):
            if time==(iter_time-1):
                output = outputs[time].detach() #block gradient flow.
                input_pred = self.decoder_out(output)
                loss_o = F.mse_loss(r_detach[time], input_pred)
                loss_o.backward(retain_graph=True)
            else:
                output = outputs[time].detach()
                input_pred = self.decoder_rec(output)
                loss_r = F.mse_loss(r_detach[time], input_pred)
                loss_r.backward(retain_graph=True) 
        '''
        self.optimizer_out.step()
        self.optimizer_rec.step()

        # train model
        target = self.get_target(output=output_detach, truth=y)
        time = iter_time-1
        while time>=0:
            if time==(iter_time-1):
                loss_f = F.mse_loss(output, target) # update gradient of the last iteration
                results['loss'] = loss_f
                loss_f.backward(retain_graph=True)
            else:
                loss_r = F.mse_loss(r_s[time-1], target) # update grad of recurrent_weight.
                loss_r.backward(retain_graph=True)      

            if time==(iter_time-1):
                target = self.decoder_out(target)
            else:
                target = self.decoder_rec(target)

            #loss_i = F.mse_loss(i_s[time], target) # update grad of input_weight.
            #loss_i.backward(retain_graph=True)
            time -= 1
        self.optimizer_forward.step()
        '''
        if self.requires_acc:
            correct_num, data_num = get_acc_from_labels(outputs[-1], y_)
            results['corret_num'] = correct_num
            results['data_num'] = data_num
        return results
        '''
    def train_on_u(self, x, y, iter_time_=None): #x_: input_data, y_:output_target.
        if iter_time_ is None:
            iter_time = self.model.dict['iter_time']
        else:
            iter_time = iter_time_
        results = {}
        self.optimizer_forward.zero_grad()
        self.optimizer_out.zero_grad()
        self.optimizer_rec.zero_grad()
        u_s = []
        u_detach_s = []
        # forward process
        self.model.forward_init(x)
        r_ = r_init = torch.zeros([x.size(0), self.model.dict['N_num']], device=self.device)
        for time in range(iter_time):
            results = self.model.forward_once(x_=x, r_in=r_, detach_i=False, detach_u=True) # i_detach=False so that gradient can propagate through i.
            r_, o_, u, u_detach = results['r_'], results['o_'], results['u'], results['u_detach']
            u_s.append(u)
            u_detach_s.append(u_detach)
        output = o_
        self.model.cal_perform_from_outputs(output, act=None, y=y)

        # train decoder
        for time in range(iter_time):
            if time==(iter_time-1):
                output_detach = output.detach() #block gradient flow.
                u_pred = self.decoder_out(output_detach)
                loss_o = F.mse_loss(u_detach_s[time-1], u_pred)
                loss_o.backward(retain_graph=True)
            elif time > 0:
                u_pred = self.decoder_rec(u_detach_s[time])
                loss_r = F.mse_loss(u_detach_s[time-1], u_pred)
                loss_r.backward(retain_graph=True)

        self.optimizer_out.step()
        self.optimizer_rec.step()

        # train model
        o_target = self.get_target(output=output_detach, truth=y).detach()
        
        loss_f = F.mse_loss(output, o_target) # update grad of output_weight.
        results['loss'] = loss_f
        loss_f.backward(retain_graph=True)
        u_target = self.decoder_out(o_target).detach()
        
        time = iter_time-1
        while time > 0:
            loss_u = F.mse_loss(u_s[time], u_target) # update grad of recurrent_weight.
            loss_u.backward(retain_graph=True)
            u_target = self.decoder_rec(u_target).detach()
            time -= 1
        self.optimizer_forward.step()

    def get_loss(self, i, target):
        o = self.forward(i)
        loss= self.loss_func(o, target)
        #print(loss)
        return loss, o
    def get_target_naive(self, output, truth):
        if len(list(truth.size()))==1:
            target = torch.zeros((truth.size(0), 10), device=self.device).scatter_(1, torch.unsqueeze(truth, 1), 1)
        return truth
    '''
    def get_o_target_single_class_mse(self, o_, target):
        if len(list(target.size()))==1:
            target = torch.zeros((target.size(0), 10), device=self.device).scatter_(1, torch.unsqueeze(target, 1), 1)
        o_ = o_.detach()
        #margin_mask = ( (target - o_) > 0.1 ).float() + ( (target - o_) < - 0.1 ).float() * (-1.0)
        #o_delta = margin_mask * 0.1 + (margin_mask==0.0).float() * (target - o_)
        o_delta = target - o_
        o_target = o_ + o_delta
        return o_target
    '''
    def get_target_single_class_cel(self, output, truth):
        output = output.detach()
        output.requires_grad = True
        #print(target)
        loss = self.loss_func(output, truth)
        loss.backward()

        ratio = torch.sum(torch.abs(output.grad)) / torch.sum(torch.abs(output)).item()
        target = output - 0.1 / ratio * output.grad
        print('grad_output_ratio: ' + str(ratio))
        print(output[0])
        print(target[0])
        return target.detach()
    
    def save(self, save_path='./', save_name=None):
        ensure_path(save_path)
        save_name = 'optimizer_tp' if save_name is None else save_name
        with open(save_path + save_name, 'wb') as f:
            for module in self.modules:
                module.to(torch.device('cpu'))

            self.dict['dict_optimizer_forward'] = self.optimizer_forward.state_dict()
            self.dict['dict_optimizer_out'] = self.optimizer_rec.state_dict()
            self.dict['dict_optimizer_rec'] = self.optimizer_out.state_dict()
            # to be implemented. turn tensors on GPU in optimizer dicts to tensors on CPU.

            torch.save(self.dict, f)
            for module in self.modules:
                module.to(self.options.device)
    