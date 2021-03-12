import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt
plt.box(False)
from matplotlib.ticker import FuncFormatter 
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import scipy

from anal_functions import *
from models import RNN_EI, RCNN_EI
#from my_logger import my_logger
#from train_tools import evaluate, prepare_CIFAR10, evaluate_iter

import time
import os
import math

#from config import CIFAR10_dir, device, logger,  large_fontsize, device, epoch_num

import matplotlib as mpl
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

from imageio import imsave

import traceback

def main():
    save_dir = save_dir_anal
    f0 = open(save_dir+"anal_log.txt",'w')
    logger=my_logger()
    logger.clear()
    logger.add_stdout()
    logger.add_flow(f0, name="log_file")
    net = load_net(50)
    if(model_type=="rnn"):
        quick_anal(net=net, logger=logger, save_dir=save_dir_anal)
        pass
    elif(model_type=="rcnn"):
        quick_anal_rcnn(net=net, logger=logger, save_dir=save_dir_anal)
        pass
    test(logger, net, save_dir_anal+"test/")    
    f0.close()

def test(logger, net, save_dir, save_dat=True):
    ensure_dir(save_dir)
    print("test analysis.")

    trainloader, testloader = prepare_CIFAR10(dataset_dir=CIFAR10_dir, augment=False, norm=True)

    trainloader = list(trainloader)
    testloader = list(testloader)
    if(anal_loader == "train"):
        dataloader = trainloader
    elif(anal_loader == "test"):
        dataloader = testloader

    ensure_dir(save_dir + "data/")
    res_data, iter_data = read_data(save_dir + "data/res_data"), read_data(save_dir + "data/iter_data")
    if res_data is None:
        res_data = net.get_res_data(random.sample(dataloader, 30), iter_time=None)
        if save_dat:
            save_data(res_data, save_dir+"data/"+"res_data")
    if iter_data is None:
        iter_data = net.get_iter_data(random.sample(dataloader, 2), iter_time=None)
        if save_dat:
            save_data(iter_data, save_dir+"data/"+"iter_data")

    res_corr = cal_res_corr_pearson(net, res_data, separate_ei=net.dict["separate_ei"], net_dict=net.dict)
    rf_corr = cal_rf_corr_pearson(net, rf=net.get_weight("i", detach=True), separate_ei=net.dict["separate_ei"], net_dict=net.dict)

    plot_res_weight_corr(net=net, save_dir=save_dir+"response-weight correlation/", res_corr=res_corr, nonlog_plot=True, density_plot=True)
    plot_rf_weight_corr(net=net, save_dir=save_dir+"rf-weight correlation/", rf_corr=rf_corr, nonlog_plot=True, density_plot=True)
    plot_rf_res_corr(net=net, save_dir=save_dir+"rf-res correlation/", rf_corr=rf_corr, res_corr=res_corr, nonlog_plot=True, density_plot=True) 

    #plot_weight(net=net, logger=logger, save_dir=save_dir + "weight/", density_plot=True)
    #visualize_res_weight_corr(net=net, res_data=res_data, save_dir = save_dir)
    '''    
    res_corr = cal_res_corr_pearson(net, res_data, separate_ei=net.dict["separate_ei"], net_dict=net.dict)
    r_corr = res_corr["E.u"]
    color_map = plt.cm.get_cmap('jet')
    r_corr = ( r_corr - np.min(r_corr) ) / ( np.max(r_corr) - np.min(r_corr) )
    r_corr = color_map(r_corr)
    #print(r_corr)
    r_corr = np.uint8(r_corr * 255)
    print(r_corr.shape)
    imsave(save_dir + "/" + "E-E correlation.png", r_corr)
    '''

def quick_anal(logger, net, save_dir, save_dat=False):
    ensure_dir(save_dir)
    print("quick analysis.")

    trainloader, testloader = prepare_CIFAR10(dataset_dir=CIFAR10_dir, augment=False, norm=True)

    trainloader = list(trainloader)
    testloader = list(testloader)
    if(anal_loader == "train"):
        dataloader = trainloader
    elif(anal_loader == "test"):
        dataloader = testloader

    ensure_dir(save_dir + "data/")

    if load_dat: #try loading from existing data files.
        res_data, iter_data = read_data(save_dir + "data/res_data"), read_data(save_dir + "data/iter_data")
    else:
        res_data, iter_data = None, None
    if res_data is None:
        res_data = net.get_res_data(random.sample(dataloader, 30), iter_time=None)
        if save_dat:
            save_data(res_data, save_dir+"data/"+"res_data")
    if iter_data is None:
        iter_data = net.get_iter_data(random.sample(dataloader, 2), iter_time=None)
        if save_dat:
            save_data(iter_data, save_dir+"data/"+"iter_data")
    
    #corr_method=["pearson", "kendall", "spearman"]
    corr_method=["pearson"]

    res_corr = cal_res_corr_pearson(net, res_data, separate_ei=net.dict["separate_ei"], net_dict=net.dict)
    rf_corr = cal_rf_corr_pearson(net, rf=net.get_weight("i", detach=True), separate_ei=net.dict["separate_ei"], net_dict=net.dict)

    plot_res_weight_corr(net=net, save_dir=save_dir+"response-weight correlation/", res_corr=res_corr)
    plot_rf_weight_corr(net=net, save_dir=save_dir+"rf-weight correlation/", rf_corr=rf_corr)
    plot_rf_res_corr(net=net, save_dir=save_dir+"rf-res correlation/", rf_corr=rf_corr, res_corr=res_corr) 

    plot_weight(net=net, logger=logger, save_dir=save_dir + "weight/")
    visualize_weight(net=net, name="r", save_dir=save_dir + "weight/plot/")

    ensure_dir(save_dir + "response_analysis/")
    for key in res_data.keys():
        #print(key)
        plot_res(res_data[key], name=key, save_dir=save_dir+"response_analysis/"+key+"/", is_act=(".u" in key))
    
    plot_iter(net, iter_data, logger=logger, save_dir=save_dir+"iter/")
    anal_stability(net=net, iter_data=iter_data, save_dir=save_dir + "stability/")
    #rf_corr
    for key in rf_corr.keys():
        plot_dist(data=rf_corr[key], logger=logger, name=key, bins=50, save_dir=save_dir + "rf/" + key + "/")
    plot_rf(net, save_dir = save_dir + "rf/" + "plot/")

    for key in res_corr.keys():
        plot_dist(data=res_corr[key], logger=logger, name=key, bins=50, save_dir=save_dir + "responses/" + key + "/")
    print("quick analysis finished.")

def quick_anal_rcnn(logger, net, save_dir, save_dat=False):
    ensure_dir(save_dir)
    print("quick analysis for rcnn.")
    trainloader, testloader = prepare_CIFAR10(dataset_dir=CIFAR10_dir, augment=False, norm=True)

    trainloader = list(trainloader)
    testloader = list(testloader)
    if(anal_loader == "train"):
        dataloader = trainloader
    elif(anal_loader == "test"):
        dataloader = testloader

    ensure_dir(save_dir + "data/")
    res_data, iter_data = read_data(save_dir + "data/res_data"), read_data(save_dir + "data/iter_data")
    if res_data is None:
        res_data = net.get_res_data(random.sample(dataloader, anal_res_batch_num), iter_time=None)
        if save_dat:
            save_data(res_data, save_dir+"data/"+"res_data")
    if iter_data is None:
        iter_data = net.get_iter_data(random.sample(dataloader, anal_iter_batch_num), iter_time=None)
        if save_dat:
            save_data(iter_data, save_dir+"data/"+"iter_data")
    
    #print(res_data.keys())

    for layer_name in anal_layers:
        print("analysis of layer: %s"%(layer_name))
        quick_anal_rconv(net.layers[net.get_layer_index(layer_name)], save_dir + layer_name + "/", res_data[layer_name], iter_data[layer_name])

    print("quick analysis for rcnn finished.")


def quick_anal_rconv(net, save_dir, res_data, iter_data, full_anal=False):
    ensure_dir(save_dir)
    #corr_method=["pearson", "kendall", "spearman"]
    corr_method=["pearson"]

    for key in res_data.keys():
        data = res_data[key] #(sample_num, N_num, feature_map_width, feature_map_height)
        #print(data.size())
        data = data.permute(0, 2, 3, 1) #(sample_num, feature_map_width, feature_map_height, N_num)
        #print(data.size())
        res_data[key] = data.contiguous().view(data.size(0) * data.size(1) * data.size(2), data.size(3))
        
    res_corr = cal_res_corr_pearson(net, res_data, separate_ei=net.dict["separate_ei"], net_dict=net.dict)

    plot_res_weight_corr(net=net, save_dir=save_dir+"response-weight correlation/", res_corr=res_corr)

    plot_weight_rcnn(net=net, logger=logger, save_dir=save_dir + "weight/")
    visualize_weight(net=net, name="r", save_dir=save_dir + "weight/plot/")

    if full_anal:
        ensure_dir(save_dir + "response_analysis/")
        for key in res_data.keys():
            #print(key)
            plot_res(res_data[key], name=key, save_dir=save_dir+"response_analysis/"+key+"/", is_act=(".u" in key))

    #randomly select one column to do analysis.
    for key in iter_data.keys():
        data = iter_data[key] #(sample_num, iter_time, N_num, feature_map_width, feature_map_height)
        #print(key, end=' ')
        #print(data.size())
        width = random.sample(range(data.size(3)), 1)[0]
        height = random.sample(range(data.size(4)), 1)[0]
        iter_data[key] = data[:, :, :, width, height]

    if full_anal:
        plot_iter(net, iter_data, logger=logger, save_dir=save_dir+"iter/")
    
    anal_stability(net=net, iter_data=iter_data, save_dir=save_dir + "stability/")

    for key in res_corr.keys():
        plot_dist(data=res_corr[key], logger=logger, name=key, bins=50, save_dir=save_dir + "responses/" + key + "/")

def plot_res_weight_corr(net, save_dir, res_corr, nonlog_plot=True, density_plot=False):
    print("analyzing response - weight correlation")
    ensure_dir(save_dir)
    net.update_weight_cache()
    weight = net.cache["weight_cache"]
    if(net.dict["separate_ei"]==True):
        pairs = [
            [res_corr['E.u'], weight['E->E'], ["E-E u","weight"], save_dir + "E-E/"],
            [res_corr['E-I.u'], weight['E->I'], ["E-I u","weight"], save_dir + "E-I/"],
            [res_corr['E-I.u'].T, weight['I->E'], ["I-E u","weight"], save_dir + "E-I/"],
            [res_corr['I.u'], weight['I->I'], ["I-I u","weight"], save_dir + "I-I/"],
            [res_corr['E.x'], weight['E->E'], ["E-E x","weight"], save_dir + "E-E/x/"], #[res_corr, weight, name, save_dir]
            [res_corr['E-I.x'], weight['E->I'], ["E-I x","weight"], save_dir + "E-I/x/"],
            [res_corr['E-I.x'].T, weight['I->E'], ["I-E x","weight"], save_dir + "E-I/x/"],
            [res_corr['I.x'], weight['I->I'], ["I-I x","weight"], save_dir + "I-I/x/"],
        ]
        #print(weight['E->E'])
    else:
        pairs = [
            [res_corr['N.x'], weight['N->N'], ["N-N x","weight"], save_dir],
            [res_corr['N.u'], weight['N->N'], ["E-I x","weight"], save_dir + "u/"],
        ]
    for pair in pairs:
        if net.dict["noself"]:
            plot_corr(x=get_flattened_array(pair[0], remove_diagonal=True), y=get_flattened_array(pair[1].detach().cpu().numpy(), remove_diagonal=True), name=pair[2], save_dir=pair[3], anal_stat=True, 
                nonlog_plot=nonlog_plot, density_plot=density_plot)

def plot_rf_weight_corr(net, save_dir, rf_corr, nonlog_plot=True, density_plot=False):
    print("analyzing rf - weight correlation")
    ensure_dir(save_dir)
    net.update_weight_cache()
    weight = net.cache["weight_cache"]
    if(net.dict["separate_ei"]==True):
        pairs = [
            [rf_corr['E'], weight['E->E'], ["E-E rf","weight"], save_dir + "E-E/"],
            [rf_corr['E-I'], weight['E->I'], ["E-I rf","weight"], save_dir + "E-I/"],
            [rf_corr['E-I'].T, weight['I->E'], ["I-E rf","weight"], save_dir + "E-I/"],
            [rf_corr['I'], weight['I->I'], ["I-I rf","weight"], save_dir + "I-I/"],
        ]
    else:
        pairs = [
            [rf_corr['N'], weight['N->N'], ["N-N rf","weight"], save_dir],
        ]
    for pair in pairs:
        plot_corr(x=get_flattened_array(pair[0], remove_diagonal=True), y=get_flattened_array(pair[1].detach().cpu().numpy(), remove_diagonal=True), name=pair[2], save_dir=pair[3], anal_stat=True, 
            nonlog_plot=nonlog_plot, density_plot=density_plot)

def plot_rf_res_corr(net, save_dir, res_corr, rf_corr, nonlog_plot=True, density_plot=False):
    print("analyzing rf - response correlation")
    ensure_dir(save_dir)
    if(net.dict["separate_ei"]==True):
        pairs = [
            [rf_corr['E'], res_corr['E.x'], ["E-E rf","E.x"], save_dir + "E.x/"],
            [rf_corr['E'], res_corr['E.u'], ["E-E rf","E.u"], save_dir + "E.u/"],
            [rf_corr['I'], res_corr['I.x'], ["I-I rf","I.x"], save_dir + "I.x/"],
            [rf_corr['I'], res_corr['I.u'], ["I-I rf","I.u"], save_dir + "I.x/"],
            [rf_corr['E-I'], res_corr['E-I.x'], ["E-I rf","E-I.x"], save_dir + "E.x-I.x/"],
            [rf_corr['E-I'], res_corr['E-I.u'], ["E-I rf","E-I.u"], save_dir + "E.u-I.u/"],
        ]

    else:
        pairs = [
            [rf_corr['N'], res_corr['N.x'], ["N-N weight","N.x"], save_dir + "N.x/"],
            [rf_corr['N'], res_corr['N.u'], ["N-N weight","N.u"], save_dir + "N.u/"],
        ]
    for pair in pairs:
        plot_corr(x=get_flattened_array(pair[0],remove_diagonal=True), y=get_flattened_array(pair[1], remove_diagonal=True), name=pair[2], save_dir=pair[3], anal_stat=False, log_plot=False, 
            nonlog_plot=nonlog_plot, density_plot=density_plot)

def eval_net(net, logger):
    logger.write("evaluating net performance")
    val_loss, val_acc=cal_iter_data(net)
    logger.write('val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc))

def plot_iter(net, iter_data, logger, save_dir=None, redo=True, whole_ylim=False):    
    ensure_dir(save_dir)


    if(iter_data.get("loss") is not None):
        iter_time = len(iter_data["loss"])
        ensure_dir(save_dir + "performance/")
        logger.write("plotting performance - iter_time")
        plt.subplot(2, 1, 1)
        if(whole_ylim==True):
            plt.ylim(0.0, np.max(val_loss)*1.1)
        #print(len(iter_data["loss"]))
        #print(iter_time)
        #print(len(iter_data["acc"]))
        #print(iter_time)
        plt.plot(range(iter_time), iter_data["loss"], '-', label='train', color='r')
        plt.title('loss - iter_time', fontsize=large_fontsize)
        plt.ylabel('loss', fontsize=large_fontsize)
        plt.xlabel('iter_time', fontsize=large_fontsize)

        plt.subplot(2, 1, 2)
        if whole_ylim:
            plt.ylim(0.0, 1.0)
        #print(len(iter_data["acc"]))
        #print(iter_time)
        plt.plot(range(iter_time), iter_data["acc"], '-', label='train', color='r')
        plt.title('acc - iter_time', fontsize=large_fontsize)
        plt.ylabel('acc', fontsize=large_fontsize)
        plt.xlabel('iter_time', fontsize=large_fontsize)
        plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
        plt.savefig(save_dir + "performance/" + "performance - iter_time.jpg")
        plt.close()
    
        if net.dict["separate_ei"]:
            ensure_dir(save_dir + "gif/")
            img_dirs=[]
            for time in range(iter_time):
                print("\r","plotting E.u dist. progress:%d/%d "%(time+1,iter_time), end="", flush=True)
                img_dir=plot_dist(logger=logger, data=iter_data["E.u"][:, time, :].detach().cpu().numpy(), name="iter_time:%d"%(time+1), ran=[], 
                    save_dir=save_dir + "gif/", kde=False, hist=True, notes="acc:%.4e loss:%.4e"%(iter_data["acc"][time], iter_data["loss"][time]), redo=True)
                img_dirs.append(img_dir["hist"])
            print("\n")
            create_gif(img_dirs, gif_name=(save_dir+ "gif/" + "E.u_gif" + ".gif"), duration = 0.2)

    ensure_dir(save_dir + "ratio/")
    if net.dict["separate_ei"]:
        iter_time = iter_data["E.x"].size(1)
        keys=[["X->E", "E->E", "I->E"], ["X->I", "E->I", "I->I"], ["E->Y", "I->Y"]]
        names = ["E.input", "I.input", "Y.input"]
        save_names=["X,E,I-E", "X,E,I-I", "E,I-Y"]
        count = 0
        for key in keys:
            for time in range(iter_time):
                try:
                    plot_ratio(data=list(map(lambda x:torch.squeeze(iter_data[x][:, time, :]).detach().cpu().numpy(), key)), 
                        data2=None, logger=logger, name=key, name2=names[count] + " iter=%d"%(time), 
                        save_dir=save_dir + "ratio/" + save_names[count] + "/")
                except Exception:
                    print("exception when anal_ratio:"+str(key))
                    traceback.print_exc()
                    input()
            count += 1
def plot_weight(net, logger, save_dir=None, density_plot=False):
    ensure_dir(save_dir)
    logger.write("analyzing weight")
    if(net.dict["separate_ei"]==True):
        weights = ["X->E", "X->I", "E->E", "E->I", "I->E", "I->I", "E->Y", "I->Y", "E.b", "I.b"]
        save_names = ["X-E", "X-I", "E-E", "E-I", "I-E", "I-I", "E-Y", "I-Y", "E.b", "I.b"]
    else:
        weights = ["i", "r", "f", "b"]
    count = 0
    for name in weights:
        weight = net.get_weight(name, detach=False, positive=True)
        if not isinstance(weight, torch.Tensor):
            if isinstance(weight, float):
                print("%s is a float"%(name))
            else:
                print("%s is not torch.Tensor"%(name))
            continue
        
        weight = weight.detach().cpu().numpy()
        print("analysis of weight: %s" %(name), end='')
        print(weight.shape)

        plot_weight_stat(weight=weight, logger=logger, name=name, save_dir=save_dir + save_names[count] + "/")
        if(name in ["r", "E->E", "I->I"]):
            plot_weight_rec(weight, name, save_dir + save_names[count] + "/", density_plot=density_plot)
            
        count += 1
    if ( ("E->I" in weights) and ("I->E" in weights) ): 
        plot_weight_lat(net.get_weight("E->I", detach=True, positive=True), net.get_weight("I->E", detach=True, positive=True).t(), name="E-I", save_dir=save_dir+"E and I/", density_plot=density_plot)

def plot_weight_rcnn(net, logger, save_dir=None, density_plot=False):
    ensure_dir(save_dir)
    logger.write("analyzing weight")
    if net.dict["separate_ei"]:
        weights = ["E->E", "E->I", "I->E", "I->I", "E->Y", "I->Y", "E.b", "I.b"]
        save_names = ["E-E", "E-I", "I-E", "I-I", "E-Y", "I-Y", "E.b", "I.b"]
    else:
        weights = ["i", "r", "f", "b"]
    count = 0
    for name in weights:

        weight = net.get_weight(name, detach=False, positive=True)

        if not isinstance(weight, torch.Tensor):
            if isinstance(weight, float):
                print("%s is a float"%(name))
            else:
                print("%s is not torch.Tensor"%(name))
            continue
        
        weight = torch.squeeze(weight)
        weight = weight.detach().cpu().numpy()
        print("analysis of weight: %s" %(name), end='')
        print(weight.shape)

        plot_weight_stat(weight=weight, logger=logger, name=name, save_dir=save_dir + save_names[count] + "/")
        if(name in ["r", "E->E", "I->I"]):
            plot_weight_rec(weight, name, save_dir + save_names[count] + "/", density_plot=density_plot)
            
        count += 1
    if ( ("E->I" in weights) and ("I->E" in weights) ): 
        e2i = torch.squeeze(net.get_weight("E->I", detach=True, positive=True))
        i2e = torch.squeeze(net.get_weight("I->E", detach=True, positive=True)).t()
        print(e2i.size())
        print(i2e.size())
        plot_weight_lat(e2i, i2e, name="E-I", save_dir=save_dir+"E and I/", density_plot=density_plot)



if __name__ == '__main__':
    main()