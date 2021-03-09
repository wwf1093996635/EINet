import time
import os
import re
import random
import math


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

import pandas as pd
import numpy as np
import scipy
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
import traceback # use traceback.print_exc() to print exception info.
from scipy.stats import gaussian_kde
import imageio
from imageio import imsave

from config import *
from models import RNN_EI, RCNN_EI

def plot_training_curve(stat_path, loss_only=False):
    try:
        f=open(stat_path, 'rb')
        train_loss_list=pickle.load(f)
        train_acc_list=pickle.load(f)
        val_loss_list=pickle.load(f)
        val_acc_list=pickle.load(f)
        f.close()
        if(loss_only==False):
            plot_training_curve_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        else:
            plot_training_curve_loss(train_loss_list, val_loss_list)
    except Exception:
        print("exception when printing training curve.")

def plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    print("plotting training curve.")
    x = range(len(train_acc_list))
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss', fontsize=large_fontsize)
    plt.ylabel('loss', fontsize=large_fontsize)
    plt.xlabel('epoch', fontsize=large_fontsize)
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, '-', label='train', color='r')
    plt.plot(x, val_acc_list, '-', label='test', color='b')
    plt.title('Accuracy', fontsize=large_fontsize)
    plt.ylabel('acc', fontsize=large_fontsize)
    plt.xlabel('epoch', fontsize=large_fontsize)
    plt.legend(loc='best')
    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.savefig(save_path_anal+"training_curve.jpg")
    plt.close()

def plot_loss(train_loss_list, val_loss_list):
    print("plotting training curve.")
    fig = plt.figure()
    x = range(len(train_loss_list))
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    x = range(len(val_loss_list))
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss - epoch', fontsize=large_fontsize)
    plt.ylabel('loss', fontsize=large_fontsize)
    plt.xlabel('epoch', fontsize=large_fontsize)
    plt.legend(loc='best')
    plt.close()

def load_net(epoch=None, save_=None): #load lastest model saved in save_path_stat
    if epoch is None:
        net_path = get_last_model(model_prefix=model_name + "_epoch_", base_path=path_, is_path=True)
    else:
        net_path=save_path_stat + model_name+"_epoch_%d/"%(epoch)

    if net_path in ["error"] or net_path is None:
        return "error"

    with open(net_path+"state_dict.pth","rb") as f:
        if(model_type=="rnn"):
            net=RNN_EI(load=True, f=f)
        elif(model_type=="rcnn"):
            net=RCNN_EI(load=True, f=f)
        else:
            print("unknown model type:"+str(model_type))
            input()

    net = net.to(device)
    return net


def print_notes(notes, y_line, y_interv):
    if(notes!=""):
        if(isinstance(notes, str)):
            plt.annotate(notes, xy=(0.02, y_line), xycoords='axes fraction')
            y_line-=y_interv
        elif(isinstance(notes, list)):
            for note in notes:
                plt.annotate(notes, xy=(0.02, y_line), xycoords='axes fraction')
                y_line-=y_interv
        else:
            print("invalid notes type")

def plot_ratio(data, data2, logger, name="undefined", name2="undefined_sum", save_path=None):
    ensure_path(save_path)
    if(data2 is not None):
        if not os.path.exists(save_path):
            os.makepaths(save_path)
        if(isinstance(data, torch.Tensor) and isinstance(data_2, torch.Tensor)):
            pass
        else:
            print("both data must be torch.Tensor")
            return        

        data = data.numpy()
        data2 = data2.numpy()

        unit_num = data.shape[0]
        try:
            input_num = data.shape[1]
        except Exception:
            logger.write("%s includes 1-d Tensor:"%(name))
            return

        data_norm = np.array(list(map(lambda x: np.linalg.norm(x, ord=1), data[i].T)))
        data2_norm = np.array(list(map(lambda x: np.linalg.norm(x, ord=1), data2[i].T)))
        np.add(data_norm, data_2_norm)

        norm_ratio = np.arange(input_num)
        #print("unit_num=%d"%(unit_num))
        for i in range(input_num):
            try:
                norm_ratio[i] = data_norm[i]/data2_norm[i]
            except Exception:
                #print("data2_norm(%d)==0.0"%(i))
                norm_ratio[i]=0.0
        plot_dist(data=norm_ratio, logger=None, name=name+" norm ratio", save_path=save_path, bins=30, sci_float=False)
        '''
        ratio = np.arange(unit_num * input_num).reshape(unit_num, input_num)
        for i in range(unit_num):
            for j in range(input_num):
                try:
                    ratio[i][j]=data[i][j]/data_2[i][j]
                except Exception:
                    print("data_2(%d,%d)==0.0"%(i,j))
                    ratio[i][j]=0.0

        plot_dist(data=ratio, logger=logger, name=name+" ratio", save_path=save_path)
        '''
    else:
        unit_num = data[0].shape[0]
        try:
            input_num = data[0].shape[1]
        except Exception:
            logger.write("%s includes 1-d Tensor:"%(name))
            return
        data_num = len(data)
        data_norm = []
        for i in range(data_num):
            data_norm.append(np.array(list(map(lambda x: np.linalg.norm(x, ord=1), data[i].T)))) #(neuron_num)
        
        data_norm_tot = data_norm[0]
        for i in range(1, data_num):
            data_norm_tot = np.add(data_norm_tot, data_norm[i])
        
        data_ratio = [0.0 for i in range(input_num)]
        for i in range(data_num):
            try:
                data_ratio[i] = np.divide(data_norm[i], data_norm_tot)
            except Exception:
                for j in range(input_num):
                    try:
                        data_ratio[i][j] = data_norm[i][j] / data_norm_tot[j]
                    except Exception:
                        data_ratio[i][j]=0.0

        for i in range(data_num):
            try:
                plot_dist(data=data_ratio[i], logger=logger, name=name[i]+" ratio in "+name2, save_path=save_path, bins=30, sci_float=False)
            except Exception:
                logger.write("exception in dist_plot:"+str(name))
                traceback.print_exc()
                input()

def plot_dist(data, logger, name="undefined", ran=[], save_path=None, bins=100, kde=False, hist=True, notes="", redo=False, 
    rug_max=False, stat=True, cumulative=False, sci_float=True):
    if not os.path.exists(save_path):
        os.makepaths(save_path)
    if not isinstance(data, np.ndarray):
        data=np.array(data)
    data=data.flatten()
    data = np.sort(data)
    if(sci_float==True):
        stat_data="mean=%.2e var=%.2e, min=%.2e, max=%.2e mid=%.2e"%(np.mean(data), np.var(data), np.min(data), np.max(data), np.median(data))
    else:
        stat_data="mean=%.3f var=%.3f, min=%.3f, max=%.3f mid=%.3f"%(np.mean(data), np.var(data), np.min(data), np.max(data), np.median(data))
    
    if(data.shape[0]>sample_threshold):
        logger.write("data is too large. sample %d elements."%(sample_threshold))
        data=np.array(random.sample(data.tolist(), sample_threshold))
    img_path={}
    
    y_interv = 0.05
    y_line=0.95
    #hist_plot

    if(hist==True):
        sig=True
        sig_check=False
        while(sig==True):
            try:
                title=name+" distribution"
                if(cumulative==True):
                    title+="(cumu)"
                hist_path = save_path + title +".jpg"
                img_path["hist"]=hist_path
                if(redo==False and os.path.exists(hist_path)):
                    print("image already exists.")
                else:
                    plt.figure()
                    if(ran!=[]):
                        plt.xlim(ran[0],ran[1])
                    else:
                        set_lim(data, None, plt)
                    #sns.distplot(data, bins=bins, color='b',kde=False)
                    #method="sns"
                    plt.hist(data, bins=bins,color="b",density=True, cumulative=cumulative)
                    method="hist"

                    plt.title(title, fontsize=large_fontsize)
                    if(stat==True):
                        plt.annotate(stat_data, xy=(0.02, y_line), xycoords='axes fraction')
                        y_line-=y_interv
                    print_notes(notes, y_line, y_interv)
                    plt.savefig(hist_path)
                    plt.close()
                sig=False
            except Exception:
                if(sig_check==True):
                    raise Exception("exception in plot dist.")
                data, note = check_array_1d(data, logger)
                if(isinstance(notes, str)):
                    notes = [notes, note]
                elif(isinstance(notes, list)):
                    notes = notes + [note]
                sig_check=True
    try:
        #kde_plot
        y_line=0.95
        if(kde==True):
            title=name+" dist(kde)"
            if(cumulative==True):
                title+="(cumu)"
            kde_path=save_path+title+".jpg"
            img_path["kde"]=kde_path
            if(redo==False and os.path.exists(kde_path)):
                print("image already exists.")
            else:
                plt.figure()
                if(ran!=[]):
                    plt.xlim(ran[0],ran[1])
                else:
                    set_lim(data, None, plt)
                sns.kdeplot(data, shade=True, color='b', cumulative=cumulative)
                if(rug_max==True):
                    sns.rugplot(data[(data.shape[0]-data.shape[0]//500):-1], height=0.2, color='r')
                plt.title(title, fontsize=large_fontsize)
                if(stat==True):
                    plt.annotate(stat_data, xy=(0.02, y_line), xycoords='axes fraction')
                    y_line-=0.05
                print_notes(notes, y_line, y_interv)
                plt.savefig(kde_path)
                plt.close()
    except Exception:
        print("exception in kde plot %s"%(kde_path))
    return img_path

def plot_log_dist(data, logger, name="undefined", save_path=None, cumulative=False, hist=True, kde=False, bins=60):
    to_np_array(data)
    data=data.flatten()
    data_log=[]
    count_0=0
    count=data.shape[0]
    for dat in data:
        if(dat>0.0):
            data_log.append(math.log(dat, 10))
        elif(dat<0.0):
            print("%s is not a positive data"%(name))
            return False
        else:
            count_0+=1
    data_log=np.array(data_log)
    note="non-zero rate: %.3f(%.1e/%.1e)"%((count-count_0)/count, count-count_0, count)
    if(count_0==count):
        logger.write("%s is all-zero."%(name))
        return True
    plot_dist(data=data_log, logger=logger, name=name+"(log)", save_path=save_path, notes=note, cumulative=cumulative, hist=hist, kde=kde, bins=bins)
    return True


def set_lim(x, y=None, plt=""):
    if(y is None):
        x_min = np.min(x)
        x_max = np.max(x)
        xlim=[1.1*x_min-0.1*x_max, -0.1*x_min+1.1*x_max]
        plt.xlim(xlim[0], xlim[1])
        return xlim
    else:
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        xlim=[1.1*x_min-0.1*x_max, -0.1*x_min+1.1*x_max]
        ylim=[1.1*y_min-0.1*y_max, -0.1*y_min+1.1*y_max]
        try:
            plt.xlim(xlim[0], xlim[1])
        except Exception:
            print(Exception)
            print("exception when setting xlim.")
            #print("input anything to continue")
            #input()
        try:
            plt.ylim(ylim[0], ylim[1])
        except Exception:
            print(Exception)
            print("exception when setting ylim.")
            #print("input anything to continue")
            #input()
        return xlim, ylim

def combine_name(name):
    return name[0] + " - " + name[1]

def visualize_res_weight_corr(net, res_data, save_path):
    ensure_path(save_path)
    res = res_data["u"].detach().cpu().numpy()
    r_corr = cal_pearson_corr(data=res.T)
    print(res.shape)
    print(np.min(r_corr))
    print(np.max(r_corr))
    color_map = plt.cm.get_cmap('jet')
    r_corr = ( r_corr - np.min(r_corr) ) / ( np.max(r_corr) - np.min(r_corr) )
    r_corr = color_map(r_corr)
    #print(r_corr)
    r_corr = np.uint8(r_corr * 255)
    print(r_corr.shape)
    imsave(save_path + "/" + "N-N correlation.png", r_corr)

def visualize_weight(net, name="r", save_path=None):
    ensure_path(save_path)
    color_map = plt.cm.get_cmap('jet')
    w = net.get_weight(name, positive=True).detach().cpu()
    #print(get_data_stat(w))
    w = torch.squeeze(w)
    w = w.numpy()
    w = ( w - np.min(w) ) / (np.max(w) - np.min(w))
    w = color_map(w)
    w = np.uint8(w * 255)
    imsave(save_path + "/" + "N-N connection.png", w)    

def plot_rf(net, save_path=None, name="undefined", sample_num=16):#RF:(neuron_num, input_num)
    #print("plotting RF:%s"%(str(list(RF.size()))))
    ensure_path(save_path)
    rf = net.get_weight("i", detach=True, positive=True)
    if(input_type=="cifar10"):
        #CIFAR_10 input size: (batch_size, 3, 32, 32)
        rf=rf.view(-1, 3, 32, 32).cpu().numpy()
        rf = np.transpose(rf,(0, 2, 3, 1)) #(batch_size, 32, 32, 3)
        img_width=32
        channel_num=3
    elif(input_type=="mnist"):
        rf=rf.view(-1, 28, 28)
        img_width=28
        channel_num=1
    
    if(net.dict["separate_ei"]==True):
        ranges = [range(0, net.dict["E_num"]), range(net.dict["E_num"], net.dict["N_num"])]
        names = ["E", "I"]
    else:
        ranges = range(rf.shape[0])
        names = ["N"]
 
    count = 0
    for range_ in ranges:
        neuron_index = np.array(random.sample(range(rf.shape[0]), sample_num))
        ''
        imgs = []
        for index in neuron_index:
            imgs.append(rf[index])
        imgs = np.array(imgs)    
        imgs = ( ( imgs - np.min(imgs) ) / np.max(imgs) ) * 255
        imgs = np.uint8(imgs)        
        row = math.ceil(sample_num ** 0.5)
        col = sample_num//row + 1
        if(sample_num % row ==0):
            col -= 1

        imgs_cat = concat_images_in_rows(imgs, row_size=col, image_width=img_width, spacer_size=4)
        imsave(save_path + "/" + "rf_%s.png"%(names[count]), imgs_cat)
        count += 1

def cal_res_corr_pearson(net, res_data, net_dict): #res_data: [key](batch_size, unit_num)
    res_corr={}
    if net.dict["separate_ei"]:
        E_num = net.dict["E_num"]
        N_num = net.dict["N_num"]

        x_data = torch.cat([res_data["E.x"].t(), res_data["I.x"].t()], dim=0) #(N_num, sample_num)
        u_data = torch.cat([res_data["E.x"].t(), res_data["I.x"].t()], dim=0) #(N_num, sample_num)

        x_corr = cal_pearson_corr(data=x_data)
        u_corr = cal_pearson_corr(data=u_data)
        
        res_corr["E.x"] = x_corr[0:E_num, 0:E_num]
        res_corr["I.x"] = x_corr[E_num:N_num, E_num:N_num]
        res_corr["E-I.x"] = x_corr[0:E_num, E_num:N_num]

        res_corr["E.u"] = u_corr[0:E_num, 0:E_num]
        res_corr["I.u"] = u_corr[E_num:N_num, E_num:N_num]
        res_corr["E-I.u"] = u_corr[0:E_num, E_num:N_num]
        
    else:
        res_corr["N.x"] = cal_pearson_corr(data=res_data["N.x"].t())
        res_corr["N.u"] = cal_pearson_corr(data=res_data["N.u"].t())

    return res_corr

def cal_rf_corr_pearson(net, rf, net_dic):#rf:(input_num, neuron_num)
    rf_corr_data = cal_pearson_corr(data=rf.t())
    rf_corr={}
    if net.dict["separate_ei"]:
        E_num = net.N.dict["E_num"]
        N_num = net.N.dict["N_num"]
        rf_corr["E"] = rf_corr_data[0:E_num,0:E_num]
        rf_corr["I"] = rf_corr_data[E_num:N_num,E_num:N_num]
        rf_corr["E-I"] = rf_corr_data[0:E_num,E_num:N_num]             
        
    else:
        rf_corr["N"] = rf_corr_data
    return rf_corr

def plot_res(data, logger=None, name="undefined", save_path=None, is_act=True, sample_num=3):
    ensure_path(save_path)
    data = to_np_array(data)#(neuron_num, input_num)
    neuron_num = data.shape[0]
    try:
        input_num = data.shape[1]
    except Exception:
        logger.write("%s is 1-d Tensor:"%(name))
        input()

    if(neuron_num==0 or input_num==0):
        print("%s is empty data:(%d,%d)"%(name, neuron_num, input_num))
        return

    if(is_act==True):
        act_data = np.array(list(map(lambda r:r>0.0, data)))
        act_ratio = np.sum(act_data) / (input_num * neuron_num)
        neuron_act_ratio = np.sum(act_data, axis=1)/input_num #(neuron_num)
        input_act_ratio = np.sum(act_data, axis=0)/neuron_num
        plot_dist(data=act_ratio, logger=logger, name=name+" single neuron active ratio", hist=True, kde=False, save_path=save_path, bins=60)
        plot_dist(data=input_act_ratio, logger=logger, name=name+" neurons active ratio to single input", hist=True, kde=False, save_path=save_path, bins=60)
        plot_dist(data=neuron_act_ratio, logger=logger, name=name+" single neuron active ratio to inputs", hist=True, kde=False, save_path=save_path, bins=60, cumulative=True) 
        plot_log_dist(data=data.flatten(), logger=logger, name=name, hist=True, kde=False, save_path=save_path, bins=60)

    plot_dist(data=data.flatten(), logger=logger, name=name, hist=True, kde=False, save_path=save_path, bins=60)   

    #plot single neuron response to images
    for i in range(sample_num):
        unit_index=random.randint(0,neuron_num-1)
        plot_dist(data=data[unit_index], logger=logger, name="response of %s unit No.%d(%d)"%(name,unit_index,neuron_num-1), save_path=save_path + "single neuron responses/", bins=100)

    #plot neurons response to single image
    for i in range(sample_num):
        input_index=random.randint(0,input_num-1)
        plot_dist(data=data[:, input_index], logger=logger, name="%s response to input No.%d(%d)"%(name,input_index,input_num-1), save_path=save_path + "single input responses/", bins=100)

def plot_corr(x, y, name=["data0", "data1"], save_path=None, anal_stat=True, density_plot=False, log_plot=True, nonlog_plot=True):
    ensure_path(save_path)
    #print(name)
    #print(x)
    #print(y)
    to_np_array(x)
    to_np_array(y)

    corr=get_corr(x, y, method="all", save=False)
    corr_note=get_corr_note(corr)

    if nonlog_plot:
        fig = plt.figure()
        plt.xlabel(name[0], fontsize=large_fontsize)
        plt.ylabel(name[1], fontsize=large_fontsize)
        set_lim(x,y,plt)
        plt.scatter(x, y, c = 'b', marker = 'o')
        title=name[0]+" - "+name[1]+" correlation"
        plt.title(title, fontsize=large_fontsize)
        plt.annotate(corr_note, xy=(0.02, 0.95), xycoords='axes fraction')
        plt.savefig(save_path+title+".jpg")
        plt.cla()
        plt.close("all")

        if density_plot: #takes much time
            fig, ax = plt.subplots()
            df = pd.DataFrame(np.array([x,y]).transpose(1,0), columns=[name[0],name[1]])
            sns.jointplot(x=name[0], y=name[1], data=df, kind="kde");
            plt.savefig(save_path+title+"(density).jpg")
            plt.cla()
            plt.close("all")

    if anal_stat:
        fig = plt.figure()
        title=name[0]+" - "+name[1]+"(avg)"
        plt.title(title, fontsize=large_fontsize)
        plt.annotate(corr_note, xy=(0.05, 0.95), xycoords='axes fraction')
        bin_mean, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='mean',bins=100)
        plt.plot(bin_edge[1:],bin_mean,color='blue',linewidth=3.0,linestyle='-')
        plt.xlabel(name[0], fontsize=large_fontsize)
        plt.ylabel(name[1], fontsize=large_fontsize)
        plt.tight_layout()
        plt.savefig(save_path+title+".jpg")
        plt.close()

        fig, ax = plt.subplots()
        title=name[0]+" - "+name[1] +"(cumu)"+" & "+name[1]+"(num)"
        ax.set_title(title, fontsize=large_fontsize)
        bin_sum, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='sum',bins=100)
        bin_cumu=[]
        sum_cumu=0.0
        for sum_ in bin_sum:
            sum_cumu += sum_
            bin_cumu.append(sum_cumu)
        bin_cumu = np.array(bin_cumu)/sum_cumu
        ax.plot(bin_edge[1:], bin_cumu, color='b', linewidth=3.0, linestyle='-')
        ax.legend(["cumulative strength"])
        
        ax1 = ax.twinx()
        bin_count, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='count',bins=100)
        bin_cumu=[]
        count_cumu=0
        for count in bin_count:
            count_cumu += count
            bin_cumu.append(count_cumu)
        bin_cumu = np.array(bin_cumu)/count_cumu
        ax1.plot(bin_edge[1:],bin_cumu,color='r',linewidth=3.0,linestyle='-')
        ax1.legend(["cumulative pair num"])

        #plt.legend(["cumulative strength", "cumulative pair num"])
        ax.set_xlabel(name[0], fontsize=large_fontsize)
        ax.set_ylabel(name[1]+"(cumulative)(ratio)", fontsize=large_fontsize, color='b')
        ax1.set_ylabel("pair num(cumulative)(ratio)", fontsize=large_fontsize, color='r')
        plt.tight_layout()
        plt.savefig(save_path+title+".jpg")
        plt.close()

    if log_plot:
        #print(x.shape)
        #print(y.shape)
        #print(x)
        #print(y)
        xy, posi_rate = get_data_log(x, y, mode="y")
        #print(xy.shape)
        x = xy[:, 0]
        y = xy[:, 1]
        fig = plt.figure()
        plt.xlabel(name[0], fontsize=large_fontsize)
        plt.ylabel(name[1]+"(log)", fontsize=large_fontsize)
        set_lim(x, y, plt)
        plt.scatter(x, y, c = 'b', marker = 'o')
        plt.legend(["reciprocal connection"], loc="lower right")
        plt.annotate(corr_note, xy=(0.01, 0.95), xycoords='axes fraction')
        plt.annotate("nonzero rate:%.3f"%(posi_rate), xy=(0.01, 0.90), xycoords='axes fraction')
        title=name[0]+" - "+name[1]+"(log) correlation"
        plt.title(title, fontsize=large_fontsize)
        plt.savefig(save_path+title+"(log).jpg")
        plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
        plt.cla()
        plt.close("all")

        if density_plot: #takes much time
            fig, ax = plt.subplots()
            df = pd.DataFrame(xy, columns=name)
            sns.jointplot(x=name[0], y=name[1], data=df, kind="kde");
            plt.savefig(save_path+title+"(density)(log).jpg")
            plt.cla()
            plt.close("all")

        if anal_stat:
            fig = plt.figure()
            title=name[0]+" - "+name[1]+"(avg)"
            plt.title(title, fontsize=large_fontsize)
            plt.annotate(corr_note, xy=(0.05, 0.95), xycoords='axes fraction')
            bin_mean, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='mean',bins=100)
            plt.plot(bin_edge[1:],bin_mean,color='blue',linewidth=3.0,linestyle='-')
            plt.xlabel(name[0], fontsize=large_fontsize)
            plt.ylabel(name[1], fontsize=large_fontsize)
            plt.tight_layout()
            plt.savefig(save_path+title+".jpg")
            plt.close()

            fig, ax = plt.subplots()
            title=name[0]+" - "+name[1] +"(cumu)"+" & "+name[1]+"(num)"
            ax.set_title(title, fontsize=large_fontsize)
            bin_sum, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='sum',bins=100)
            bin_cumu=[]
            sum_cumu=0.0
            for sum_ in bin_sum:
                sum_cumu += sum_
                bin_cumu.append(sum_cumu)
            bin_cumu = np.array(bin_cumu)/sum_cumu
            ax.plot(bin_edge[1:], bin_cumu, color='b', linewidth=3.0, linestyle='-')
            ax.legend(["cumulative strength"])
            
            ax1 = ax.twinx()
            bin_count, bin_edge, bin_num=scipy.stats.binned_statistic(x=x,values=y,statistic='count',bins=100)
            bin_cumu=[]
            count_cumu=0
            for count in bin_count:
                count_cumu += count
                bin_cumu.append(count_cumu)
            bin_cumu = np.array(bin_cumu)/count_cumu
            ax1.plot(bin_edge[1:],bin_cumu,color='r',linewidth=3.0,linestyle='-')
            ax1.legend(["cumulative pair num"])

            #plt.legend(["cumulative strength", "cumulative pair num"])
            ax.set_xlabel(name[0], fontsize=large_fontsize)
            ax.set_ylabel(name[1]+"(cumulative)(ratio)", fontsize=large_fontsize, color='b')
            ax1.set_ylabel("pair num(cumulative)(ratio)", fontsize=large_fontsize, color='r')
            plt.tight_layout()
            plt.savefig(save_path+title+"(log).jpg")
            plt.close()

def anal_stability(net, iter_data, save_path=None, input_sample_num=5, neuron_sample_num=20):
    ensure_path(save_path)
    if net.dict["separate_ei"]:
        keys = ["E.x", "E.u", "I.x", "I.u"]
        iter_time = iter_data["E.x"].size(1)
    else:
        keys = ["N.x", "N.u"]
        iter_time = iter_data["N.x"].size(1)
    x_ = range(iter_time)
    E_num = net.dict["E_num"]
    I_num = net.dict["I_num"]
    N_num = net.dict["N_num"]
    for key in keys:
        data = iter_data[key]
        ensure_path(save_path + key + "/")
        if("E." in key):
            neuron_range = range(0, E_num)
        elif("I." in key):
            neuron_range = range(0, I_num)
        elif("N." in key):
            neuron_range = range(data.size(2))
        else:
            print("invalid key:" + key)
            input()
        for input_index in random.sample(range(data.size(0)), input_sample_num):
            fig = plt.figure()
            plt.ylabel("response", fontsize=large_fontsize)
            plt.xlabel("iteration time", fontsize=large_fontsize)
            for neuron_index in random.sample(neuron_range, neuron_sample_num):
                plt.plot(x_, torch.squeeze(data[input_index, :, neuron_index]).detach().cpu().numpy(), linewidth=3.0, linestyle="-")
            plt.savefig(save_path + key + "/" + "sampled %d neuron responses to image No.%d"%(neuron_sample_num, input_index) + ".jpg")
            plt.close()
def check_NaN(data, logger):
    if(len(data.shape)==1):
        x_num = data.shape[0]
        count=0
        for i in range(x_num):
            if(np.isnan(data[i]) == True):
                print("data(%d) is NaN."%(i), end=' ')
                data[i]=0.0
                count+=1
            elif(np.isinf(data[i]) == True):
                print("data(%d) is Inf."%(i), end=' ')
                data[i]=0.0
                count+=1
        note = "%.3f(%d/%d) elements in data is NaN or Inf."%(count/(x_num), count, x_num)
        logger.write(note)
        return data, note
    elif(len(data.shape)==2):
        x_num = data.shape[0]
        y_num = data.shape[1]
        count=0
        for i in range(x_num):
            for j in range(y_num):
                if(np.isnan(data[i][j]) == True):
                    print("data(%d,%d) is NaN."%(i,j), end=' ')
                    data[i][j]=0.0
                    count+=1
                elif(np.isinf(data[i][j]) == True):
                    print("data(%d,%d) is Inf."%(i,j), end=' ')
                    data[i][j]=0.0
                    count+=1

        note = "%.43(%d/%d) elements in data is NaN or Inf."%(count/(x_num*y_num), count, x_num*y_num)
        logger.write(note)
        return data, note

def get_corr_note(corr):
    note="corr"
    if(corr.get("pearson") is not None):
        note+=" pearson:%.3f"%(corr["pearson"])
    if(corr.get("kendall") is not None):
        note+=" kendall:%.3f"%(corr["kendall"])
    if(corr.get("spearman") is not None):
        note+=" spearman:%.3f"%(corr["spearman"])
    return note

def set_corr_method(method):
    if(isinstance(method, str)):
        if(method=="all"):
            method=["pearson", "kendall", "spearman"]
        else:
            method=[method]
        return method
    elif(isinstance(method, list)):
        return method
    else:
        print("invalid method type")
        return None

def to_np_array(data):
    if(isinstance(data, list)):
        data=np.array(data)
    if(isinstance(data, torch.Tensor)):
        data=data.numpy()
    return data

def get_flattened_array(data, remove_diagonal=False, only_diagonal=False):
    if(data.shape[0] != data.shape[1]):
        return data.flatten()
    if not remove_diagonal:
        if(only_diagonal==False):
            return data.flatten()
        else:
            data1 = []
            unit_num = data.shape[0]
            for i in range(unit_num):
                data1.append(data[i][i])
            return np.array(data1).flatten()
    else:
        data1 = []
        unit_num = data.shape[0]
        for i in range(unit_num):
            for j in range(unit_num):
                if(i != j):
                    data1.append(data[i][j])
        return np.array(data1).flatten()

def get_corr(x, y, method="all", save=False, name="undefined", save_path=None):
    corr={}
    method=set_corr_method(method)

    for m in method:
        corr[m]=pd.Series(x).corr(pd.Series(y), method=m)

    if(len(method)==1):
        corr = corr[method[0]]
    if(save==True):
        f=open(save_path + name, "wb")
        pickle.dump(corr, f)
        f.close()
    return corr

def cal_pearson_corr(data, data_2=None, name="unnamed_corr", method="pearson", save=False, save_path=None):
    #calculate pearson correlation indices between any 2 of M variables. Each variable has N observed values.
    #data: observed values, in shape of (variable_num, observed_value_num) or (neuron_num, input_data_num).
    #pearson correlation compuation can be computed on gpu, thus faster than the general function "cal_corr".
    if(isinstance(data, np.ndarray)):
        data = torch.from_numpy(data)
    data = data.to(device)
    if(data_2 is not None):
        if(isinstance(data_2, np.ndarray)):
            data_2 = torch.from_numpy(data_2)
        data_2 = data_2.to(device)      
    sample_num=data.size(1)
    if(data_2 is None): #self-correlation
        x = data.t()
        #print(len(x))
        #print(list(x[0].size()))
        act_var = torch.var(x, dim=0).detach().cpu().numpy() #(neuron_num)
        act_var = act_var * (sample_num - 1)
        act_mean = torch.mean(x, dim=0).detach().cpu().numpy() #(neuron_num)
        #convert from tensor and numpy prevents including the process of computing var and mean into the computation graph.
        act_std = torch.from_numpy(act_var).to(device) ** 0.5
        act_mean = torch.from_numpy(act_mean).to(device)
        std_divider = torch.mm(torch.unsqueeze(act_std, 1), torch.unsqueeze(act_std, 0)) #(neuron_num, neuron_num)
        x_normed = (x - act_mean) #broadcast
        act_dot = torch.mm(x_normed.t(), x_normed)
        try:
            act_corr = act_dot / std_divider
        except Exception:
            abnormal_coords = []
            for i in range(list(act_std.size())[0]):
                for j in range(list(act_std.size()[0])):
                    if(std_divider[i][j] == 0.0):
                        print("act_std[%d]][%d]==0.0"%(i, j))
                        std_divider[i][j] = 1.0
                        abnormal_coords.append([i,j])
            act_corr = act_dot / std_divider
            for coord in abnormal_coords:
                act_corr[coord[0]][coord[1]] = 0.0
        corr = act_corr.detach().cpu().numpy()
    else:
        x = data.t()
        #print(len(x))
        #print(list(x[0].size()))
        x_var = torch.var(x, dim=0).detach().cpu().numpy() #(neuron_num)
        x_var = x_var * (sample_num - 1)
        x_mean = torch.mean(x, dim=0)
        x_std = torch.from_numpy(x_var).to(device) ** 0.5
        x_normed = (x - x_mean) #broadcast
        y = data_2.t()
        #print(len(x))
        #print(list(x[0].size()))
        y_var = torch.var(y, dim=0).detach().cpu().numpy() #(neuron_num)
        y_var = y_var * (sample_num - 1)
        y_mean = torch.mean(y, dim=0)
        y_std = torch.from_numpy(y_var).to(device) ** 0.5
        y_normed = (y - y_mean) #broadcast

        std_divider = torch.mm(torch.unsqueeze(x_std, 1), torch.unsqueeze(y_std, 0)) #(neuron_num_1, neuron_num_2)

        act_dot = torch.mm(x_normed.t(), y_normed)
        try:
            act_corr = act_dot / std_divider
        except Exception:
            abnormal_coords = []
            for i in range(list(x_std.size())[0]):
                for j in range(list(y_std.size())[0]):
                    if(std_divider[i][j] == 0.0):
                        print("act_std[%d]][%d]==0.0"%(i, j))
                        std_divider[i][j] = 1.0
                        abnormal_coords.append([i,j])
            act_corr = act_dot / std_divider
            for coord in abnormal_coords:
                act_corr[coord[0]][coord[1]] = 0.0
        corr = act_corr.detach().cpu().numpy()
    return corr

def cal_corr(data, data_2=None, name="unnamed_corr", method="pearson", save=False, save_path=None):#data:(unit_num, feature_vector_size)
    unit_num = data.size(0)
    print("calculating %s corr method="%(name)+str(method))
    method=set_corr_method(method)
    if(method==None):
        return None
    corr={}
    count_m=0
    if(data_2 is None):#self-correlation
        for m in method:
            count_m+=1
            corr[m]=[]
            for i in range(unit_num):
                print("\r","progress: unit_num:%d/%d method:%d/%d"%(i+1, unit_num, count_m, len(method)), end="", flush=True)
                corr_0=[]
                for j in range(unit_num):
                    corr_0.append(pd.Series(data[i]).corr(pd.Series(data[j]), method=m))
                    if(np.isnan(corr_0[-1])):
                        print("corr(%d,%d) is NaN."%(i,j), end=' ')
                corr[m].append(corr_0)
            print("\n")
    else:#mutual correlation
        unit_num_2 = data_2.size(0)
        for m in method:
            count_m+=1
            corr[m]=[]
            for i in range(unit_num):
                print("\r","progress: unit_num:%d/%d method:%d/%d"%(i+1, unit_num, count_m, len(method)), end="", flush=True)
                corr_0=[]
                for j in range(unit_num_2):
                    corr_0.append(pd.Series(data[i]).corr(pd.Series(data_2[j]),method=m))
                corr[m].append(corr_0)
            print("\n")
    if(save==True):
        f=open(save_path + name, "wb")
        pickle.dump(corr, f)
        f.close()
    return corr

def plot_weight_stat(weight, logger, name="undefined", save_path=None):
    ensure_path(save_path)

    is_positive = plot_log_dist(weight, logger, name=name, save_path=save_path)
    plot_log_dist(weight, logger, name=name, save_path=save_path, cumulative=True)

    weight_flat=weight.flatten()
    weight_flat=np.sort(weight_flat) #small -> large
    weight_num=len(weight_flat)
    if(is_positive):
        #calculate ratio of weights that takes 50% of total weight
        weight_cumu = []
        weight_sum = 0.0
        for w in weight_flat:
            weight_sum += w
            weight_cumu.append(weight_sum) 
        count = 0
        sig=False
        #print(weight_sum)
        for w in weight_cumu:
            count += 1
            #print(w)
            if w>(weight_sum * 0.5):
                note0="%.4f(%d/%d) of weights takes 50%% of total strength"%(count/weight_num, count,weight_num)
                sig=True
                break
        if(sig==False):
            note0="Unable to determine percentage of weight that takes 50% of total strength."
    else:
        note0="%s is not a positive weight"%(name)
    plot_dist(data=weight_flat, logger=logger, name=name, save_path=save_path, notes=note0)
    plot_dist(data=weight_flat, logger=logger, name=name, save_path=save_path, cumulative=True, notes=note0)
    
    x = list(map(lambda n:n/weight_num, range(weight_num)))
    plt.plot(x, weight_flat, color='b', linewidth=3.0, linestyle='-')
    plt.xlabel("weight_num", fontsize=large_fontsize)
    plt.ylabel("weight", fontsize=large_fontsize)
    title=name+" weight(ascent)"
    plt.title(title, fontsize=large_fontsize)
    plt.savefig(save_path+title+".jpg")
    plt.close()

    if(is_positive):
        weight_ratio = list(map(lambda w:w/weight_sum, weight_cumu))
        plt.plot(x, weight_ratio, color='b', linewidth=3.0, linestyle='-')
        plt.xlabel("weight_num", fontsize=large_fontsize)
        plt.ylabel("weight_ratio", fontsize=large_fontsize)
        title=name+" weight(ascent)(cumulative)"
        plt.title(title, fontsize=large_fontsize)
        plt.annotate(note0, xy=(0.02, 0.95), xycoords='axes fraction')
        plt.savefig(save_path+title+".jpg")
        plt.close()
    
def plot_weight_rec(data, name="undefined", save_path=None, density_plot=False, log_plot=True):
    ensure_path(save_path)
    unit_num = data.shape[0]
    data_num = unit_num ** 2
    pair_num = unit_num * (unit_num-1) // 2
    #connect rate
    zero_count=0
    for w in data.flatten():
        if w==0.0 or w==-0.0:
            zero_count=zero_count+1
    connect_num=data_num - zero_count
    logger.write("analyzing %s weight"%(name))
    note0="connect_rate:%.4f (%.1e/%.1e)"%(connect_num/data_num, connect_num, data_num)

    #no_connect, lat_connect, bi_connect
    no_connect = 0
    lat_connect = 0
    bi_connect = 0
    
    for i in range(unit_num):
        for j in range(i+1, unit_num):
            if(data[i][j]>0.0):
                if(data[j][i]>0.0):
                    bi_connect = bi_connect + 1
                else:
                    lat_connect = lat_connect + 1
            else:
                if(data[j][i]>0.0):
                    lat_connect = lat_connect + 1
                else:
                    no_connect = no_connect + 1
    note1="no:%.3f(%.1e/%.1e)"%(no_connect/pair_num, no_connect, pair_num)
    note2="lat:%.3f(%.1e/%.1e)"%(lat_connect/pair_num, lat_connect, pair_num)
    note3="bi:%.3f(%.1e/%.1e)"%(bi_connect/pair_num, bi_connect, pair_num)

    #self_connect
    self_connect=0
    for i in range(unit_num):
        if(data[i][i]>0.0):
            self_connect=self_connect+1
    note4 = "self_connect:%.3f(%.1e/%.1e)"%(self_connect/unit_num, self_connect, unit_num)
    connect_note = note0 + " " + note1 + " " + note2 + " " + note3 + " " + note4
    logger.write(connect_note)

    #reciprocal connection
    co_pair=[]
    self_pair=[]

    for i in range(unit_num):
        for j in range(i+1, unit_num):
            co_pair.append([data[i][j],data[j][i]])     
        self_pair.append([data[i][i],data[i][i]])

    co_pair=np.array(co_pair)
    self_pair=np.array(self_pair)
    
    corr_note=get_corr_note(get_corr(co_pair[:,0], co_pair[:,1],save=False))
    #plot reciprocal weight
    fig = plt.figure()
    plt.xlabel('connection strength', fontsize=large_fontsize)
    plt.ylabel('connection strength', fontsize=large_fontsize)
    set_lim(data, data,plt)
    plt.scatter(self_pair[:,0], self_pair[:,1], c = 'r', marker = 'o')
    plt.scatter(co_pair[:,0], co_pair[:,1], c = 'b', marker = 'o')
    plt.legend(["self connection","reciprocal connection"], loc="lower right")
    plt.annotate(connect_note, xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(corr_note, xy=(0.02, 0.90), xycoords='axes fraction')
    title=name+" reciprocal connect distribution"
    plt.title(title, fontsize=large_fontsize)
    plt.savefig(save_path+title+".jpg")
    plt.close()

    if(density_plot): #takes much time
        fig, ax = plt.subplots()
        df = pd.DataFrame(co_pair, columns=[name+"0", name+"1"])
        sns.jointplot(x=name+"0", y=name+"1", data=df, kind="kde");
        plt.savefig(save_path+title+"(density).jpg")
        plt.close()

    if log_plot:
        names = [name+"0(log)", name+"1(log)"]
        xy, posi_rate = get_data_log(co_pair[:, 0], co_pair[:, 1], mode="xy")
        x = xy[:, 0]
        y = xy[:, 1]
        fig = plt.figure()
        plt.xlabel(names[0], fontsize=large_fontsize)
        plt.ylabel(names[1], fontsize=large_fontsize)
        set_lim(x, y, plt)
        plt.scatter(x, y, c = 'b', marker = 'o')
        plt.legend(["reciprocal connection"], loc="lower right")
        plt.annotate(connect_note, xy=(0.01, 0.95), xycoords='axes fraction')
        plt.annotate(corr_note, xy=(0.01, 0.90), xycoords='axes fraction')
        plt.annotate("nonzero rate:%.3f"%(posi_rate), xy=(0.01, 0.85), xycoords='axes fraction')
        title=name+" reciprocal connect distribution"
        plt.title(title, fontsize=large_fontsize)
        plt.savefig(save_path+title+"(log).jpg")
        plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
        plt.close()

        if density_plot: #takes much time
            fig, ax = plt.subplots()
            df = pd.DataFrame(xy, columns=names)
            sns.jointplot(x=names[0], y=names[1], data=df, kind="kde");
            plt.savefig(save_path+title+"(density)(log).jpg")
            plt.close()   

def plot_weight_lat(data, data2, name="undefined", save_path=None, density_plot=False, log_plot=True):
    ensure_path(save_path)
    if(list(data.size())!=list(data2.size())):
        print("weight:%s and weight2:%s must be of same size."%(str(list(data.size())), str(list(data2.size()))))
        return
    unit_num = data.shape[0]
    unit_num_2 = data.shape[1]
    data_num = 2 * unit_num * unit_num_2
    pair_num = unit_num * unit_num_2

    #connect rate
    zero_count=0
    for weight in data.flatten():
        if weight==0.0 or weight==-0.0:
            zero_count=zero_count+1
    connect_num = data_num - zero_count
    note0="connect_rate:%.3f(%.1e/%.1e)"%(connect_num/data_num, connect_num, data_num)

    no_connect = 0
    lat_connect = 0
    bi_connect = 0
    for i in range(unit_num):
        for j in range(i+1, unit_num_2):
            if(data[i][j]>0.0):
                if(data2[i][j]>0.0):
                    bi_connect = bi_connect + 1
                else:
                    lat_connect = lat_connect + 1
            else:
                if(data2[i][j]>0.0):
                    lat_connect = lat_connect + 1
                else:
                    no_connect = no_connect + 1
    note1="no:%.3f(%.1e/%.1e)"%(no_connect/pair_num, no_connect, pair_num)
    note2="lat:%.3f(%.1e/%.1e)"%(lat_connect/pair_num, lat_connect, pair_num)
    note3="bi:%.3f(%.1e/%.1e)"%(bi_connect/pair_num, bi_connect, pair_num)
    connect_note = note0 + " " + note1 + " " + note2 + " " + note3

    x=data.detach().cpu().numpy().flatten()
    y=data2.detach().cpu().numpy().flatten()

    corr = get_corr(x,y,method="all", save=False)
    corr_note = get_corr_note(corr)

    names = ["E->I strength", "I->E strength"]
    fig = plt.figure()
    plt.xlabel(names[0], fontsize=large_fontsize)
    plt.ylabel(names[1], fontsize=large_fontsize)
    set_lim(x, y, plt)
    plt.scatter(x, y, c = 'b', marker = 'o')
    plt.legend(["reciprocal connection"], loc="lower right")
    plt.annotate(connect_note, xy=(0.01, 0.95), xycoords='axes fraction')
    plt.annotate(corr_note, xy=(0.01, 0.90), xycoords='axes fraction')
    title=name+" reciprocal connect distribution"
    plt.title(title, fontsize=large_fontsize)
    plt.savefig(save_path+title+".jpg")
    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.close()

    if density_plot: #takes much time
        fig, ax = plt.subplots()
        df = pd.DataFrame(np.array([x,y]).transpose(1,0), columns=names)
        sns.jointplot(x=names[0], y=names[1], data=df, kind="kde");
        plt.savefig(save_path+title+"(density).jpg")
        plt.close()

    if log_plot:
        names = ["E->I strength(log)", "I->E strength(log)"]
        xy, posi_rate = get_data_log(x, y, mode="xy")
        x = xy[:, 0]
        y = xy[:, 1]
        fig = plt.figure()
        plt.xlabel(names[0], fontsize=large_fontsize)
        plt.ylabel(names[1], fontsize=large_fontsize)
        set_lim(x, y, plt)
        plt.scatter(x, y, c = 'b', marker = 'o')
        plt.legend(["reciprocal connection"], loc="lower right")
        plt.annotate(connect_note, xy=(0.01, 0.95), xycoords='axes fraction')
        plt.annotate(corr_note, xy=(0.01, 0.90), xycoords='axes fraction')
        plt.annotate("nonzero rate:%.3f"%(posi_rate), xy=(0.01, 0.85), xycoords='axes fraction')
        title=name+" reciprocal connect distribution"
        plt.title(title, fontsize=large_fontsize)
        plt.savefig(save_path+title+"(log).jpg")
        plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
        plt.close()

        if density_plot: #takes much time
            fig, ax = plt.subplots()
            df = pd.DataFrame(xy, columns=names)
            sns.jointplot(x=names[0], y=names[1], data=df, kind="kde");
            plt.savefig(save_path+title+"(density)(log).jpg")
            plt.close()   
    

def get_data_log(x, y, mode="x"):
    points = np.array([x,y]).transpose(1,0)
    points_log = []
    
    count = 0
    if(mode=="x"):
        for point in points:
            if(point[0]>0.0):
                points_log.append( [ math.log(point[0], 10), point[1] ])
            elif(point[0]==0.0):
                count += 1
            else:
                print("data is not non-negative.")
                input()
                count += 1
    elif(mode=="y"):
        for point in points:
            if(point[1]>0.0):
                points_log.append( [ point[0], math.log(point[1], 10) ] )
            elif(point[1]==0.0):
                count += 1
            else:
                print("data is not non-negative.")
                input()
                count += 1
    elif(mode=="xy"):
        for point in points:
            if(point[0]>0.0 and point[1]>0.0):
                points_log.append( [ math.log(point[0], 10), math.log(point[1], 10) ] )
            elif(point[0]==0.0):
                count += 1
            else:
                print("data is not non-negative.")
                input()
                count += 1
    else:
        print("invalid mode:"+str(mode))

    xy = np.array(points_log)
    return xy, 1.0 - ( count / points.shape[0] )

def add_bracket(name):
    return "("+name+")"


def read_data(read_path):
    if not os.path.exists(read_path):
        return None
    f = open(read_path, "rb")
    data = torch.load(f)
    f.close()
    return data

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    #print(images[0])
    #print("aaa")
    spacer = np.ones([image_width, spacer_size, 3], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    
    #print(images_with_spacers[0])
    #print(images_with_spacers[1])
    ret = np.hstack(images_with_spacers)
    #print(ret)
    #input()
    return ret

def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 3],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    #print(images[0])
    #print("bbb")
    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def create_gif(image_list, gif_name, duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间, 单位s
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def get_data_stat(data):
    return "mean=%.2e var=%.2e, min=%.2e, max=%.2e mid=%.2e"%(np.mean(data), np.var(data), np.min(data), np.max(data), np.median(data))

if __name__ == '__main__':
    main()
