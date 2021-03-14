import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

#training parameters.
#from param_config import *
from utils import get_name, get_args
from utils_model_ import *

def get_cons_func(func_str):
    if func_str in ['abs']:
        return lambda x: torch.abs(x)
    elif func_str in ['square', '^2', '2']:
        return lambda x: x * x
    else:
        raise Exception('Invalid cons_func str: %s'%func_str)

def get_act_func(params):
    name = get_name(params)
    coeff = get_args(params)
    if name=='none':
        return lambda x:x
    elif name in ['relu', 'ReLU']:
        return lambda x:coeff * F.relu(x)
    elif name in ['tanh']:
        return lambda x:coeff * torch.tanh(x)
    elif name=='relu_tanh':
        return lambda x:coeff * F.relu(torch.tanh(x))
    elif name=='relu+tanh':
        return lambda x:coeff * F.relu(x) + (1.0 - coeff) * F.relu(torch.tanh(x))
    elif name=='e_1':
        net.act_func=lambda x:F.relu(x)
    elif name=='i_1':
        net.act_func=lambda x:(F.relu(x))**0.6
    elif name=='e_2':
        net.act_func=lambda x:(F.relu(x))**0.9
    elif name=='i_2':
        net.act_func=lambda x:5*(F.relu(x))**0.6
    elif name=='e_3':
        if param=='': coeff = 0.1
        else: coeff = param
        net.act_func = lambda x:coeff*(F.relu(x)) + (1.0-coeff)*(F.relu(torch.tanh(x)))
    elif(name=='i_3'):
        if(param==''):
            coeff = 0.2
        else:
            coeff = param
        net.act_func = lambda x:coeff*(F.relu(x)) + (1.0-coeff)*(F.relu(torch.tanh(x)))

    #old e4_i4 == e3(0.1, thres=1.0)_i3(0.2)
    elif(name=='e_4'):
        if(param==''):
            coeff = 0.2
        else:
            coeff = param
        net.act_func=lambda x:coeff*(F.relu(x))
    elif(name=='i_4'):
        if(param==''):
            coeff = 0.4
        else:
            coeff = param
        net.act_func=lambda x:coeff*(F.relu(x))

def set_act_func(net, act_func_str='relu'):
    if(isinstance(act_func_str, str)):
        set_act_func_from_name(net, act_func_str)
    elif(isinstance(act_func_str, dict)):
        if(net.dict.get('type')!=None and act_func_str.get(net.type)!=None):
            set_act_func_from_name(net, act_func_str[net.type])
        else:
            set_act_func_from_name(net, act_func_str['default'])

def cat_dict(dict_0, dict_1, dim_unsqueeze=None, dim=0):
    for key in dict_1.keys():
        if(len(dict_1[key])>0):
            if dim_unsqueeze is not None:
                dict_1[key] = list(map(lambda x:torch.unsqueeze(x, dim), dict_1[key]))
            dict_1[key] = torch.cat(dict_1[key], dim=dim) #(iter_time, batch_size, output_num)
            if(dict_0.get(key) is None):
                dict_0[key] = dict_1[key].detach().cpu()
            else:
                #print(dict_0[key].size())
                #print(dict_1[key].size())
                dict_0[key] = torch.cat([dict_0[key], dict_1[key].detach().cpu()], dim=dim)
            dict_1[key] = []

def get_ei_mask(E_num, N_num, kernel_size=None, device=None):
    if device is None:
        device = torch.device('cpu')
    ei_mask = torch.zeros((N_num, N_num), device=device, requires_grad=False)
    for i in range(0, E_num):
        ei_mask[i][i] = 1.0
    for i in range(E_num, N_num):
        ei_mask[i][i] = -1.0
    return ei_mask

def get_mask(N_num, output_num, device=None):
    if device is None:
        device = torch.device('cpu')
    mask = torch.ones((N_num, output_num), device=device, requires_grad=False)
    return mask

def get_mask_from_tuple(tuple):
    mask = torch.ones(tuple, device=device, requires_grad=False)
    return mask

def init_weight(weight, params, cons_method='abs'):
    name = get_name(params)
    coeff = get_args(params)
    if name=='output':
        divider = weight.size(1)
    elif name=='input':
        divider = weight.size(0)

    lim = coeff / divider
    #print('coeff=%.4e'%(coeff))
    #print('lim=%4e'%(lim))
    if cons_method=='force':
        torch.nn.init.uniform_(weight, 0.0, 2 * lim)
    else:
        torch.nn.init.uniform_(weight, -lim, lim)  

def save_dict(net, dict=None):
    return net.dict
def load_dict(net, f):
    net.dict=pickle.load(f)

def train(net, epochs, trainloader, testloader, train_loss_list_0=[], train_acc_list_0=[], val_loss_list_0=[], val_acc_list_0=[], save_dir='undefined', save=True, evaluate_before_train=True, save_interval=20, evaluate=None, logger=None, mode_name='model'):
    train_loss_list = [0.0 for _ in range(epoch_num)]
    train_acc_list = [0.0 for _ in range(epoch_num)]
    val_loss_list = [0.0 for _ in range(epoch_num)]
    val_acc_list = [0.0 for _ in range(epoch_num)]

    if(evaluate is None):
        evaluate = evaluate_iter

    if(isinstance(epochs, list)):
        epoch_start=epochs[0]
        epoch_end=epochs[1]
    elif(isinstance(epochs, int)):
        epoch_start=0
        epoch_end=epochs
    else:
        print('invalid epochs type.')

    if(evaluate_before_train==True):
        with torch.no_grad():
            val_loss, val_acc = evaluate(net, testloader, criterion, scheduler, augment, device)
            note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
            val_loss_list[epoch]=val_loss
            val_acc_list[epoch]=val_acc
            if(save==True and epoch%save_interval==0):
                net_path=save_dir_stat+model_name+'_epoch_%d_0/'%(epoch_start)
                if not os.path.exists(net_path):
                    os.makedirs(net_path)
                #torch.save(net.state_dict(), net_path + 'torch_save.pth')
                net.save(net_path)
    
    for epoch in range(epoch_start, epoch_end+1):
        note0 = 'epoch=%d'%(epoch)

        loss_total = 0.0
        labels_count=0
        correct_count=0
        count=0
        for i, data in enumerate(trainloader, 0):
            #print('\r','progress:%d/50000 '%(labels_count), end='', flush=True)
            count=count+1
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = list(map(lambda x:x.to(device), outputs))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct_count += (torch.max(outputs[-1], 1)[1]==labels).sum().item()
            labels_count += labels.size(0)
            loss_total += loss.item()

        train_loss_list[epoch]=train_loss
        train_acc_list[epoch]=train_acc

        note1='train_loss:%.4f train_acc:%.4f'%(train_loss, train_acc)

        with torch.no_grad():
            val_loss, val_acc = evaluate(net,testloader,criterion,scheduler,augment, device)
            note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
            val_loss_list[epoch]=val_loss
            val_acc_list[epoch]=val_acc
            if(save==True and epoch%save_interval==0):
                net_path=save_dir_stat+model_name+'_epoch_%d/'%(epoch)
                if not os.path.exists(net_path):
                    os.makedirs(net_path)
                #torch.save(net.state_dict(), net_path + 'torch_save.pth')
                net.save(net_path)

        logger.write(note0+note1+note2)

    with torch.no_grad():
        val_loss, val_acc = evaluate(net,testloader,criterion,scheduler,augment, device)
        note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
        val_loss_list[epoch]=val_loss
        val_acc_list[epoch]=val_acc
        if(save==True and epoch%save_interval==0):
            net_path=save_dir_stat+model_name+'_epoch_%d/'%(epoch_end)
            if not os.path.exists(net_path):
                os.makedirs(net_path)
            #torch.save(net.state_dict(), net_path + 'torch_save.pth')
            net.save(net_path)

    train_loss_list_0 = train_acc_list_0 + train_loss_list
    train_acc_list_0 = train_acc_list_0 + train_acc_list
    val_loss_list_0 = val_loss_list_0 + val_loss_list
    val_acc_list_0 = val_acc_list_0 + val_acc_list

def test():
    print('MyLib test.')
def pytorch_info():
    if(torch.cuda.is_available()==True):
        print('Cuda is available')
    else:
        print('Cuda is unavailable')

    print('Torch version is '+torch.__version__)
'''
def prepare_CIFAR10(dataset_dir=CIFAR10_dir, norm=True, augment=False, batch_size=64):
    if(augment==True):
        feature_map_width=24
    else:
        feature_map_width=32

    if(augment==True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.TenCrop(24),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform_train, download=False)
    testset = torchvision.datasets.CIFAR10(root=dataset_dir,train=False, transform=transform_test, download=False)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader
'''
def evaluate(net, testloader, criterion, scheduler, augment, device):
    net.eval()
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    loss_total=0.0
    #torch.cuda.empty_cache()
    for data in testloader:
        #print('\r','progress:%d/%d '%(count,len(testloader)), end='', flush=True)
        count=count+1
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        if augment:
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs = net(inputs) 
            outputs = outputs.to(device)
        loss_total += criterion(outputs, labels).item()
        correct_count+=(torch.max(outputs, 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
    #print('\n')
    val_loss=loss_total/count
    val_acc=correct_count/labels_count
    net.train()
    return val_loss, val_acc

def evaluate_iter(net, testloader, criterion, scheduler, augment, device):
    net.eval()
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    loss_total=0.0
    #torch.cuda.empty_cache()
    for data in testloader:
        #print('\r','progress:%d/%d '%(count,len(testloader)), end='', flush=True)
        count=count+1
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        if augment:
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs, act = net(inputs)
            #outputs = list(map(lambda x:x.to(device), outputs))  
        loss_total += net.get_loss(inputs, labels).item()
        correct_count+=(torch.max(outputs[-1], 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
    #print('\n')
    val_loss=loss_total/count
    val_acc=correct_count/labels_count
    net.train()
    return val_loss, val_acc

'''
class model(nn.Module):
    def __init__(self):
        super(net_model, self).__init__()
        self.w1 = torch.nn.Parameter(1e-3*torch.rand(784, hidden_layer_size, device=device))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_layer_size, device=device))
        self.r1 = torch.nn.Parameter(1e-3*torch.rand(hidden_layer_size, hidden_layer_size,device=device))

        self.w2 = torch.nn.Parameter(1e-3*torch.rand(512, 10, device=device))
        self.b2 = torch.nn.Parameter(torch.zeros(10,device=device))

        
        self.mlp = torch.nn.Linear(96, 96, bias=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x): # inputs : [batchsize, input_dim, time_windows]
        #change shape
        x = inputs.view(-1, 28 * 28)
        batch_size=x.size(0)
        input_dim=x.size(1)
        h = torch.zeros(batch_size, hidden_layer_size, device=device)
        for step in range(iter_time):
            h=h.mm(self.r1)            
            h=x+h+b1
            h=self.relu(h)
        
        x=h.mm(w2)+b2
        return x
'''

def prepare_fashion():        
    transform = transforms.Compose(
    [transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./data/fashion_MNIST', transform=transform, train=True, download=True)
    testset = torchvision.datasets.FashionMNIST(root='./data/fashion_MNIST', transform=transform, train=False, download=True)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(dataset=testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

    return trainloader, testloader


def prepare_net(batch_size=64, load=False, model_name=''):
    if(load==False):   
        model=model()
    else:
        net=torch.load(model_name)
        print('loading model:'+model_name)

    batch_size=64
    net.train()
    return net

def print_training_curve(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    
    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, '-', label='train', color='r')
    plt.plot(x, val_acc_list, '-', label='test', color='b')
    plt.title('Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='best')

    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.savefig('train_statistics_'+model_name+'.jpg')
    print(net.parameters())
    for parameters in net.parameters():
        print(parameters)

    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())

    plt.hist(r1, normed=True, facecolor='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('weights')
    plt.ylabel('frequency')
    plt.title('r1 Distribution')
    plt.savefig('r1_hist.jpg')

def get_layer(layer_type, layer_dict):
    if(layer_type in ['maxpool']):
        layer = nn.MaxPool2d(kernel_size=layer_dict['kernel_size'], stride=layer_dict['stride'], padding=layer_dict['padding'])
    elif(layer_type in ['global_avg']):
        layer = lambda x:torch.mean(x, dim=(2,3), keepdim=False)
    else:
        print('invalid layer_type:', end='')
        print(layer_type)
        input()
    return layer

def split_data_into_batches(data, batch_size): #data:(batch_size, image_size)
    sample_num = data.size(0)
    batch_sizes = [batch_size for _ in range(sample_num // batch_size)]
    if not sample_num % batch_size==0:
        batch_sizes.apend(sample_num % batch_size)
    return torch.split(data, section=batch_sizes, dim=0)

def combine_batches(dataloader): #data:(batch_num, batch_size, image_size)
    if not isinstance(dataloader, list):
        dataloader = list(dataloader)
    return torch.cat(data, dim=0)