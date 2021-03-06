import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms

from utils import get_from_dict

class data_loader():
    def _init_(self, dict_=None):
        if dict_ is None:
            self.dict = {}
        else:
            sekf.dict = dict_

        if self.dict.get("data_dir") is None:
            self.dict["data_dir"] = {}
        self.data_dir = self.dict["data_dir"]

    def set_data_dir(self, type_, dir_):
        if type_ in ["cifar10", "CIFAR10", "cifar_10"]:
            self.data_dir["cifar10"] = dir_
        elif type_ in ["MNIST","mnist"]:
            self.data_dir["mnist"] = dir_

    def prepare_cifar10(self, dict_=None):
        if self.data_dir.get("cifar10") is None:
            print("cifar 10 data path is undesignated!")
            input()
            return
        else:
            data_dir_ = self.data_dir["cifar10"]

        batch_size_ = dict_["batch_size"]
        fetch_thread_num = dict_["fetch_thread_num"]

        trans_train=[]
        trans_test=[]
        prepare_method = dict_["prepare_method"]

        shuffle = get_from_dict(dict_, "shuffle", True)
        
        trans_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # (value - mean) / std. (0, 1) --> (~-2.5, 2.5) with 0 mean and 1.0 std.

        if prepare_method=="tencrop":
            feature_map_width=24
            trans_train.append(transforms.ToTensor()) # transform from pixel value range from (0, 255) tp (0, 1)
            trans_test.append(transforms.ToTensor())

            TenCrop=[
                transforms.TenCrop(24),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([trans_norm(crop) for crop in crops]))
                ]
            trans_train.append(TenCrop)
            trans_test.append(TenCrop)

        elif prepare_method=="norm":
            feature_map_width=32
            trans_train.append(transforms.ToTensor())
            trans_test.append(transforms.ToTensor())
            trans_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
            trans_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        transform_test=transforms.Compose(trans_test)
        transform_train=transforms.Compose(trans_train)
        
        trainset = torchvision.datasets.CIFAR10(root=data_dir_, train=True, transform=transform_train, download=False)
        testset = torchvision.datasets.CIFAR10(root=data_dir_,train=False, transform=transform_test, download=False)
        
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size_, shuffle=shuffle, num_workers=fetch_thread_num)
        testloader = DataLoader(dataset=testset, batch_size=batch_size_, shuffle=shuffle, num_workers=fetch_thread_num)
        return trainloader, testloader
    
    def anal_img_data():
         
    
    def prepare_mnist():
        return trainloader, testloader
