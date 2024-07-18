import os
import random
import torch
import glob
import numpy as np


def read_pair(input_path, train, seed):
    suffix="pair_train" if train else "pair_test"
    succ_list=[]
    fail_list=[]
    succ_path = os.path.join(input_path,"pair",suffix,"succ","*.npy")
    succ_list = glob.glob(succ_path)

    fail_path = os.path.join(input_path,"pair",suffix,"fail","*.npy")
    fail_list = glob.glob(fail_path)

    if len(succ_list)>len(fail_list):
        random.seed(seed)
        succ_list=random.sample(succ_list,len(fail_list))
    if len(succ_list)<len(fail_list):
        random.seed(seed)
        fail_list=random.sample(fail_list,len(succ_list))

    return succ_list,fail_list

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class PointProcess_LDR():
    def __init__(self,path_list,point_num,device):
        super().__init__()
        self.path_list=path_list
        self.point_num=point_num
        self.device=device

    def get_pair(self):
        partpoint1_list=[]
        partpoint2_list=[]
        action_list=[]
        result_list=[]
        for i in range(len(self.path_list)):
            full_data=np.load(self.path_list[i])
            full_data=torch.from_numpy(full_data).to(self.device)
            partpoint1_list.append(full_data[0:self.point_num,:].unsqueeze(0))
            partpoint2_list.append(full_data[self.point_num:2*self.point_num,:].unsqueeze(0))
            action_list.append(full_data[2*self.point_num].unsqueeze(0))
            result_list.append(full_data[2*self.point_num+1,0].unsqueeze(0))
        self.partpoint1=torch.cat(partpoint1_list,dim=0)
        self.partpoint2=torch.cat(partpoint2_list,dim=0)
        self.action=torch.cat(action_list,dim=0)
        self.result=torch.cat(result_list,dim=0)