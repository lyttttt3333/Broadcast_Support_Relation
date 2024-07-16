import torch
import numpy as np
import os


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def find_batch(root_path, id_length, phase:str, type:str):
    id_list=[]
    path=root_path+f"/{phase}/{type}"
    for item in os.listdir(path):
        id_list.append(int(item[id_length:]))
    batch_list=[]
    for idx in id_list:
        full_path=path+f"/{idx}"
        env_path=full_path+"/env.npy"
        obj_path=full_path+"/obj.npy"
        score_path=full_path+"/scores.npy"
        pkg=(env_path, obj_path, score_path)
        batch_list.append(pkg)
    return batch_list

class PointProcess():
    def __init__(self,path_list,device):
        super().__init__()
        self.path_list=path_list
        self.device=device

    def get_data(self,batch_num,env_point_num,obj_point_num):
        self.env_point_batch=np.zeros(batch_num,env_point_num,3)
        self.obj_point_batch=np.zeros(batch_num,obj_point_num,3)
        self.ids_batch=np.zeros(batch_num)
        self.scores_batch=np.zeros(batch_num)
        for i in range(len(self.path_list)):
            path=self.path_list[i]
            env_point_path=path[0]
            obj_point_path=path[1]
            score_path=path[2]
            env_point=np.load(env_point_path)
            self.env_point_batch[i]=env_point
            obj_point=np.load(obj_point_path)
            self.obj_point_batch[i]=obj_point
            input=np.load(score_path)
            self.ids_batch[i]=input[0]
            self.scores_batch[i]=input[1]
        self.env_point_batch=torch.from_numpy(self.env_point_batch).to(self.device)
        self.obj_point_batch=torch.from_numpy(self.obj_point_batch).to(self.device)
        self.ids_batch=torch.from_numpy(self.ids_batch).to(self.device)
        self.scores_batch=torch.from_numpy(self.scores_batch).to(self.device)