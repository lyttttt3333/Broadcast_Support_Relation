import torch
import numpy as np
import os
import glob
import h5py

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class PointProcess():
    def __init__(self,path_list,device,scoring):
        super().__init__()
        self.path_list=path_list
        self.device=device
        self.scoring=scoring

    def fake_input(self,batch_num,env_point_num,obj_point_num,direction_num=None):
        self.env_point_batch=np.zeros([batch_num,env_point_num,3])
        self.obj_point_batch=np.zeros([batch_num,obj_point_num,3])
        self.dir_batch_batch=np.zeros([batch_num,direction_num,3])
        self.scores_batch=np.random.rand(batch_num,direction_num)
        self.env_point_batch=torch.from_numpy(self.env_point_batch).to(self.device)
        self.obj_point_batch=torch.from_numpy(self.obj_point_batch).to(self.device)
        self.dir_batch_batch=torch.from_numpy(self.dir_batch_batch).to(self.device)
        self.scores_batch=torch.from_numpy(self.scores_batch).to(self.device)

    def get_data(self,batch_num,env_point_num,obj_point_num,direction_num=None):
        if self.scoring:
            assert direction_num is not None
        else:
            assert direction_num is None
            direction_num=1
        data_dict=dict()
        obj_idx=list()
        for path in self.path_list:
            with h5py.File(path,"r") as file:
                for key in file.keys():
                    if key not in data_dict.keys():
                        data_dict[key]=list()
                    data_dict[key].append(file[key][:])
        for key in data_dict.keys():
            data_dict[key]=np.concatenate(data_dict[key],axis=0)
        self.data_dict = data_dict
        self.direction_num = direction_num
        self.batch_num = batch_num
    
    def flatten_data(self):
        new_batch_num = self.direction_num * self.batch_num
        for key in self.data_dict.keys():
            if key == "env_pts" or key == "target_pts":
                pts = self.data_dict[key].reshape(self.batch_num,1,-1,3)
                pts = np.repeat(pts, repeats=self.direction_num, axis=1)
                self.data_dict[key] = pts.reshape(new_batch_num,-1,3)
            elif key == "object_idx":
                pass
            else:
                self.data_dict[key]=self.data_dict[key].reshape(new_batch_num,-1)
        for key in self.data_dict.keys():
            self.data_dict[key] = torch.from_numpy(self.data_dict[key].astype(np.float32)).to(self.device)

    def squeeze_data(self):
        self.data_dict["best_direction"]=list()
        for idx in range(self.batch_num):
            direction_score = self.data_dict["scores"][idx]
            best_idx = np.argmax(direction_score)
            directions = self.data_dict["directions"][idx]
            best_direction = directions[best_idx]
            self.data_dict["best_direction"].append(best_direction[None,:])
        self.data_dict["best_direction"]=np.concatenate(self.data_dict["best_direction"],axis=0)
        for key in self.data_dict.keys():
            self.data_dict[key] = torch.from_numpy(self.data_dict[key].astype(np.float32)).to(self.device)
        del self.data_dict["directions"]
        del self.data_dict["scores"]
        




def find_batch(root_path, train_ratio):
    total_paths = glob.glob(os.path.join(root_path,"*.h5"))
    total_num = len(total_paths)
    train_num = int(total_num * train_ratio)
    eval_num = total_num - train_num
    np.random.shuffle(total_paths)
    train_path = total_paths[:train_num]
    eval_path = total_paths[train_num:train_num+eval_num]
    return train_path, eval_path
