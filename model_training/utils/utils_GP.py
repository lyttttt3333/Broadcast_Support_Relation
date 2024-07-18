import torch
import numpy as np
import os
import glob
import h5py

def find_batch(root_path):
    data_path = os.path.join(root_path,"succ_*.h5")
    succ_list=glob.glob(data_path)
    data_path = os.path.join(root_path,"fail_*.h5")
    fail_list=glob.glob(data_path)
    assert len(succ_list) == len(fail_list)
    return succ_list, fail_list


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class PointProcess():
    def __init__(self,path_list,device):
        super().__init__()
        self.path_list=path_list
        self.device=device
    
    def load_data(self):
        data_dict=dict()
        for path in self.path_list:
            with h5py.File(path,"r") as file:
                for key in file.keys():
                    if key not in data_dict.keys():
                        data_dict[key]=list()
                    data_dict[key].append(file[key][:])
        for key in data_dict.keys():
            data_dict[key] = np.concatenate(data_dict[key],axis=0)
        self.data_dict = data_dict

    def split_point_pose(self):
        self.data_dict["grasp_pt"]=self.data_dict["pose"][:,:,:3]
        self.data_dict["grasp_pose"]=self.data_dict["pose"][:,:,3:]
        self.data_dict["grasp_pose_result"]=self.data_dict["pose_result"][:]

    @property
    def env_pts(self):
        return torch.from_numpy(self.data_dict["env_pts"]).to(self.device).to(torch.float32)

    @property
    def result(self):
        return torch.from_numpy(self.data_dict["result"]).to(self.device).to(torch.float32).reshape(-1,1)

    @property
    def target_pts(self):
        return torch.from_numpy(self.data_dict["target_pts"]).to(self.device).to(torch.float32)

    @property
    def trials(self):
        return torch.from_numpy(self.data_dict["trial_pts"]).to(self.device).to(torch.float32)
    
    @property
    def grasp_pt(self):
        return torch.from_numpy(self.data_dict["grasp_pt"]).to(self.device).to(torch.float32)

    @property
    def grasp_pose(self):
        return torch.from_numpy(self.data_dict["grasp_pose"]).to(self.device).to(torch.float32)
    
    @property
    def grasp_result(self):
        return torch.from_numpy(self.data_dict["grasp_pose_result"]).to(self.device).to(torch.float32)


