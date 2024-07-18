import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import random
import glob
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import h5py

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz, points, query_pts):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, query_pts)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x=conv(new_points)
            new_points =  F.relu(bn(x))

        new_points = torch.max(new_points, 2)[0]
        #new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

def sample_and_group(radius, nsample, xyz, ft, query_pts):
    new_xyz = query_pts.reshape(-1,1,3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx).squeeze(1)
    grouped_ft = index_points(ft, idx).squeeze(1)
    grouped_pts = torch.cat([grouped_xyz,grouped_ft],dim=-1)
    return grouped_pts

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group_all(xyz, points):
    xyz = xyz.permute(0, 2, 1)
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def segment_target(pts,obj_idx):
    target_pts = pts[obj_idx]
    env_pts = np.delete(pts,[obj_idx],axis=0)
    env_pts = env_pts.reshape(-1,3)
    env_pts = torch.from_numpy(env_pts)[None,:]
    idx = farthest_point_sample(env_pts, npoint=1024)
    env_pts = index_points(env_pts, idx)
    env_pts = env_pts.cpu().numpy().reshape(-1,3)
    return target_pts, env_pts

def get_part_point_batch(point_loc,mask,device):
    # mask [B,N,max_mask_piece]
    num_for_part=256
    batched_part_point=[]
    for i in range(len(mask)):
        mask_piece=mask[i]
        point_piece=point_loc[i]
        part_point_piece=torch.zeros([mask_piece.shape[1],num_for_part,3]).to(device)
        for j in range(mask_piece.shape[1]):
            mask_line=mask_piece[:,j]
            aim_part_indice=torch.where(mask_line==1)[0].tolist()
            aim_part_point=point_piece[aim_part_indice,:]
            num_point=len(aim_part_indice)
            num_aim_list=list(range(num_point))
            chose_list=[]
            if num_point>num_for_part:
                chose_list=random.choices(num_aim_list,k=num_for_part)
                part_point=aim_part_point[chose_list,:]
            else:
                while len(chose_list)<num_for_part:
                    chose_list+=random.choices(num_aim_list,k=num_point)
                chose_list=chose_list[:num_for_part]
                part_point=aim_part_point[chose_list,:]
            part_point_piece[j,:,:]=part_point
        batched_part_point.append(part_point_piece)
    return batched_part_point


def find_batch(dataset_dir,train_ratio=0.9):
    env_paths = glob.glob(os.path.join(dataset_dir,"*"))
    full_list = list()
    for env_path in env_paths:
        trial_paths = glob.glob(os.path.join(env_path,"rigid_?.npy"))
        trial_paths += glob.glob(os.path.join(env_path,"rigid_??.npy"))
        mask_line_path = os.path.join(env_path,"mask_line.npy")
        mask_mat_path = os.path.join(env_path,"mask_mat.npy")
        pts_path = os.path.join(env_path,"points.npy")
        if os.path.exists(mask_line_path) and os.path.exists(mask_mat_path) and os.path.exists(pts_path):
            condition=[pts_path,mask_line_path,mask_mat_path]
        else:
            continue
        for trial_path in trial_paths:
            obj_idx = int(trial_path.split("_")[-1].split(".")[0])
            result_path = os.path.join(env_path,f"rigid_{obj_idx}_result.npy")
            pose_path = os.path.join(env_path,f"rigid_{obj_idx}_pose.npy")
            if os.path.exists(result_path):
                full_list.append([trial_path,result_path,condition,obj_idx,pose_path])
    train_num = int(train_ratio * len(full_list))
    np.random.shuffle(full_list)
    train_path = full_list[:train_num]
    eval_path = full_list[train_num:]
    return train_path, eval_path


class PreCache():
    def __init__(self,path_list,points_num,device):
        super().__init__()
        self.path_list=path_list
        self.num=points_num
        self.time=dict()
        self.device = device

    def preprocess(self):
        path_list=self.path_list
        data_list=[]
        mask_piece_list=[]
        mask_line_list=[]
        direction_list=[]
        for path in path_list:
            path_data=path[0]
            path_mask_piece=path[2]
            path_mask_line=path[1]
            data = torch.tensor(np.load(path_data)).unsqueeze(0).to(self.device)
            mask_piece = torch.tensor(np.load(path_mask_piece).astype(np.int64)).unsqueeze(0).to(self.device)
            mask_line = torch.tensor(np.load(path_mask_line).astype(np.int64)).unsqueeze(0).to(self.device)
            points = list(range(data.shape[1])) 
            sampled_points = random.sample(points, self.num)
            data_list.append(data[:,sampled_points,:])
            mask_line_list.append(mask_line[:,sampled_points])
            mask_piece_list.append(mask_piece)
        self.data=torch.cat(data_list,dim=0)
        self.mask_piece=torch.cat(mask_piece_list,dim=0)
        self.mask_line=torch.cat(mask_line_list,dim=0)

    def reorder(self):
        self.relation_list=[]
        mask_piece_new_list=[]
        mask_line_list=[]
        for i in range(self.mask_piece.shape[0]):
            relation,new_mask_line=torch.unique(self.mask_line[i], sorted=True, return_inverse=True)
            mask_line_list.append(new_mask_line.unsqueeze(0))
            self.relation_list.append(relation.tolist())
            relation_=relation.tolist()
            exist_num=len(relation_)
            mask_piece_new=100*torch.ones_like(self.mask_piece[i]).to(self.device)
            for item in range(exist_num):
                id=relation_[item]
                mask_piece_new[self.mask_piece[i]==id]=item
            mask_piece_new_list.append(mask_piece_new.unsqueeze(0))
        self.mask_piece=torch.cat(mask_piece_new_list,dim=0)
        self.mask_line=torch.cat(mask_line_list,dim=0)

    def expand(self):
        self.expand_mask_line=[]
        self.expand_mask_piece=[]
        for i in range(self.mask_piece.shape[0]):
            new_mask_line=torch.zeros(self.mask_line.shape[1],len(self.relation_list[i])).to(self.device)
            new_mask_piece=torch.zeros(self.mask_piece.shape[1],self.mask_piece.shape[2],len(self.relation_list[i])).to(self.device)
            for j in range(len(self.relation_list[i])):
                new_mask_line[:,j][self.mask_line[i]==j]=1
                new_mask_piece[:,:,j][self.mask_piece[i]==j]=1
            self.expand_mask_line.append(new_mask_line)
            self.expand_mask_piece.append(new_mask_piece)


def save_h5py(data:dict,path):
    with h5py.File(path,"w") as file:
        for name in data.keys():
            file.create_dataset(name=name,data=data[name])

def query_adjacent(env_pts,query_pts,mask_line,device):
    mask_line = mask_line[:,:,None]
    query_pts = torch.from_numpy(query_pts.astype(np.float32)).to(device)
    pts_group = sample_and_group(radius=0.6,nsample=256,xyz=env_pts,ft=mask_line,query_pts=query_pts)
    return pts_group.cpu().numpy()

def get_squeezed_mask_line(expand_mask_line,obj_idx_list):
    mask_line_list=list()
    for idx, mask_line in enumerate(expand_mask_line):
        obj_idx = obj_idx_list[idx]
        target_mask_line = mask_line[:,obj_idx][None,:]
        mask_line_list.append(target_mask_line)
    mask_line_list = torch.cat(mask_line_list,dim=0)
    return mask_line_list


def load_pts(data_dict,device):
    full_pts_list = data_dict["condition"]
    point=PreCache(full_pts_list,10240,device)
    point.preprocess()
    point.reorder()
    point.expand()
    batched_partpoint_list=get_part_point_batch(point.data,point.expand_mask_line,device)
    data_dict["target_pts"]=list()
    for idx, full_pts in enumerate(batched_partpoint_list):
        target_idx = data_dict["target_idx"][idx]
        target_pts = full_pts[target_idx][None,:].cpu().numpy()
        data_dict["target_pts"].append(target_pts)
    data_dict["target_pts"]=np.concatenate(data_dict["target_pts"],axis=0)
    mask_line=get_squeezed_mask_line(point.expand_mask_line,data_dict["target_idx"])
    query_pt = data_dict["trial_pts"][:,:3]
    adjacent_pts = query_adjacent(env_pts=point.data,query_pts=query_pt,mask_line=mask_line,device=device)
    data_dict["env_pts"]=adjacent_pts

    del data_dict["target_idx"]
    del data_dict["condition"]
    del point
    return data_dict

def data_filter(data_list):
    new_data_list=[]
    for data in data_list:
        result_path=data[1]
        result=np.load(result_path)
        valid_flag=np.sum(result)
        if valid_flag:
            new_data_list.append(data)
    return new_data_list

def data_balance(data_list):
    succ_list=[]
    fail_list=[]
    for item in data_list:
        point_path=item[0]
        result_path=item[1]
        points_pool=torch.from_numpy(np.load(point_path))
        result=torch.from_numpy(np.load(result_path))
        succ_flag=torch.sum(result,dim=-1,keepdim=False)
        succ_indices=torch.where(succ_flag>=1)[0].tolist()
        fail_indices=torch.where(succ_flag<1)[0].tolist()
        succ_pool=points_pool[succ_indices,:]
        fail_pool=points_pool[fail_indices,:]
        succ_len=succ_pool.shape[0]
        fail_len=fail_pool.shape[0]
        select_num=succ_len if succ_len < 6 else 6
        if select_num == succ_len:
            succ_selected=succ_pool
        else:
            selected_succ_indices=random.sample(range(succ_len),select_num)
            succ_selected=succ_pool[selected_succ_indices,:]
        for pt_idx in range(select_num):
            succ_point=succ_selected[pt_idx]
            new_item=item+[succ_point.cpu().numpy()]+[1,]
            succ_list.append(new_item)
            selected_fail_indices=random.sample(range(fail_len),5)
            fail_selected=fail_pool[selected_fail_indices,:]
            dist_mat=torch.cdist(succ_pool,fail_selected)
            min_dist=torch.min(dist_mat,dim=0)[0]
            ratio=random.uniform(0,1)
            if ratio < 1/3:
                max_value=torch.max(min_dist,dim=0)[0]
                indice=torch.where(min_dist==max_value)[0].tolist()[0]
            elif ratio > 1/3:
                min_value=torch.min(min_dist,dim=0)[0]
                indice=torch.where(min_dist==min_value)[0].tolist()[0]
            else:
                indice=0
            fail_point=fail_selected[indice]
            new_item=item+[fail_point.cpu().numpy()]+[0,]
            fail_list.append(new_item)
    random.shuffle(succ_list)
    random.shuffle(fail_list)
    return succ_list,fail_list

def data_convert(data_list):
    data_dict = dict()
    key_list=["condition","target_idx","trial_pts","point","point_result","pose","pose_result"]
    for key in key_list:
        data_dict[key]=list()
    for data in data_list:

        item = np.load(data[0])[None,:,:3]
        data_dict["point"].append(item)

        item = data[2]
        data_dict["condition"].append(item)

        item = np.array([data[3]])
        data_dict["target_idx"].append(item)

        item = np.load(data[4])[None,:,:,:]
        data_dict["pose"].append(item)

        item = np.load(data[1])[None,:,:]
        data_dict["pose_result"].append(item)

        item = np.array([data[5]])
        data_dict["trial_pts"].append(item)

        item = np.array([data[6]])
        data_dict["point_result"].append(item)

    for key in ["target_idx","trial_pts","point_result","pose","pose_result","point"]:
        data_dict[key]=np.concatenate(data_dict[key],axis=0)

    data_dict = merge_pose(data_dict)

    return data_dict

def merge_pose(data_dict):
    n = data_dict["pose"].shape[2]
    pts = data_dict["point"][:,:,None,:]
    pts = np.repeat(pts,repeats=n,axis=2)
    data_dict["pose"]= np.concatenate([pts,data_dict["pose"]],axis=-1)
    data_dict["pose"]=data_dict["pose"].reshape(data_dict["pose"].shape[0],-1,7)
    data_dict["pose_result"]=data_dict["pose_result"].reshape(data_dict["pose_result"].shape[0],-1,1)
    return data_dict
