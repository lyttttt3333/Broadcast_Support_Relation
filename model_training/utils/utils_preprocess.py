import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import random
import os

def find_all_ids(train:bool,input_path):
    id_list=[]
    path=input_path+"/train" if train else input_path+"/test"
    for item in os.listdir(path):
        id_list.append(int(item[4+1:]))
    return id_list,train


def find_batch_outer_env_obj(id:list):
    outer_env_list=[]
    for i in id[0]:
        outer_env_list+=find_batch_inner_env_obj(i,id[1])
    return outer_env_list

def find_batch_inner_env_obj(id:int,train:bool,input_path):
    path=input_path+"/train" if train else input_path+"/test"
    path=os.path.join(path, f"env_{id}")
    inner_env_list=[]
    all_need_path=[]
    for item in os.listdir(path):
        if item.split(".")[-1] == "npy":
            path_item=os.path.join(path,item)
            all_need_path.append(path_item)
    for item in os.listdir(path):
        if item.split("_")[0] == "object":
            object_path=os.path.join(path, item)
            inner_object_path=[]
            for iter in os.listdir(object_path):
                iter_path=os.path.join(object_path, iter)
                inner_path=os.listdir(iter_path)
                iter_full_path_list=[]
                for iter_path_item in inner_path:
                    iter_path_item_full=os.path.join(iter_path,iter_path_item)
                    iter_full_path_list.append(iter_path_item_full)
                iter_full_path_list+=all_need_path
                if len(iter_full_path_list)==6:
                    inner_object_path.append(iter_full_path_list)
            if len(inner_object_path) != 0:
                inner_env_list.append(inner_object_path)
    return inner_env_list

def data_filter(all_path_train,device):
    new_path_list=[]
    for action_batch_path in all_path_train:
        delta_list=[]
        for action_path in action_batch_path:
            pos_init=torch.from_numpy(np.load(action_path[1])).to(device)
            pos_final=torch.from_numpy(np.load(action_path[2])).to(device)
            delta=torch.norm(pos_final-pos_init,p=2,dim=-1)
            delta[delta<1]=0
            delta[delta>=1]=1
            delta_list.append(delta.unsqueeze(0))
        delta_mat=torch.cat(delta_list,dim=0)
        unique_element = torch.unique(delta_mat, dim=0)  
        for index in range(unique_element.shape[0]):
            new_path_list.append(action_batch_path[index])
    return new_path_list


def data_valid(all_path_train):
    new_list=[]
    for path in all_path_train:
        path_action=path[0]
        with open(path_action,'r') as f:
            direction=f.readline()
            part=f.readline()
            direction=direction.split(',')
            for xyz in range(3):
                if xyz==2:
                    direction[xyz]=direction[xyz].replace("\n", "")
                direction[xyz]=float(direction[xyz])         
            direction.append(int(part))
        direction=torch.tensor(direction)
        path_mask_line=path[5]
        mask_line = torch.tensor(np.load(path_mask_line).astype(np.int64))
        num = torch.sum(mask_line==direction[-1].item())
        if num >=256:
            new_list.append(path)
    return new_list

class Scene_into_Pair():

    def __init__(self,path_list,points_num,device):
        super().__init__()
        self.path_list=path_list
        self.num=points_num
        self.time=dict()
        self.device=device

    def preprocess(self):
        path_list=self.path_list
        data_list=[]
        mask_piece_list=[]
        mask_line_list=[]
        direction_list=[]
        for path in path_list:
            path_data=path[4]
            path_mask_piece=path[3]
            path_mask_line=path[5]
            path_action=path[0]
            data = torch.tensor(np.load(path_data)).unsqueeze(0).to(self.device)
            mask_piece = torch.tensor(np.load(path_mask_piece).astype(np.int64)).unsqueeze(0).to(self.device)
            mask_line = torch.tensor(np.load(path_mask_line).astype(np.int64)).unsqueeze(0).to(self.device)
            with open(path_action,'r') as f:
                direction=f.readline()
                part=f.readline()
                direction=direction.split(',')
                for xyz in range(3):
                    if xyz==2:
                        direction[xyz]=direction[xyz].replace("\n", "")
                    direction[xyz]=float(direction[xyz])         
                direction.append(int(part))
            direction_list.append(torch.tensor(direction).to(self.device).unsqueeze(0))
            points = list(range(data.shape[1])) 
            sampled_points = random.sample(points, self.num)
            data_list.append(data[:,sampled_points,:])
            mask_line_list.append(mask_line[:,sampled_points])
            mask_piece_list.append(mask_piece)
        self.data=torch.cat(data_list,dim=0)
        self.mask_piece=torch.cat(mask_piece_list,dim=0)
        self.mask_line=torch.cat(mask_line_list,dim=0)
        self.action=torch.cat(direction_list,dim=0)

    def get_gt_motion(self):
        init_gt_action_path_list=[]
        for i in self.path_list:
            init_gt_action_path_list.append(i[1])
        final_gt_action_path_list=[]
        for i in self.path_list:
            final_gt_action_path_list.append(i[2])
        motion_list=[]
        for i in range(len(init_gt_action_path_list)):
            init_gt_action=torch.tensor(np.load(init_gt_action_path_list[i])).to(self.device).squeeze(0)
            final_gt_action=torch.tensor(np.load(final_gt_action_path_list[i])).to(self.device).squeeze(0)
            delta=torch.norm(final_gt_action-init_gt_action,dim=-1,p=2,keepdim=False)
            delta[delta>=1]=1
            delta[delta<1]=0
            motion_list.append(delta)
        self.motion=motion_list

    def reorder(self):
        self.relation_list=[]
        motion_new_list=[]
        mask_piece_new_list=[]
        mask_line_list=[]
        for i in range(self.mask_piece.shape[0]):
            relation,new_mask_line=torch.unique(self.mask_line[i], sorted=True, return_inverse=True)
            mask_line_list.append(new_mask_line.unsqueeze(0))
            self.relation_list.append(relation.tolist())
            relation_=relation.tolist()
            exist_num=len(relation_)
            motion_new=torch.zeros(exist_num).to(self.device)
            mask_piece_new=100*torch.ones_like(self.mask_piece[i]).to(self.device)
            for item in range(exist_num):
                id=relation_[item]
                motion_new[item]=self.motion[i][id]
                mask_piece_new[self.mask_piece[i]==id]=item
            motion_new_list.append(motion_new)
            mask_piece_new_list.append(mask_piece_new.unsqueeze(0))
        self.motion=motion_new_list
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


def get_part_point_batch(point_loc,mask,num_for_part,device):
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

def get_action_info(action_batch,relation_list,device):
    action_graph=[]
    action_indices=[]
    for i in range(len(relation_list)):
        relation=relation_list[i]
        action=action_batch[i]
        action_mat=torch.zeros([len(relation),4]).to(device)
        init_id=int(action[-1].item())
        tran_id=relation.index(init_id)
        action_mat[tran_id][-1]=1
        action_mat[tran_id][:-1]=action[:-1]/torch.norm(action[:-1],dim=-1,p=2,keepdim=False)
        action_graph.append(action_mat)
        action_indices.append(tran_id)
    return action_graph,action_indices


def get_init_state(action_graph,action_index):
    state_list=[]
    for i in range(len(action_index)):
        state=torch.zeros_like(torch.sum(action_graph[i],dim=-1,keepdim=False))
        state[action_index[i]]=1
        state_list.append(state)
    return state_list



def find_relation(state_info_list,batched_mask,device):
    B=len(state_info_list)
    relation_mat_list=[]
    for i in range(B):
        state_info=state_info_list[i]
        relation_mat=torch.zeros(state_info.shape[0]).to(device)
        mask=batched_mask[i]
        for j in range(state_info.shape[0]):
            if state_info[j]==1:
                relation_mat=_get_part_adj_2d(mask,j)
                relation_mat[j]=0
        relation_mat_list.append(relation_mat)
    return relation_mat_list


def _get_part_adj_2d(mask_mat,action_index):
    half_length=8
    whole_length=2*half_length+1
    pool = nn.MaxPool2d(kernel_size=whole_length, stride=1, padding=half_length)  
    mask_piece=mask_mat[:,:,action_index].unsqueeze(0).unsqueeze(0)
    after_pool=pool(mask_piece).squeeze(0).squeeze(0).unsqueeze(-1).repeat(1,1,mask_mat.shape[-1])
    mask_block_exam=mask_mat*after_pool
    exam_line=torch.sum(torch.sum(mask_block_exam,dim=0),dim=0).squeeze(0)[:25]
    exam_line[exam_line>0]=1
    nearbys=exam_line
    return nearbys

def merget_pairs(motion,relation,partpoint_list,state_list,action_list,save_path,parent_index,train):
    index=0
    suffix="_train" if train else "_test"
    for i in range(len(partpoint_list)):
        state=state_list[i]
        state_id=torch.where(state==1)[0].item()
        result=motion[i]
        adj=relation[i]
        partpoint=partpoint_list[i]
        indices_to_save=torch.where(adj==1)[0].tolist()
        for indice in indices_to_save:
            point_receive=partpoint[indice]
            point_source=partpoint[state_id]
            if not compute_adj(point_receive,point_source):
                continue
            else:
                pass
            kinetic=result[indice]
            action=action_list[state_id]
            cent=torch.mean(point_source,dim=0,keepdim=False)
            cent_receive=torch.mean(point_receive,dim=0,keepdim=False)
            action_succ=cent_receive-cent
            action_succ=action_succ/torch.norm(action_succ,dim=-1,p=2)
            point_merge=torch.cat([point_receive,point_source],dim=0)
            point_merge=point_merge-cent
            arm=torch.norm(point_merge,p=2,dim=-1,keepdim=False)
            max_arm=torch.max(arm)
            point_merge=point_merge/max_arm
            action=action[:3]
            addition=torch.zeros_like(action)
            addition[0]=kinetic
            full_data=torch.cat([point_merge,action.unsqueeze(0),addition.unsqueeze(0)],dim=0)
            save_data=full_data.cpu().numpy()
            if kinetic.item() == 1:
                np.save(save_path+"_pair"+suffix+"/succ/"+f"{parent_index+index}.npy",save_data)
            else:
                np.save(save_path+"_pair"+suffix+"/fail/"+f"{parent_index+index}.npy",save_data)
            index+=1



def compute_adj(point1,point2):
    dist=torch.cdist(point1,point2)
    min_value=torch.min(dist)
    if min_value < 1.5:
        return True
    else:
        return False