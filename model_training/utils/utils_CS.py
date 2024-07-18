import torch
import torch.nn as nn
import numpy as np
import random


class PointProcess_eval():
    def __init__(self,sample_num,resample_num,device):
        super().__init__()
        self.num=sample_num
        self.re_num=resample_num
        self.device=device

    def input_form(self,point_path,mask_path,init_mask_path):
        self.point=torch.from_numpy(np.load(point_path)).to(self.device)
        self.mask=torch.from_numpy(np.load(mask_path).astype(np.float32)).to(self.device)
        self.init_mask=torch.from_numpy(np.load(init_mask_path).astype(np.float32)).to(self.device)

    def sample_point(self):
        init_len=self.point.shape[0]
        sampled_points = random.sample(list(range(init_len)), self.num)
        self.point=self.point[sampled_points,:]
        self.mask=self.mask[sampled_points]

    def mask_expand(self):
        self.mask_relation, self.mask = torch.unique(self.mask, return_inverse=True)
        self.init_mask_relation, self.init_mask = torch.unique(self.init_mask, return_inverse=True)
        self.total_num=len(self.mask_relation)
        expand_mask_list=[]
        expand_init_mask_list=[]
        for i in range(self.total_num):
            mask_line=torch.zeros_like(self.mask)
            mask_line[self.mask==i]=1
            expand_mask_list.append(mask_line.unsqueeze(-1))
            mask_piece=torch.zeros_like(self.init_mask)
            mask_piece[self.init_mask==i]=1
            expand_init_mask_list.append(mask_piece.unsqueeze(-1))
        self.mask=torch.cat(expand_mask_list,dim=-1)
        self.init_mask=torch.cat(expand_init_mask_list,dim=-1)

    def get_part_point_batch(self):
        num_for_part=self.re_num
        mask=self.mask
        point=self.point
        part_point_piece_list=[]
        for j in range(mask.shape[1]):
            mask_line=mask[:,j]
            aim_part_indice=torch.where(mask_line==1)[0].tolist()
            aim_part_point=point[aim_part_indice,:]
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
            part_point_piece_list.append(part_point.unsqueeze(0))
        self.part_point=torch.cat(part_point_piece_list,dim=0)


def update_states(action_pred,state_before,threshold):
    states_new=torch.zeros_like(state_before)
    for j in range(states_new.shape[0]):
        if state_before[j]==1 or state_before[j]==-1:
            states_new[j]=-1
        else:
            if action_pred[j]>threshold:
                states_new[j]=1
            else:
                states_new[j]=0
    return states_new
        


def find_relation(state,num,mask,type,threshold,device):
    relation_mat=torch.zeros(num,num).to(device)
    for j in range(num):
        if state[j]==1:
            if type == "2d_adjacent":
                relation_mat[j,:]=_get_part_adj_2d(mask,j, threshold)
            elif type == "3d_adjacent":
                relation_mat[j,:]=_get_part_adj_3d(mask,threshold)
            else:
                raise ValueError
        if state[j]==0:
            relation_mat[j,:]=torch.zeros_like(relation_mat[:,j])
        if state[j]==-1:
            relation_mat[j,:]=torch.zeros_like(relation_mat[:,j])
    return relation_mat

def _get_part_adj_2d(mask,action_index,adjacent_threshold):
    whole_length=2*adjacent_threshold+1
    pool = nn.MaxPool2d(kernel_size=whole_length, stride=1, padding=adjacent_threshold)  
    mask_piece=mask[:,:,action_index].unsqueeze(0).unsqueeze(0).float()
    after_pool=pool(mask_piece).squeeze(0).squeeze(0).unsqueeze(-1).repeat(1,1,mask.shape[-1])
    mask_block_exam=mask*after_pool
    exam_line=torch.sum(torch.sum(mask_block_exam,dim=0),dim=0).squeeze(0)[:25]
    exam_line[exam_line>0]=1
    nearbys=exam_line
    return nearbys

def _get_part_adj_3d(mask,action_index,adjacent_threshold):
    whole_length=2*adjacent_threshold+1
    pool = nn.MaxPool2d(kernel_size=whole_length, stride=1, padding=adjacent_threshold)  
    mask_piece=mask[:,:,action_index].unsqueeze(0).unsqueeze(0)
    after_pool=pool(mask_piece).squeeze(0).squeeze(0).unsqueeze(-1).repeat(1,1,50)
    mask_block_exam=mask*after_pool
    exam_line=torch.sum(torch.sum(mask_block_exam,dim=0),dim=0).squeeze(0)[:25]
    exam_line[exam_line>0]=1
    nearbys=exam_line
    return nearbys

def update_layer(relation,
                 state,
                 action_graph,
                 part_point,
                 model_cls,
                 model_rds,
                 model_rdp,
                 order,
                 threshold,
                 device):
    action_pred_next=torch.zeros_like(action_graph)
    for j in range(relation.shape[1]):
        motion_sum_cls=torch.tensor([0.]).to(device)
        for k in range(relation.shape[0]):
            if relation[k,j].item()==1:
                p1=(part_point[k]).unsqueeze(0)
                p2=(part_point[j]).unsqueeze(0)
                direction=find_optimal_directions(model_rds,model_rdp,p1,p2)
                candidate_cls=model_cls.forward(p1,p2,direction.reshape(1,-1),train=False)
                if candidate_cls.item()>motion_sum_cls.item():
                    motion_sum_cls=candidate_cls.squeeze(0)
                if candidate_cls.item()>=threshold:
                    order[k,j]=1
        if state[j] == 0:
            action_pred_next[j]=motion_sum_cls
        if state[j] == 1 or state[j] == -1:
            action_pred_next[j]=action_graph[j]
    return action_pred_next,order

def find_optimal_directions(model_rds,
                            model_rdp,
                            receive,
                            source,):
    direction_candidate=model_rdp.propose(16,receive,source)
    direction_scores=model_rds.predict(receive,source,direction_candidate).squeeze(-1)
    idx=torch.argmax(direction_scores)
    return direction_candidate[idx]


def get_init_state(num,action_index,device):
    state=torch.zeros([num]).to(device)
    state[action_index]=1
    return state

def get_action_graph(num,action_index,device):
    state=torch.zeros([num]).to(device)
    state[action_index]=1
    return state

def get_init_order(num,device):
    order=torch.zeros([num,num]).to(device)
    return order

def update_access(access_list,batched_relation_list):
    new_list=[]
    for i in range(len(access_list)):
        new=torch.sum(batched_relation_list[i],dim=0,keepdim=False)
        old=access_list[i]
        accessed=new+old
        accessed[accessed>=1]=1
        new_list.append(accessed.clone().detach())
    return new_list

def update_stop(state):
    ones_indices=torch.where(state==1)[0].tolist()
    if len(ones_indices)!=0:
        return False
    return True

def clone(input_list):
    output_list=[]
    for i in input_list:
        output_list.append(i.clone().detach())
    return output_list