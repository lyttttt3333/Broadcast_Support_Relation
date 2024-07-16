import numpy as np
import torch
import h5py
import random
import torch
import glob
import os 
import shutil

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

def get_part_point_batch(point_loc,mask):
    # mask [B,N,max_mask_piece]
    num_for_part=256
    batched_part_point=[]
    for i in range(len(mask)):
        mask_piece=mask[i]
        point_piece=point_loc[i]
        part_point_piece=torch.zeros([mask_piece.shape[1],num_for_part,3]).to("cuda:0")
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

def find_all_envs_path(root_dir,train=True):
    if train == True:
        path_list = glob.glob(os.path.join(root_dir,"train","env_*"))
    else:
        path_list = glob.glob(os.path.join(root_dir,"test","env_*"))
    full_pts_list=[]
    for path in path_list:
        pts_path = os.path.join(path,"points.npy")
        mask_line_path = os.path.join(path,"mask_line.npy")
        mask_piece_path = os.path.join(path,"mask_piece.npy")        
        full_pts_list.append([pts_path,mask_line_path,mask_piece_path])
    return full_pts_list, path_list

class PreCache():
    def __init__(self,path_list,points_num, device):
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


root_dir = "D:\\broadcast_final\\data"

input_root_path = os.path.join(root_dir,"data_interaction")
save_root_path = os.path.join(input_root_path,"direction")

if os.path.exists(save_root_path):
    shutil.rmtree(save_root_path)
os.mkdir(save_root_path)

envs_wrapper_list, envs_root_list=find_all_envs_path(input_root_path)



point=PreCache(envs_wrapper_list,10240, device="cuda:0")
point.preprocess()
point.reorder()
point.expand()


batched_partpoint_list=get_part_point_batch(point.data,point.expand_mask_line)

k_exp=4

point_cloud = None
for env_idx, env_root_path in enumerate(envs_root_list):
    object_root_path_list = glob.glob(os.path.join(env_root_path, "object_*"))
    for obj_idx, object_root_path in enumerate(object_root_path_list):
        save_data=dict()
        iter_path_list = glob.glob(os.path.join(object_root_path, "iter_*"))
        direction_score_list = []
        direction_list=[]
        for iter_path in iter_path_list:
            if not os.path.exists(os.path.join(iter_path,"motion_checkpoint.npy")) :
                drop=True
                break
            else:
                motion_path = os.path.join(iter_path,"motion_checkpoint.npy")
                motion = np.load(motion_path)
                direction_score=np.sum(motion) - 1
                direction_score_list.append(direction_score)

                action_path = os.path.join(iter_path,"action_info.txt")
                with open(action_path,'r') as f:
                    direction=f.readline()
                    part=f.readline()
                    direction=direction.split(',')
                    for xyz in range(3):
                        if xyz==2:
                            direction[xyz]=direction[xyz].replace("\n", "")
                        direction[xyz]=float(direction[xyz])   
                direction = np.array(direction)/np.linalg.norm(np.array(direction),ord = 2)      
                
                direction_list.append(direction[None,:])


        direction_score_list = np.array(direction_score_list)
        max_mv=np.max(direction_score_list)
        min_mv=np.min(direction_score_list)
        if max_mv == min_mv:
            drop = True
            continue
        norm_direction_score=1 - (direction_score_list-min_mv)/(max_mv-min_mv)
        print(norm_direction_score)
        norm_direction_score=(np.exp(k_exp*norm_direction_score)-1)/(np.exp(k_exp) - 1)
        print(norm_direction_score)

        direction_list = np.concatenate(direction_list,axis=0)
        object_index = np.array([int(object_root_path.split("_")[-1])])
        save_data["object_idx"]=object_index
        save_data["directions"]=direction_list[None,:]
        save_data["scores"]=norm_direction_score[None,:]
        full_pts= batched_partpoint_list[env_idx].cpu().numpy()
        target_pts, env_pts = segment_target(full_pts, object_index[0])
        save_data["target_pts"]=target_pts[None,:]
        save_data["env_pts"]=env_pts[None,:]
        save_index = f"{env_idx}"+"_"+f"{obj_idx}"
        save_path = os.path.join(save_root_path,f"{save_index}.h5")
        save_h5py(save_data,save_path)


