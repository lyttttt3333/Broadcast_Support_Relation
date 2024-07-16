import numpy as np
import os
import torch
import yaml
import json
import random
import glob


from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims import GeometryPrim,RigidPrim,XFormPrim
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrim,RigidPrim,XFormPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, create_prim
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage, open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.cloner import GridCloner
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.manipulators.grippers import ParallelGripper, SurfaceGripper
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, create_prim, get_prim_at_path
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.franka import Franka
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, VisualCuboid



TRANSFORM_SCALE=88

def read_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return yaml_data


def random_select_orientation(device):
    theta=np.random.uniform(0,np.pi/2)
    pha=np.random.uniform(0,np.pi*2)
    v1=np.cos(theta)
    v2=np.sin(theta)*np.cos(pha)
    v3=np.sin(theta)*np.sin(pha)
    return 0.6*torch.tensor([v3,v2,v1,0,0,0],dtype=torch.float,device=device),v1,v2,v3

def check_mask_visiable(mask,tolerance,max_num):
    is_visiable=[]
    for i in range(max_num):
        if np.sum(mask==i) > tolerance:
            is_visiable.append(i)
    return is_visiable

class LogInteraction():
    def __init__(self,id,path,scene_setting,camera,mask_tolerance=200,mask_max_num=20,device=None):
        import shutil
        self.mask_tolerance=mask_tolerance
        self.mask_max_num=mask_max_num
        self.scene_setting=scene_setting
        self.camera=camera
        folder_name = f"env_{id}"
        self.id=id
        self.device = device
        self.root_path=os.path.join(path, folder_name)
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)
        os.mkdir(self.root_path)
        root_info_name='root_info.txt'
        file = open(os.path.join(self.root_path, root_info_name), 'w')
        file.write(self.scene_setting)
        file.close()

    def log_object_level(self,object_iter):
        self.object_path=os.path.join(self.root_path, f"object_{object_iter}")
        os.mkdir(self.object_path)

    def log_init_info(self,iter):
        self.iter_path=os.path.join(self.object_path, f"iter_{iter}")
        os.mkdir(self.iter_path)

    def log_action_info(self,vx,vy,vz,lab):
        action_info_name='action_info.txt'
        file = open(os.path.join(self.iter_path, action_info_name), 'w')
        file.write(f"{vx},{vy},{vz}\n")
        file.write(lab)
        file.close()

    def log_checkpoint(self,time_step,rigid_label,save=False):
        action_order_is_label=[]
        index=0
        for name in rigid_label:
            rigid_name=name[0]
            index+=1
            states=rigid_name.get_current_dynamic_state()
            position=states.position
            orientation=states.orientation
            dy_states=torch.cat([position,orientation],dim=0)
            action_order_is_label.append(dy_states.unsqueeze(0))
        action_order_is_label=torch.cat(action_order_is_label,dim=0)[:,:]
        if time_step == 0:
            self.init=action_order_is_label
        if time_step == 1:
            motion=torch.norm(action_order_is_label-self.init, dim=-1,p=2)
            self.last_motion = motion
        if time_step >= 2:
            motion=torch.norm(action_order_is_label-self.init, dim=-1,p=2)
            motion_compare = torch.cat([self.last_motion[:,None],motion[:,None]],dim=-1)
            self.last_motion = torch.max(motion_compare,dim=-1)[0].reshape(-1)
        if save:
            self.last_motion[self.last_motion>1e-1] = 1
            self.last_motion[self.last_motion<=1e-1] = 0
            check_point_path=os.path.join(self.iter_path, f"motion_checkpoint")
            np.save(check_point_path,self.last_motion.clone().cpu().numpy())


    
    def save_init_point(self):
        file_name="points"+".npy"
        file_path=os.path.join(self.root_path,file_name)
        np.save(file_path,self.camera.get_current_frame()['pointcloud']["data"])

        seg = self.camera.get_current_frame()['semantic_segmentation']["info"]['idToLabels']
        mask = self.camera.get_current_frame()['semantic_segmentation']["data"]
        tran_mask=np.zeros_like(mask)*100
        visiable_list=check_mask_visiable(mask,self.mask_tolerance,self.mask_max_num)
        for i in visiable_list:
            if seg[f"{i}"]["class"] == 'BACKGROUND' or seg[f"{i}"]["class"] == 'UNLABELLED':
                tran_mask[mask==i] = 100
            else:
                tran_mask[mask==i] = int(seg[f"{i}"]["class"])
        file_name="mask_piece"+".npy"
        file_path=os.path.join(self.root_path,file_name)
        np.save(file_path,tran_mask)

        file_name="mask_line"+".npy"
        file_path=os.path.join(self.root_path,file_name)
        points = self.camera.get_current_frame()['pointcloud']['data']
        points = torch.from_numpy(points).to(self.device)
        loc=self.camera.get_image_coords_from_world_points(points)
        mask_line=find_mask(loc,tran_mask)
        np.save(file_path,mask_line)
    
def find_mask(point_loc,mask):
    mask_list=[]
    for i in range(point_loc.shape[0]):
        init_x=point_loc[i,0].item()
        init_y=point_loc[i,1].item()
        x=int(init_x)
        y=int(init_y)
        mask_ft=mask[y,x]
        mask_list.append(mask_ft)
    return np.array(mask_list)


def random_select(part_point,num):
    total_num=part_point.shape[0]
    indices=random.sample(range(total_num),num)
    part_point=part_point[indices,:]
    return part_point

def find_tran_mask(mask,seg):
    tran_mask=np.zeros_like(mask)*100
    seen_num=len(seg)
    for i in range(seen_num):
        if seg[f"{i}"]["class"] == 'BACKGROUND' or seg[f"{i}"]["class"] == 'UNLABELLED':
            tran_mask[mask==i] = 100
        else:
            tran_mask[mask==i] = int(seg[f"{i}"]["class"])
    return tran_mask


def keep_stability(world,camera):
    world.reset()
    camera.initialize()
    camera.post_reset()
    for _ in range(20):
        world.play()


def set_init_pos(world,rigid_label,state_list,device):
    for i in range(len(rigid_label)):
        rigid_label[i][0].set_world_pose(position=state_list[i][0],orientation=state_list[i][1])
    for i in range(0):
        world.play()
        for j in rigid_label:
            target_vel = torch.tensor([[0.,0.,0., 0.,0.,0.]]).to(device)
            j[0]._rigid_prim_view.set_velocities(target_vel)

def import_objects(world, structure_info, transform=None ,root_name=None, root_path=None):
    rigid_num = len(structure_info["rigid"].keys())
    var=[0] * rigid_num
    
    for _, key in enumerate(structure_info["rigid"].keys()):
        rigid_info=structure_info["rigid"][key]

        if root_path is None:
            prim_path = rigid_info["prim_path"]
        else:
            prim_path = os.path.join(root_path,key)
        if root_name is None:
            name = key
        else:
            name = root_name + "_" + key

        add_reference_to_stage(
            usd_path=rigid_info["usd_path"],
            prim_path=prim_path,
        )
        
        index = int(key.split("_")[-1])
    
        geom=GeometryPrim(prim_path=prim_path,collision=True)
        geom.set_collision_approximation("convexDecomposition")

        rigid=RigidPrim(prim_path=prim_path,name=name, mass=0.01)
        rigid.enable_rigid_body_physics()


        xform=XFormPrim(prim_path=prim_path)
        if transform is not None:
            position = np.array(rigid_info["position"]) + transform
            xform.set_world_pose(position=position,orientation=rigid_info["orientation"])
            xform.set_local_scale(xform.get_local_scale()/TRANSFORM_SCALE)
        else:
            xform.set_world_pose(position=rigid_info["position"],orientation=rigid_info["orientation"])
            xform.set_local_scale(xform.get_local_scale()/TRANSFORM_SCALE)


        prim=get_prim_at_path(prim_path)
        add_update_semantics(prim=prim, semantic_label=f"{index}")

        var[index]=[rigid,prim,geom]

        world.scene.add(rigid)

    return var

def save_state_list(state_list,path):
    json_data=json.dumps(state_list)
    with open(path,"w") as f:
        f.write(json_data)

def save_target_list(target_list,path):
    target_name_list=list()
    for target in target_list:
        target_name_list.append(target.name)
    json_data=json.dumps(target_name_list)
    with open(path,"w") as f:
        f.write(json_data)

def read_json(path):
    with open(path) as f:
        data=json.load(f)
    return data

def get_init_pose(my_world,camera,rigid_label,phy):
    my_world.reset()
    camera.initialize()
    camera.post_reset()
    for i in range(100):
        my_world.step(render=True)

    state_list=[]
    for i in range(len(rigid_label)):
        rigid=rigid_label[i][0]
        pos,ori=rigid.get_world_pose()
        pos_list=pos.cpu().tolist()
        ori_list=ori.cpu().tolist()
        state_list.append((rigid.name,pos_list,ori_list))
    return state_list

def get_target_object(my_world,camera,rigid_label,state_list,device):
    target_list=[]
    if False:
        mc=move_checker()
        for rigid_iter in range(len(rigid_label)):
            move=False
            set_init_pose(my_world,camera,rigid_label,state_list)
            for j in range(36):
                my_world.step(render=True)
                target_vel = torch.tensor([[0.,0.,0.6,0.,0.,0.]]).to(device)
                rigid_label[rigid_iter][0]._rigid_prim_view.set_velocities(target_vel)
                rigid_label[rigid_iter][0].set_mass(100)
                move=mc.check_move(j,rigid_label,1,1)
                if move:
                    break
            if not move:
                target_list.append(rigid_label[rigid_iter][0])
            else:
                pass
    target_list.append(rigid_label[13][0])
    return target_list

class move_checker():
    def __init__(self):
        pass

    def check_move(self,time,rigid_label,bound,num:int):
        if time==0:
            self.init=np.zeros([len(rigid_label),7])
            for rigid_index in range(len(rigid_label)):
                self.init[rigid_index,:3]=rigid_label[rigid_index][0].get_current_dynamic_state().position
                self.init[rigid_index,3:]=rigid_label[rigid_index][0].get_current_dynamic_state().orientation
            return False
        if time != 0 and time % 3 == 0:
            self.final=np.zeros([len(rigid_label),7])
            for rigid_index in range(len(rigid_label)):
                self.final[rigid_index,:3]=rigid_label[rigid_index][0].get_current_dynamic_state().position
                self.final[rigid_index,3:]=rigid_label[rigid_index][0].get_current_dynamic_state().orientation
            move=np.linalg.norm(self.final-self.init,axis=-1)
            if np.count_nonzero(move>bound) > num:
                return True
            
def multi_import_scene(world,structure_info,env_line,grid):
    env_num=env_line**2
    rigid_label_list=[]
    for env in range(env_num):
        x_index=env%env_line
        y_index=env//env_line
        transform = np.array([x_index*grid, y_index*grid, 0])
        var_list = import_objects(world=world, structure_info=structure_info, 
                                  transform=transform, 
                                  root_name=f"env_{env}",
                                  root_path=f"/World/env_{env}")
        rigid_label_list.append(var_list)
    return rigid_label_list

def multi_set_init_pose(state_list,envs_line,device,rigid_label_list):
    for i in range(envs_line**2):
        x_index=i%envs_line
        y_index=i//envs_line
        step=torch.tensor([-2*y_index, -2*x_index, 0.]).to(device)
        for state in state_list:
            name="xform"+"_"+f"{i}"+"_"+state[0].split("_")[-1]
            rigid_idx = int(state[0].split("_")[-1])
            rigid_label_list[i][rigid_idx][-1].set_world_pose(position=torch.tensor(state[1]).to(device)+step,orientation=torch.tensor(state[2]).to(device))


def import_gripper(world, env_line, grid, gripper_path):
    gripper_list=[]
    env_num = env_line ** 2
    for env in range(env_num):
        x_index=env%env_line
        y_index=env//env_line
        transform = np.array([x_index*grid, y_index*grid, 0.2])
        gripper=Gripper(input_path=gripper_path, 
                        root_path = f"/World/gripper_{env}",
                        name=f"gripper_{env}", offset=transform)
        world.scene.add(gripper.rigid_prim)
        #gripper.apply_physics_material(physics_material)
        gripper_list.append(gripper)
    return gripper_list

class Gripper():
    def __init__(self, input_path, root_path, name, offset):
        self._root = root_path
        self.name=name
        add_reference_to_stage(usd_path=input_path,prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root
        #matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=offset,
            scale = np.array([100,100,100])
            #orientation=euler_angles_to_quat(rigid_config["orientation"]),
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexDecomposition")
        self.rigid_prim=RigidPrim(
            prim_path=mesh_path,
            name=self.name
        )
        self.rigid_prim.enable_rigid_body_physics()
        self.rigid_prim._rigid_prim_view.disable_gravities()
        #self.rigid_prim.set_mass(1e2)


class WrapFranka:
    def __init__(self,world,Position=torch.tensor,orientation=None,prim_path:str=None,robot_name:str=None,):
        self.world=world
        self._franka_prim_path=prim_path
        self._franka_robot_name=robot_name
        self.init_position=Position
        self._robot=Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name,position=Position,orientation=orientation)
        self.world.scene.add(self._robot)
        self._articulation_controller=self._robot.get_articulation_controller()
        self._controller=RMPFlowController(name="rmpflow_controller",robot_articulation=self._robot,physics_dt=1/240.0)
        self._kinematic_solover=KinematicsSolver(self._robot)
        self._pick_place_controller=PickPlaceController(name="pick_place_controller",robot_articulation=self._robot,gripper=self._robot.gripper)



    def initialize(self):
        self._controller.reset()
        #self._robot.initialize()


    def get_cur_ee_pos(self):
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        return ee_pos, R

    def get_current_position(self):
        position_left,orientation=self._robot.gripper.get_world_pose()
        position_right,_=self._robot._gripper_right.get_world_pose()
        return (position_left+position_right)/2,orientation
    
    def move(self,position,orientation=None):
        #position,orientation=self.input_filter(position,orientation)
        #orientation=np.array([0.,1.,0.,0.])
        position=position.reshape(-1)
        actions = self._controller.forward(
            target_end_effector_position=position,
            target_end_effector_orientation=orientation
            )
        #self._robot.set_joint_positions(joint)

        action_info=self._articulation_controller.apply_action(actions,True)
        #self._articulation_controller.apply_discrete_action(action_info)
 
    def open(self):
        self._robot.gripper.open()
    
    def close(self):
        self._robot.gripper.close()

    @staticmethod
    def interpolate(start_loc, end_loc, speed):
        start_loc = np.array(start_loc)
        end_loc = np.array(end_loc)
        dist = np.linalg.norm(end_loc - start_loc)
        chunks = dist // speed
        if chunks==0:
            chunks=1
        return start_loc + np.outer(np.arange(chunks+1,dtype=float), (end_loc - start_loc) / chunks)
    
    def position_reached(self, target,thres=0.01):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        pos_diff = np.linalg.norm(ee_pos- target)
        if pos_diff < thres:
            return True
        else:
            return False 


class multi_move_checker():
    def __init__(self,env_num,rigid_label):
        self.env_num=env_num
        self.init=np.zeros([env_num,len(rigid_label),7])
        self.final=np.zeros([env_num,len(rigid_label),7])
        self.move_list=[True]*env_num
        self.rigid_label=rigid_label

    def check_move(self,time,bound):
        if time==0:
            for env_index in range(self.env_num):
                for rigid_index in range(len(self.rigid_label[env_index])):
                    self.init[env_index,rigid_index,:3]=self.rigid_label[env_index][rigid_index][0].get_current_dynamic_state().position
                    self.init[env_index,rigid_index,3:]=self.rigid_label[env_index][rigid_index][0].get_current_dynamic_state().orientation
        if time > 0:
            for env_index in range(self.env_num):
                for rigid_index in range(len(self.rigid_label[env_index])):
                    self.final[env_index,rigid_index,:3]=self.rigid_label[env_index][rigid_index][0].get_current_dynamic_state().position
                    self.final[env_index,rigid_index,3:]=self.rigid_label[env_index][rigid_index][0].get_current_dynamic_state().orientation
            move=np.linalg.norm(self.final-self.init,axis=-1)
            for env_index in range(self.env_num):
                if np.count_nonzero(move[env_index]>bound) > 1:
                    self.move_list[env_index]=False
        return np.array(self.move_list)
    
def generate_random_euler_angles(n):  
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n, 3))  
    return euler_angles 

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

def find_all_envs_path(root_dir, index):
    path_list = [os.path.join(root_dir,"train",f"env_{index}")]
    full_pts_list=[]
    for path in path_list:
        pts_path = os.path.join(path,"points.npy")
        mask_line_path = os.path.join(path,"mask_line.npy")
        mask_piece_path = os.path.join(path,"mask_piece.npy")        
        full_pts_list.append([pts_path,mask_line_path,mask_piece_path])
    return full_pts_list, path_list

class PreCache():
    def __init__(self,path_list,points_num,):
        super().__init__()
        self.path_list=path_list
        self.num=points_num
        self.time=dict()
        if torch.cuda.is_available():
            self.device="cuda:1"
        else:
            raise

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

def convert_pts(envs_wrapper_list, save_path):
    pts_path = envs_wrapper_list[0][0]
    mask_line_path = envs_wrapper_list[0][1]
    mask_mat_path = envs_wrapper_list[0][2]
    pts = np.load(pts_path)
    mask_line = np.load(mask_line_path)
    mask_mat = np.load(mask_mat_path)
    np.save(os.path.join(save_path, "points.npy"),pts)
    np.save(os.path.join(save_path, "mask_line.npy"),mask_line)
    np.save(os.path.join(save_path, "mask_mat.npy"),mask_mat)


def get_target_pts(root_dir, index, save_path):
    root_dir = os.path.join(root_dir, "data_interaction")
    envs_wrapper_list, envs_root_list=find_all_envs_path(root_dir,index=index)
    point=PreCache(envs_wrapper_list,10240)
    point.preprocess()
    point.reorder()
    point.expand()

    convert_pts(envs_wrapper_list, save_path)


    batched_partpoint_list=get_part_point_batch(point.data,point.expand_mask_line)

    k_exp=4

    point_cloud = None
    grasp_target_list=[]
    for env_idx, env_root_path in enumerate(envs_root_list):
        object_root_path_list = glob.glob(os.path.join(env_root_path, "object_*"))
        for _, object_root_path in enumerate(object_root_path_list):
            obj_idx = int(object_root_path.split("_")[-1])
            iter_path_list = glob.glob(os.path.join(object_root_path, "iter_*"))
            motion_list=list()
            for iter_path in iter_path_list:
                if not os.path.exists(os.path.join(iter_path,"motion_checkpoint.npy")):
                    drop=True
                    break
                else:
                    motion_path = os.path.join(iter_path,"motion_checkpoint.npy")
                    motion = np.load(motion_path)
                    direction_score=np.sum(motion) - 1
                    motion_list.append(direction_score)
            if min(motion_list) == 0:
                grasp_target_list.append(obj_idx)

    grasp_pts = batched_partpoint_list[0][grasp_target_list,:,:].cpu().numpy()
    return grasp_pts, grasp_target_list

def sample_orientation(pre_defined_index = None):
    if pre_defined_index is not None:
        candidate_list=[
            np.array([0, 0, 0]),
            ###
            np.array([-np.pi/3, 0, 0]),
            np.array([-np.pi/6, 0, 0]),
            np.array([np.pi/6, 0, 0]),
            np.array([np.pi/3, 0, 0]),
            ###
            np.array([0, 0, -np.pi/2]),
            np.array([0, 0, -np.pi/3]),
            np.array([0, 0, -np.pi/6]),
            np.array([0, 0, np.pi/6]),
            np.array([0, 0, np.pi/3]),
            ####
            np.array([0, -np.pi/3, 0]),
            np.array([0, -np.pi/6, 0]),
            np.array([0, np.pi/6, 0]),
            np.array([0, np.pi/3, 0]),
            #### ...
        ]
        return candidate_list[pre_defined_index]
    else:
        z_theta = np.random.uniform(low = -np.pi, high= np.pi)
        y_theta = np.random.uniform(low = -np.pi/2, high= np.pi/2)
        x_theta = np.random.uniform(low = -np.pi/2, high= np.pi/2)
        return np.array([x_theta, y_theta, z_theta])
    
def get_rotation_axis(eular_angle):
    R = euler_to_rot_matrix(eular_angle)
    z_form = np.array([[0],[0],[-0.2]])
    z_transformed = R @ z_form
    return z_transformed.reshape(-1)

def euler_to_rot_matrix(eular_angle):  
    roll, pitch, yaw = eular_angle
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],  
                   [np.sin(yaw), np.cos(yaw), 0],  
                   [0, 0, 1]])  
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],  
                   [0, 1, 0],  
                   [-np.sin(pitch), 0, np.cos(pitch)]])  
    Rx = np.array([[1, 0, 0],  
                   [0, np.cos(roll), -np.sin(roll)],  
                   [0, np.sin(roll), np.cos(roll)]])  
    R = np.dot(Rz, np.dot(Ry, Rx))  
    return R  


class Gripper_controller():
    def __init__(self, gripper_list, envs_line, device, sim_dt, save_path, target_idx) -> None:
        self.gripper_list = gripper_list
        self.envs_line = envs_line
        self.envs_num = envs_line ** 2
        self.device = device
        self.sim_dt = sim_dt
        self.RUNNING_STEP = 20
        self.data=list()
        self.save_dir = save_path
        self.target_idx = target_idx

    def set_gripper_pose(self, grid, env_line, target_point):
        self.orientation=list()
        self.start_position=list()
        self.move_vec=list()
        self.target_pts=list()
        self.vis_list=list()
        self.init_target_pts=list()
        for env in range(self.envs_num):
            x_index=env % env_line
            y_index=env // env_line
            transform = np.array([x_index*grid, y_index*grid, 0.])

            gripper = self.gripper_list[env]
            orientation = sample_orientation(pre_defined_index=None)
            offset = get_rotation_axis(orientation)
            orientation = euler_angles_to_quat(orientation)
            start_position = target_point[env] - offset + transform

            # VisualCuboid(prim_path=f"/World/v1/ev_{env}",name=f"v1_{env}",scale=np.array([0.015,0.015,0.015]),
            #             position=start_position,color=np.array([1,0,0]),
            #             orientation =orientation)
            # VisualCuboid(prim_path=f"/World/v2/ev_{env}",name=f"v2_{env}",scale=np.array([0.015,0.015,0.015]),
            #             position=target_point[env]+transform,color=np.array([1,0,0]),
            #             orientation =orientation)
            gripper.rigid_form.set_world_pose(orientation=orientation,
                                            position=start_position)
            self.orientation.append(orientation)
            self.start_position.append(start_position)
            self.move_vec.append(offset)
            self.target_pts.append(target_point[env] + transform - 0.1 * offset)
            self.init_target_pts.append(target_point[env])

    def create_buffer(self):
        self.buffer=dict()
        self.buffer["result"]=list()
        self.buffer["orientation"]=list()

    def move_toward(self):
        for i in range(self.envs_num):
            gripper = self.gripper_list[i]
            move_vec = self.move_vec[i]
            vel = (move_vec/self.RUNNING_STEP)/(self.sim_dt)
            vel = torch.from_numpy(vel[None,:]).to(self.device).to(torch.float32)
            rad_padding = torch.zeros_like(vel)
            full_vel = torch.cat([vel, rad_padding],dim=-1)
            gripper.rigid_prim._rigid_prim_view.set_velocities(full_vel)

    def stop(self):
        for i in range(self.envs_num):
            gripper = self.gripper_list[i]
            vel = torch.zeros([1,6]).to(self.device).to(torch.float32)
            gripper.rigid_prim._rigid_prim_view.set_velocities(vel)

    def check_results(self):
        result=list()
        for i in range(self.envs_num):
            gripper = self.gripper_list[i]
            position, orientaion = gripper.rigid_prim._rigid_prim_view.get_world_poses()
            position = position.cpu().numpy().reshape(-1)
            orientaion = orientaion.cpu().numpy().reshape(-1)
            target_position = self.target_pts[i]
            target_orientation = self.orientation[i]
            error = np.linalg.norm(target_position-position,ord=2) + np.linalg.norm(target_orientation-orientaion,ord=2)
            if error < 1e-2:
                result.append(np.array([1]))
            else:
                result.append(np.array([0]))
        result = np.concatenate(result)[:,None]
        orientaion = np.concatenate(self.orientation).reshape(-1,4)
        
        self.buffer["result"].append(result)
        self.buffer["orientation"].append(orientaion[:,None,:])
        

    def cat_buffer(self):
        target_pts = np.concatenate(self.init_target_pts).reshape(-1,3)
        self.buffer["pts"]=target_pts
        key = "result"
        self.buffer[key]=np.concatenate(self.buffer[key],axis=1)
        key = "orientation"
        self.buffer[key]=np.concatenate(self.buffer[key],axis=1)
        self.data.append(self.buffer)

    def log_results(self):
        result = list()
        orientation = list()
        pts = list()
        for data_dict in self.data:
            result.append(data_dict["result"])
            orientation.append(data_dict["orientation"])
            pts.append(data_dict["pts"])
        result = np.concatenate(result,axis=0)
        orientation = np.concatenate(orientation,axis=0)
        pts = np.concatenate(pts,axis=0)

        pts_path = os.path.join(self.save_dir,f"rigid_{self.target_idx}.npy")
        result_path = os.path.join(self.save_dir, f"rigid_{self.target_idx}_result.npy")
        pose_path = os.path.join(self.save_dir, f"rigid_{self.target_idx}_pose.npy")
        np.save(pts_path,pts)
        np.save(result_path,result)
        np.save(pose_path,orientation)

