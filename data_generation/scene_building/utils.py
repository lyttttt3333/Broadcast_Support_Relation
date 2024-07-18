import random
import numpy as np
import yaml
import torch
import copy

from omni.isaac.core import World, PhysicsContext, SimulationContext
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims import GeometryPrim,RigidPrim,XFormPrim
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, create_prim, get_prim_at_path,get_all_matching_child_prims,get_first_matching_child_prim,get_prim_type_name,get_prim_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.cloner import GridCloner
from pxr import UsdGeom, Gf
from omni.isaac.sensor import Camera


TRANSFORM_SCALE=88


class allocate_space():
    def __init__(self,num_prim,low_bound,high_bound) -> None:
        self.all=num_prim
        self.time=0
        self.order=np.random.permutation(num_prim) 
        self.low_bound=low_bound
        self.high_bound=high_bound 
        self.interval=(high_bound-low_bound)/num_prim

    def add_primitive(self):
        order_of_item=self.order[self.time]
        height=np.random.normal(self.low_bound+(order_of_item+0.5)*self.interval)
        self.time+=1
        return height
    
def read_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return yaml_data

def random_orientation():
    return np.random.normal(from_er_to_quat([0,0,0]))
    
def from_er_to_quat(er):
    a_1=er[0]
    a_2=er[1]
    a_3=er[2]
    q1=np.cos(a_1/2)*np.cos(a_2/2)*np.cos(a_3/2)+np.sin(a_1/2)*np.sin(a_2/2)*np.sin(a_3/2)
    q2=np.sin(a_1/2)*np.cos(a_2/2)*np.cos(a_3/2)-np.cos(a_1/2)*np.sin(a_2/2)*np.sin(a_3/2)
    q3=np.cos(a_1/2)*np.sin(a_2/2)*np.cos(a_3/2)+np.sin(a_1/2)*np.cos(a_2/2)*np.sin(a_3/2)
    q4=np.cos(a_1/2)*np.cos(a_2/2)*np.sin(a_3/2)-np.sin(a_1/2)*np.sin(a_2/2)*np.cos(a_3/2)
    return [q1,q2,q3,q4]

def get_primitive_rigid_dict(min_num,max_num,cate_dist,cate_path,init_high_bound,init_low_bound,random_thm):
    path_dict_list=[]
    for item in cate_path:
        path_dict=read_yaml(item)
        path_dict_list.append(path_dict)

    primitive_rigid_dict={}
    primitive_rigid_dict["rigid"]={}
    index_whole_dict=0

    num_prim=random.randint(min_num,max_num)
    space_manager=allocate_space(num_prim=num_prim,
                                 high_bound=init_high_bound,
                                 low_bound=init_low_bound)
    
    cate_dist=np.array(cate_dist)
    cate_dist=cate_dist/np.sum(cate_dist)
    num_whole_cat=10
    cate_dist=np.random.multinomial(num_prim,cate_dist)
    print(cate_dist)

    for i in range(cate_dist.shape[0]):
        path_dict=path_dict_list[i]
        num_in_cat=cate_dist[i]
        num_whole_cat=len(path_dict["online"])
        dist_in_cat=np.random.multinomial(num_in_cat,np.ones(num_whole_cat)/num_whole_cat)
        for j in range(len(dist_in_cat)):
            if dist_in_cat[j]==0:
                continue
            for _ in range(dist_in_cat[j]):
                index_in_cat=j
                height=space_manager.add_primitive()
                rigid = {
                    "usd_path": path_dict["online"][index_in_cat],
                    "position": torch.tensor(random_pos(height,random_thm),device="cuda:0").cpu(),
                    "orientation": torch.tensor(random_orientation(),device="cuda:0").cpu(),
                    "prim_path":f"/World/envs/env_0/rigid_{index_whole_dict}",
                }
                primitive_rigid_dict["rigid"][f"rigid_{index_whole_dict}"]=rigid
                index_whole_dict+=1
    return primitive_rigid_dict

def judge_dataset(input, central_scale, edge_scale, judge_ratio_central, judge_ratio_out):
    dist=input-np.mean(input,axis=0)
    central=np.linalg.norm(dist)-central_scale
    in_range=np.sum(central<0)
    out=np.linalg.norm(dist)-edge_scale
    out_range=np.sum(out>0)
    if in_range/input.shape[0] >= judge_ratio_central and out_range/input.shape[0] <= judge_ratio_out:
        return True
    else:
        return False

def generate_id():
    return random.randint(1000,9999)

def apply_random(primitive_for_this_iter:dict,random_thm):
    for item in primitive_for_this_iter["rigid"].keys():
        rigid_info=primitive_for_this_iter["rigid"][item]
        primitive_for_this_iter["rigid"][item]["position"]=random_pos(rigid_info["position"][-1],random_thm)
        primitive_for_this_iter["rigid"][item]["orientation"]=np.random.normal(rigid_info["orientation"])
    return copy.deepcopy(primitive_for_this_iter)

def random_pos(input,length):
    theta=np.random.uniform(0,2*np.pi)
    height=input
    return [length*np.cos(theta)/TRANSFORM_SCALE,length*np.sin(theta)/TRANSFORM_SCALE,height]


def write_end_info(scene_info,save_path):
    print(save_path)
    with open(save_path, 'w') as f:
        f.write(yaml.dump(scene_info, allow_unicode=True))
    return True

def read_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return yaml_data

def is_mesh(input_usd):
    if get_prim_type_name(input_usd)=="Mesh":
        return True
    else:
        return False

def change_states(geom_list,current_step,force=False):
    if current_step % 100 == 0 and current_step % 200 == 0:
        for geom_var in geom_list:
            geom_var.set_collision_approximation("convexDecomposition")
    if current_step % 100 == 0 and current_step % 200 != 0:
        for geom_var in geom_list:
            geom_var.set_collision_approximation("convexHull")
    if force:
        for geom_var in geom_list:
            geom_var.set_collision_approximation("convexDecomposition")

def place_init_env(primitive_rigid_dict):
    
    rigid_var_list=[]
    mesh_var_list=[]
    xform_var_list=[]
    for item in primitive_rigid_dict["rigid"].keys():
        rigid_info=primitive_rigid_dict["rigid"][item]
        add_reference_to_stage(
            usd_path=rigid_info["usd_path"],
            prim_path=rigid_info["prim_path"],
        )

        mesh_list=get_all_matching_child_prims(rigid_info["prim_path"],is_mesh)
        for mesh in mesh_list:
            mesh_prim_path=get_prim_path(mesh)
            mesh_prim=GeometryPrim(prim_path=mesh_prim_path,collision=True)
            mesh_prim.set_collision_approximation("convexDecomposition")
            mesh_var_list.append(mesh_prim)

        rigid_prim=RigidPrim(prim_path=rigid_info["prim_path"])
        rigid_prim.enable_rigid_body_physics()
        rigid_prim.set_mass(1)
        rigid_var_list.append(rigid_prim)

        xform_prim=XFormPrim(prim_path=rigid_info["prim_path"])
        xform_prim.set_world_pose(position=np.array(rigid_info["position"]),orientation=rigid_info["orientation"])
        xform_prim.set_local_scale(xform_prim.get_local_scale()/TRANSFORM_SCALE)
        xform_var_list.append(xform_prim)
    return rigid_var_list, mesh_var_list, xform_var_list