from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
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



import argparse
import os
from utils import *


class clutter_scene():
    def __init__(self, input_args) -> None:
        self.dir_path=input_args.dir_path
        self.save_path = os.path.join(self.dir_path, "data_scene")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.input_path=input_args.input_path
        self.device=input_args.device

        self.physics_dt=input_args.physics_dt
        self.gravity=input_args.gravity

        self.init_high_bound=input_args.init_high_bound
        self.init_low_bound=input_args.init_low_bound
        self.min_num=input_args.min_num
        self.max_num=input_args.max_num
        self.random_thm=input_args.random_thm
        self.cate_dist=input_args.cate_distribution

        self.central_scale=input_args.central_scale
        self.edge_scale=input_args.edge_scale
        self.judge_ratio_central=input_args.central_ratio
        self.judge_ratio_out=input_args.out_ratio
        
        self.simulation_step=input_args.max_simulation_step
        

    def build_env(self):
        self.world = World(backend="numpy")
        self.world.scene.add_default_ground_plane()

        self.physics=PhysicsContext(device=self.device)
        self.physics.enable_gpu_dynamics(True)
        self.physics.enable_ccd(True)
        self.physics.enable_stablization(True)
        self.physics.set_gravity(self.gravity)
        self.sim=SimulationContext(physics_dt=self.physics_dt,backend="torch",device=self.device)

        self.primitive_rigid_dict=get_primitive_rigid_dict(min_num=self.min_num,
                                                           max_num=self.max_num,
                                                           cate_dist=self.cate_dist,
                                                           cate_path=self.input_path,
                                                           random_thm=self.random_thm,
                                                           init_high_bound=self.init_high_bound,
                                                           init_low_bound=self.init_low_bound,
                                                           )
        rigid_var_list, mesh_var_list, xform_var_list=place_init_env(primitive_rigid_dict=self.primitive_rigid_dict)
        self.rigid_var_list=rigid_var_list
        self.mesh_var_list=mesh_var_list
        self.xform_var_list=xform_var_list
    

    def simulation(self):
        print("################################ a new iter ################################")
        self.simulation_id=generate_id()
        self.input_rigid_dict=apply_random(self.primitive_rigid_dict,random_thm=self.random_thm)

        for index, item in enumerate(self.primitive_rigid_dict["rigid"].keys()):
            rigid_info=self.input_rigid_dict["rigid"][item]
            self.xform_var_list[index].set_world_pose(position=rigid_info["position"],orientation=rigid_info["orientation"])

        
        for current_step in range(self.simulation_step):
            self.sim.play()
            change_states(self.mesh_var_list,current_step)
        change_states(self.mesh_var_list,current_step,force=True)
        for _ in range(100):
            self.sim.play()
            

    def document(self,iteration):
        final_state=[]  
        output_rigid_dict=self.input_rigid_dict.copy()
        output_rigid_dict["end_id"]=str(iteration)
        for index, item in enumerate(self.input_rigid_dict["rigid"].keys()):
            state=self.rigid_var_list[index].get_current_dynamic_state()
            output_rigid_dict["rigid"][item]["position"]=state.position.tolist()
            output_rigid_dict["rigid"][item]["orientation"]=state.orientation.tolist()
            final_state.append(state.position)
        final_state=np.concatenate(final_state,axis=0)
        self.sim.stop()
        #result=judge_dataset(final_state,central_scale=self.central_scale,edge_scale=self.edge_scale,
        #                     judge_ratio_central=self.judge_ratio_central,judge_ratio_out=self.judge_ratio_out)
        print("Valid Result")
        write_end_info(output_rigid_dict,os.path.join(self.save_path,f"{self.simulation_id}.yaml"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='add arguments to build clutter solver')

    INPUT_PATH=["/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/book_big.yaml",
                "/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/clock.yaml",
                "/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/frame.yaml",
                "/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/desk_bottle.yaml",
                "/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/sc.yaml",
                "/home/sim/correspondence/safe_manipulation/shapes_pool/kitchen_cat/book_small.yaml"]

    parser.add_argument('--input_path', type=str,
                        default=INPUT_PATH,
                        help="the path for object mesh")
    parser.add_argument('--cate_distribution', type=list,
                        default=[1,1,1,3,1,6],
                        help="the h")
    parser.add_argument('--dir_path', type=str,
                        default="/home/sim/.local/share/ov/pkg/isaac_sim-2023.1.0/standalone_examples/api/omni.isaac.kit/data",
                        help="the path to save clutter configuration")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--physics_dt', type=float, default=1/20)
    parser.add_argument('--gravity', type=float, default=-9.8)
    parser.add_argument('--init_high_bound', type=float, default=0.5)
    parser.add_argument('--init_low_bound', type=float, default=3)
    parser.add_argument('--min_num', type=int, default=10)
    parser.add_argument('--max_num', type=int, default=20)
    parser.add_argument('--random_thm', type=int, default=25)
    parser.add_argument('--central_scale', type=int, default=35)
    parser.add_argument('--edge_scale', type=int, default=50)
    parser.add_argument('--central_ratio', type=float, default=0.3)
    parser.add_argument('--out_ratio', type=float, default=0.3)
    parser.add_argument('--max_iteration', type=int, default=100)
    parser.add_argument('--max_simulation_step', type=int, default=500)



    args = parser.parse_args()

    scene=clutter_scene(args)
    scene.build_env()
    for index in range(args.max_iteration):
        scene.simulation()
        scene.document(iteration=index)
    simulation_app.close()