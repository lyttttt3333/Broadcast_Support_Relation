from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import os
import shutil
from generation_utils import *
from omni.isaac.sensor import Camera
from omni.isaac.core import World




class clutter_scene():
    def __init__(self, input_args) -> None:

        self.pts_dir = "/home/sim/.local/share/ov/pkg/isaac_sim-2023.1.0/standalone_examples/api/omni.isaac.kit/data_interaction"
        self.indice=input_args.indice
        self.dir_path=input_args.dir_path
        self.save_path=input_args.save_path

        self.physics_dt=input_args.physics_dt
        self.rendering_dt=input_args.rendering_dt
        self.gravity=input_args.gravity
        self.max_direction_sample_num=input_args.max_direction_sample_num
        self.max_simulation_time=input_args.max_simulation_time

        self.device=input_args.device
        self.backend=input_args.backend
        self.stage_unit=input_args.stage_unit
        self.camera_position=input_args.camera_position
        self.camera_orientation=input_args.camera_orientaion
        self.env_line = 2
        self.env_num = self.env_line ** 2
        self.grid = 2
        self.gripper_path = "omniverse://localhost/Users/sim/frank_gripper_rigid.usd"

        self.save_dir = os.path.join(self.dir_path, "data_affordance")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, f"{self.indice}")
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.mkdir(self.save_path)

        

    def build_env(self):
        self.world= World(stage_units_in_meters=self.stage_unit,
                           backend=self.backend,
                           device=self.device)
        self.world.scene.add_default_ground_plane()
        self.world.set_simulation_dt(physics_dt=self.physics_dt,rendering_dt=self.rendering_dt)
        self.phy=self.world.get_physics_context()
        self.phy.set_gravity(self.gravity)
        structure_path=os.path.join(self.dir_path,"data_scene",f"{self.indice}.yaml")
        self.rigid_label=multi_import_scene(world=self.world,
                                             structure_info=read_yaml(structure_path),
                                             env_line=self.env_line,
                                             grid=self.grid)

        self.gripper_list=import_gripper(world=self.world,env_line=self.env_line,grid=self.grid,gripper_path=self.gripper_path)



        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array(self.camera_position),
            orientation=np.array(self.camera_orientation),
            resolution=(512,512)
        )
        self.camera.initialize()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_pointcloud_to_frame(include_unlabelled=False)

        self.logger=LogInteraction(self.indice,self.save_path,structure_path,self.camera,device=self.device)
        self.target_pts, self.target_rigid = get_target_pts(root_dir=self.dir_path, index=self.indice, save_path = self.save_path) 

        

    def simulation(self):
        
        #for rigid_idx in range(len(self.target_rigid)):
        for rigid_idx in range(2):
            target_idx = self.target_rigid[rigid_idx]
            rigid_pts = self.target_pts[rigid_idx]
            batch_num = rigid_pts.shape[0]//self.env_num

            controller = Gripper_controller(gripper_list=self.gripper_list, 
                                            envs_line=self.env_line, 
                                            device=self.device, 
                                            sim_dt=self.physics_dt,
                                            save_path=self.save_path,
                                            target_idx=int(target_idx))
            batch_num = 2
            for batch_idx in range(batch_num):

                batch_pts = rigid_pts[self.env_num * batch_idx: self.env_num * (batch_idx + 1)]

                controller.create_buffer()

                for pose_sample_idx in range(6):

                    controller.set_gripper_pose(grid=self.grid,env_line=self.env_line,
                                                target_point=batch_pts)
                    self.world.reset()
                    for _ in range(18):
                        controller.move_toward()
                        self.world.step()

                    for _ in range(5):
                        controller.stop()
                        self.world.step()

                    controller.check_results()
                    self.world.stop()

                controller.cat_buffer()

            controller.log_results()




if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--indice", type=int, default=1436,
                           help="the scene indice for simulation")
    argparser.add_argument("--dir_path", type=str, default="/home/sim/.local/share/ov/pkg/isaac_sim-2023.1.0/standalone_examples/api/omni.isaac.kit/data",
                           help="the path for saving the scenes configurations")
    argparser.add_argument("--save_path", type=str, default="/home/sim/.local/share/ov/pkg/isaac_sim-2023.1.0/standalone_examples/api/omni.isaac.kit/data_interaction",
                           help="the path to save the interaction data")
    argparser.add_argument("--physics_dt", type=float, default=1/20)
    argparser.add_argument("--rendering_dt", type=float, default=1/20)
    argparser.add_argument("--gravity", type=float, default=-9.8)
    argparser.add_argument("--max_direction_sample_num", type=int, default=6)
    argparser.add_argument("--max_simulation_time", type=int, default=36)
    argparser.add_argument("--device", type=str, default="cuda:1")
    argparser.add_argument("--backend", type=str, default="torch")
    argparser.add_argument("--stage_unit", type=float, default=0.01)
    argparser.add_argument("--camera_position", type=list, default=[0,0,5])
    argparser.add_argument("--camera_orientaion", type=list, default=[1,0,0,0])

    args= argparser.parse_args()

    scene=clutter_scene(args)
    scene.build_env()
    scene.simulation()
    simulation_app.close()
