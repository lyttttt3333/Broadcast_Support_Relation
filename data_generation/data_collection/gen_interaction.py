from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import os
from generation_utils import *
from omni.isaac.sensor import Camera
from omni.isaac.core import World

class clutter_scene():
    def __init__(self, input_args) -> None:
        self.indice=input_args.indice
        self.dir_path=input_args.dir_path
        self.save_path = os.path.join(input_args.dir_path,"data_interaction")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.input_path = os.path.join(input_args.dir_path, "data_scene")

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

    def build_env(self):
        self.world= World(stage_units_in_meters=self.stage_unit,
                           backend=self.backend,
                           device=self.device)
        self.world.scene.add_default_ground_plane()
        self.world.set_simulation_dt(physics_dt=self.physics_dt,rendering_dt=self.rendering_dt)
        self.phy=self.world.get_physics_context()
        self.phy.set_gravity(self.gravity)
        structure_path=os.path.join(self.dir_path,"data_scene",f"{self.indice}.yaml")
        self.rigid_label=import_objects(world=self.world, structure_info=read_yaml(structure_path))

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array(self.camera_position),
            orientation=np.array(self.camera_orientation),
            resolution=(512,512)
        )
        self.camera.initialize()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_pointcloud_to_frame(include_unlabelled=False)

        self.state_list=keep_stability(world=self.world,
                                        camera=self.camera)

        self.logger=LogInteraction(self.indice,self.save_path,structure_path,self.camera,device=self.device)

    def simulation(self):
        self.logger.save_init_point()

        for rigid_index in range(len(self.rigid_label)):
            self.logger.log_object_level(rigid_index)
            rigid_to_execute=self.rigid_label[rigid_index][0]
            rigid_to_execute.set_mass(100000)

            for iter in range(self.max_direction_sample_num):
                self.world.reset()
                self.camera.initialize()
                self.camera.post_reset()
                self.logger.log_init_info(iter)
                velocity,vel1,vel2,vel3=random_select_orientation(self.device)
                self.logger.log_action_info(vel1,vel2,vel3,str(rigid_index))
                self.logger.log_checkpoint(time_step=0,rigid_label=self.rigid_label)
                for time_step in range(self.max_simulation_time):
                    rigid_to_execute._rigid_prim_view.set_velocities(velocity)
                    self.world.step()
                    self.logger.log_checkpoint(time_step=time_step,rigid_label=self.rigid_label)
                self.logger.log_checkpoint(time_step=time_step,rigid_label=self.rigid_label,save=True)
                self.world.stop()

            rigid_to_execute.set_mass(0.01)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--indice", type=int, default=None,
                           help="the scene indice for simulation")
    argparser.add_argument("--dir_path", type=str, default=None,
                           help="the path for saving the scenes configurations")
    argparser.add_argument("--save_path", type=str, default=None,
                           help="the path to save the interaction data")
    argparser.add_argument("--physics_dt", type=float, default=1/20)
    argparser.add_argument("--rendering_dt", type=float, default=1/20)
    argparser.add_argument("--gravity", type=float, default=-9.8)
    argparser.add_argument("--max_direction_sample_num", type=int, default=6)
    argparser.add_argument("--max_simulation_time", type=int, default=36)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--backend", type=str, default="torch")
    argparser.add_argument("--stage_unit", type=float, default=0.01)
    argparser.add_argument("--camera_position", type=list, default=[0,0,5])
    argparser.add_argument("--camera_orientaion", type=list, default=[1,0,0,0])

    args= argparser.parse_args()

    scene=clutter_scene(args)
    scene.build_env()
    scene.simulation()
    simulation_app.close()
