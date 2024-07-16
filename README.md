# Introduction

This folder includes the code for our paper, Broadcasting Support Relations Recursively from Local Dynamics for Object Retrieval in Clutters, especially about clutter environment building, data generation, module training and inference.

## About This Repository
    data_generation/        # contains data generation process

        scene_building/     

            gen_clutter_scene.py    # contains the code for building the clutter scenes

        data_collection/     
        
            gen_interation.py       # contains the code for collecting data for relation inference part
            gen_pre_grasp.py        # prepare for manipulation in simulator
            gen_grasp.py            # contains the code for collecting data for manipulation part

    model_training/         # contains code and scripts for training different modules

        utils/              # utils for training modules
        preprocess.py       # prepare the data for training
        train_LDP.py        # train Local Dynamics Predictor
        train_RDS.py        # train Retrieval Direction Scoring Module
        train_RDP.py        # train Retrieval Direction Proposal Module
        train_GPS.py        # train Grasp Pose Scoring Module
        train_GPP.py        # train Grasp Pose Proposal Module
        train_GA.py         # train Grasp Affordance Module
        build_CS.py         # construct the Broadcast and Clutter Solver framework for support relation inference 


## Dependencies

For module training, this code can be ran on Ubuntu 18.04 with Cuda 10.1, Python 3.7, PyTorch 1.7.0. 
For data collection, we use ISAAC SIM 23.1.1 and the built-in python environment.

## Generate Training Data

Before training the network, we need to collect a large set of clutter scenes in the simulator, using the script 

    python data_generation/scene_building/gen_clutter_scene.py --save_path --input_path

For the arguments of this script, `--input_path` for loading the objects model such as shapenet and `--save_path` for saving the generated scenes configurations in `{index}.yaml`.

Based on the scenes, we can collect interactions for training modules of relation inference. 

    python data_generation/data_collection/gen_interaction.py --save_path --input_path --indice

For the arguments of this script, `--input_path` and `--indice` for loading the clutter scenes configurations you have saved in `data_generation/scene_building/gen_clutter_scene.py` and `--save_path` for saving the generated interaction data for training mainly in `.npy`.

Second, for grasp modules, we first document which points to try by `data_generation/data_collection/gen_pre_grasp.py`. And then we can collection the grasp ground truth parallel, which can be run by

    python data_generation/data_collection/gen_grasp.py --robot_path ...

where a list of path is required, for example, `--robot_path` for the franka source file in ISAAC SIM. More information can be found in python script `data_generation/data_collection/gen_grasp.py`.

Generating enough offline interaction trials is necessary for a successful learning, and it may require many CPU hours for the data collection.  

## Modules Training

After the data generation, you can use `preprocess.py` to process the raw data into the formation of modules' inputs.

Then we can use the following scripts to train different modules. In the following commands, the `--input_path` means the save path for pre-process and the `--save_path` means the path to save the trained modules.

For `Local Dynamics Predictor`,

    python train_LDP.py --input_path --save_path

For `Retrieval Direction Scoring Module`,

    python train_RDS.py --input_path --save_path

For `Retrieval Direction Proposal Module`,

    python train_RDP.py --input_path --save_path

For `Grasp Pose Scoring Module`,

    python train_GPS.py --input_path --save_path

For `Grasp Pose Proposal Module`,

    python train_GPP.py --input_path --save_path

For `Grasp Affordance Module`,

    python train_GA.py --input_path --save_path

## Support Relation Inference

After that, we can construct the `Clutter Solver` based on the trained modules by, which can output the inferred support relations from a given scene. As the core part of this work, `Clutter Solver` does not need any additional training process but a list of trained modules, which can be run by

    python build_CS.py --modules_path --pointcloud_path...

where `--modules_path` is a list of paths which can lead to the trained modules and `--pointcloud_path...` arguments is the data for the given scene. More information can be found in the `help` of `build_CS.py`.



