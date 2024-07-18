# Broadcasting Support Relations Recursively from Local Dynamics for Object Retrieval in Clutters 

![Overview](/images/teaser.png)

## about this paper
This paper has been accepted by RSS 2024!

Our team: 
[Yitong Li*](https://lyttttt3333.github.io/YitongLi.github.io/),
[Ruihai Wu*](https://warshallrho.github.io/),
[Haoran Lu](https://openreview.net/profile?id=~Haoran_Lu2),
[Chuanruo Ning](https://tritiumr.github.io/),
[Yan Shen](https://sxy7147.github.io/)
[Guanqi Zhan](https://www.robots.ox.ac.uk/~guanqi/)
[Hao Dong](https://zsdonghao.github.io/)

Project Page: https://lyttttt3333.github.io/broadcast.github.io/

If there is any question, please contact Yitong (liyitong_thu@gmail_com) and Ruihai (wuruihai@pku_edu_cn).

## abstract

In our daily life, cluttered scenarios are everywhere, from scattered stationery and books cluttering the table to bowls and plates filling the kitchen sink. Retrieving a target object from clutters is an essential while challenging skill for robots, for the difficulty of safely manipulating an object without disturbing others, which requires the robot to plan a manipulation sequence and first move away a few other objects supported by the target object step by step. However, due to the diversity of object configurations (e.g., categories, geometries, locations and poses) and their combinations in clutters, it is difficult for a robot to accurately infer the support relations between objects faraway with various objects in between. 

In this paper, we study retrieving objects in complicated clutters via a novel method of recursively broadcasting the accurate local dynamics to build a support relation graph of the whole scene, which largely reduces the complexity of the support relation inference and improves the accuracy. Experiments in both simulation and the real world demonstrate the efficiency and effectiveness of our method.

# Citations
    
    @inproceedings{
        li2024broadcasting,
        title={Broadcasting Support Relations Recursively from Local Dynamics for Object Retrieval in Clutters},
        author={Li, Yitong and Wu, Ruihai and Lu, Haoran and Ning, Chuanruo and Shen, Yan and Zhan, Guanqi and Dong, Hao},
        booktitle={Robotics: Science and Systems},
        year={2024}
        }

# Introduction

This folder includes the code for our paper, Broadcasting Support Relations Recursively from Local Dynamics for Object Retrieval in Clutters, especially about clutter environment building, data generation, module training and inference.

## About This Repository
    data_generation/        # contains data generation process

        scene_building/     

            gen_clutter_scene.py    # contains the code for building the clutter scenes

        data_collection/     
        
            gen_interation.py       # contains the code for collecting data for relation inference part
            gen_grasp.py            # contains the code for collecting data for manipulation part

    model_training/         # contains code and scripts for training different modules

        utils/              # utils for training modules
        pre_process_*.py     # prepare the data for corresponding part
        train_LDP.py        # train Local Dynamics Predictor
        train_RDS.py        # train Retrieval Direction Scoring Module
        train_RDP.py        # train Retrieval Direction Proposal Module
        train_GPS.py        # train Grasp Pose Scoring Module
        train_GPP.py        # train Grasp Pose Proposal Module
        train_GA.py         # train Grasp Affordance Module
        infereence_Clutter_Solver.py         # construct the Broadcast and Clutter Solver framework for support relation inference 


## Dependencies

For module training, this code can be ran on Ubuntu 18.04 with Cuda 10.1, Python 3.7, PyTorch 1.7.0. 
For data collection, we use ISAAC SIM 23.1.1 and the built-in python environment.

## Generate Training Data

Before training the network, we need to collect a large set of clutter scenes in the simulator, using the script 

    python data_generation/scene_building/gen_clutter_scene.py

Based on the scenes, we can collect interactions for training modules of relation inference. 

    python data_generation/data_collection/gen_interaction.py

Second, for grasp modules, we first document which points to try by `data_generation/data_collection/gen_pre_grasp.py`. And then we can collection the grasp ground truth parallel, which can be run by

    python data_generation/data_collection/gen_grasp.py

Generating enough offline interaction trials is necessary for a successful learning, and it may require many CPU hours for the data collection.  

## Modules Training

After the data generation, you can use `preprocess_*.py` to process the raw data into the formation of modules' inputs.

Then we can use the following scripts to train different modules. 

For example, to train `Local Dynamics Predictor`,

    python train_LDP.py

## Support Relation Inference

After that, we can construct the `Clutter Solver` based on the trained modules by, which can output the inferred support relations from a given scene. As the core part of this work, `Clutter Solver` does not need any additional training process but a list of trained modules, which can be run by

    python inference_Clutter_Solver.py




