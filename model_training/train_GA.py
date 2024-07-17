from module import Affordance
from utils.utils_GP import *
import random
from torch.utils.data import  DataLoader
import argparse


def iteration(path_list,train:bool):
    Point=PointProcess(path_list=path_list,device=device)
    Point.load_data()
    Point.split_point_pose()

    scores=afford.forward(obj_point=Point.target_pts,
                          env_point=Point.env_pts,
                          grasp_point=Point.grasp_pt)
    loss,rate=afford.train(scores,Point.result,train=train)
    if train:
        print(f"{epoch} train :",loss,rate )
    else:
        print(f"{epoch} test :",loss,rate )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=None, help="training data path")
    parser.add_argument('--save_path', type=str, default=None, help="the root path to store ckpts of all models")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--env_point_num', type=int, default=1024)
    parser.add_argument('--obj_point_num', type=int, default=256)
    parser.add_argument('--batch_num', type=int, default=4)
    parser.add_argument('--epoch_num', type=int, default=180)

    args = parser.parse_args() 

    data_dir=args.data_dir
    save_path=args.save_path
    device=args.device
    batch_num=args.batch_num
    env_point_num=args.env_point_num
    obj_point_num=args.obj_point_num
    epoch_num=args.epoch_num

    total_succ_path, total_fail_path=find_batch(root_path=data_dir)
    num_train=len(total_succ_path)
    index_list_train=list(range(num_train))

    train_dataset=MyDataset(index_list_train)
    dataloader_train=DataLoader(train_dataset,batch_size=batch_num,shuffle=True)

    afford=Affordance(device)

    for epoch in range(epoch_num):

        random.shuffle(total_succ_path)
        random.shuffle(total_fail_path)

        for batch_index in dataloader_train:
            batch_path_succ=[total_fail_path[item] for item in batch_index]
            batch_path_fail=[total_fail_path[item] for item in batch_index]
            batch_path=batch_path_succ+batch_path_fail
            
            iteration(batch_path,train=True)
    afford.save_model(save_path)
        