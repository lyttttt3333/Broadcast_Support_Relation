from torch.utils.data import DataLoader
from utils.utils_GP import *
import argparse
from module import Pose_Scoring


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add arguments to build clutter solver')

    parser.add_argument('--data_dir', type=str, default="D:\\broadcast_final\\data\\data_affordance\\train")

    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--save_path', type=str, default="D:\\broadcast_final\\Code_BroadcastSupportRelation\\data_model_ckpt")

    parser.add_argument('--env_point_num', type=int, default=1024)
    parser.add_argument('--obj_point_num', type=int, default=256)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=1)

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

    scoring=Pose_Scoring(device=device)

    for epoch in range(epoch_num):
        for batch_index in dataloader_train:
            batch_path=[total_succ_path[item] for item in batch_index] #+ [total_fail_path[item] for item in batch_index]

            Point=PointProcess(path_list=batch_path,device=device)
            Point.load_data()
            Point.split_point_pose()

            pred_scores =scoring.forward(obj_point=Point.target_pts,
                                        env_point=Point.env_pts,
                                        grasp_pt=Point.grasp_pt,
                                        grasp_pose=Point.grasp_pose)
            loss=scoring.loss(score_hat=pred_scores, score_true=Point.grasp_result)
            scoring.train(loss)
            print(epoch,"epoch",loss.item())

    scoring.save_model(save_path)