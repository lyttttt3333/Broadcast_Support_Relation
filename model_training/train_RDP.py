import random
import argparse
from utils.utils_RD import *
from module import DIR_Proposal
from torch.utils.data import DataLoader


def iteration(path,num,train:bool):
    Point=PointProcess(path_list=path,device=device, scoring=True)
    Point.get_data(batch_num=num,
                    env_point_num=env_point_num,
                    obj_point_num=obj_point_num,
                    direction_num=direction_num)
    Point.squeeze_data()
    direction_hat, direction_true, mu, logvar=proposer.forward(obj_point=Point.data_dict["target_pts"],
                            env_point=Point.data_dict["env_pts"],
                            direction_true=Point.data_dict["best_direction"])
    loss=proposer.loss(recon_x=direction_hat, x=direction_true, mu=mu, logvar=logvar)
    if train:
        print(epoch,"epoch train",loss.item())
        proposer.train(loss)
    else:
        print(epoch,"epoch test",loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--id_length', type=int, default=4)

    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--input_path', type=str, default=None, help="root path for data")
    parser.add_argument('--save_path', type=str, default=None, help="root path for saving data")

    parser.add_argument('--env_point_num', type=int, default=1024)
    parser.add_argument('--obj_point_num', type=int, default=256)
    parser.add_argument('--direction_num', type=int, default=5)
    parser.add_argument('--batch_num', type=int, default=None)
    parser.add_argument('--test_num', type=int, default=None)
    parser.add_argument('--epoch_num', type=int, default=100)

    args = parser.parse_args() 

    input_path=args.input_path
    save_path=args.save_path
    device=args.device
    id_length=args.id_length
    batch_num=args.batch_num
    env_point_num=args.env_point_num
    direction_num=args.direction_num
    obj_point_num=args.obj_point_num
    epoch_num=args.epoch_num
    test_num=args.test_num

    input_path = os.path.join(input_path,"data_interaction","direction")
    
    train_paths, eval_paths=find_batch(root_path=input_path,train_ratio=0.7)
    index_list_train=list(range(len(train_paths)))
    train_dataset=MyDataset(index_list_train)
    dataloader_train=DataLoader(train_dataset,batch_size=batch_num,shuffle=True)
    
    proposer=DIR_Proposal(device=device)

    for epoch in range(epoch_num):

        for idx, batch_index in enumerate(dataloader_train):
            batch_path=[train_paths[item] for item in batch_index]
            iteration(batch_path, num=batch_num, train=True)

        iteration(eval_paths,num=len(eval_paths),train=False)

    proposer.save_model(save_path)
            