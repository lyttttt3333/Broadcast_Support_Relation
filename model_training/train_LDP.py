from torch.utils.data import DataLoader
from module import CLS
from utils.utils_LDP import *
import random
import argparse

def iteration(path,train:bool):
    Point=PointProcess_LDR(path,point_num=point_num,device=device)
    Point.get_pair()
    if train:
        scores=cls.forward(Point.partpoint1,Point.partpoint2,Point.action,train=True)
        loss,rate=cls.get_loss(scores,Point.result)
        cls.train(loss)
        print("train:  ", rate)
    else:
        scores=cls.forward(Point.partpoint1,Point.partpoint2,Point.action,train=False)
        loss,rate=cls.get_loss(scores,Point.result)
        print("eval:  ", rate)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()  

    parser.add_argument("--input_path", type=str,default=None, help="root path for data")
    parser.add_argument("--save_path", type=str,default=None, help="root path for saving data")
    parser.add_argument("--point_num", type=int,default=256)
    parser.add_argument("--batch_num", type=int,default=32)
    parser.add_argument("--device", type=str,default="cuda:0")
    parser.add_argument("--epoch", type=int,default=120)
    parser.add_argument("--seed", type=int,default=0)

    args=parser.parse_args()

    input_path=args.input_path
    save_path=args.save_path
    point_num=args.point_num
    device=args.device
    epoch_num=args.epoch
    batch_num=args.batch_num
    seed=args.seed

    input_path = os.path.join(input_path,"data_interaction")

    cls=CLS(num=point_num,device=device)

    all_path_train_succ,all_path_train_fail=read_pair(train=True,input_path=input_path,seed=seed)
    all_path_train=all_path_train_succ+all_path_train_fail
    train_dataset=MyDataset(list(range(len(all_path_train))))
    dataloader_train=DataLoader(train_dataset,batch_size=batch_num,shuffle=True)
    all_path_test_succ,all_path_test_fail=read_pair(train=False,input_path=input_path,seed=seed)
    path_for_test=random.sample(all_path_train_succ,batch_num)+random.sample(all_path_train_fail,batch_num)

    for epoch in range(epoch_num):
        for batch_index in dataloader_train:
            batch_path=[all_path_train[item] for item in batch_index]
            iteration(batch_path,train=True)
        iteration(path_for_test,train=False)

    cls.save_model(save_path)


