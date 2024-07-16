
from torch.utils.data import Dataset, DataLoader
from utils.GA_utils import *
import shutil
import os



    


if __name__ == '__main__':

    process_batch = 1
    device = "cuda:0"
    root_dir = "D:\\broadcast_final\\data"
    affordance_data_root_dir = os.path.join(root_dir, "data_affordance")
    train_paths,eval_paths =find_batch(affordance_data_root_dir)
    train_paths=train_paths

    train_root_path = os.path.join(affordance_data_root_dir,"train")
    if os.path.exists(train_root_path):
        shutil.rmtree(train_root_path)
    os.mkdir(train_root_path)

    test_root_path = os.path.join(affordance_data_root_dir,"test")
    if os.path.exists(test_root_path):
        shutil.rmtree(test_root_path)
    os.mkdir(test_root_path)

    train_num = int(len(train_paths)/process_batch)
    if (train_num%process_batch) != 0:
        train_num+=1

    for batch_idx in range(train_num):
        if batch_idx != train_num -1:
            batch_paths = train_paths[batch_idx*process_batch:(batch_idx+1)*process_batch]
        else:
            batch_paths = train_paths[batch_idx*process_batch:]

        batch_paths=data_filter(batch_paths)
        succ_list,fail_list=data_balance(batch_paths)
        succ_dict = data_convert(succ_list)
        succ_dict = load_pts(succ_dict,device)
        save_path = os.path.join(train_root_path,f"succ_{batch_idx}.h5")
        save_h5py(data=succ_dict,path=save_path)
        del succ_dict

        fail_dict = data_convert(fail_list)
        fail_dict = load_pts(fail_dict,device)
        save_path = os.path.join(train_root_path,f"fail_{batch_idx}.h5")
        save_h5py(data=fail_dict,path=save_path)
        del fail_dict

    test_num = int(len(eval_paths)/process_batch)
    if (test_num % process_batch) != 0:
        test_num+=1

    for batch_idx in range(test_num):
        if batch_idx != test_num -1:
            batch_paths = eval_paths[batch_idx*process_batch:(batch_idx+1)*process_batch]
        else:
            batch_paths = eval_paths[batch_idx*process_batch:]

        batch_paths=data_filter(batch_paths)
        succ_list,fail_list=data_balance(batch_paths)
        succ_dict = data_convert(succ_list)
        succ_dict = load_pts(succ_dict,device)
        save_path = os.path.join(test_root_path,f"succ_{batch_idx}.h5")
        save_h5py(data=succ_dict,path=save_path)
        del succ_dict

        fail_dict = data_convert(fail_list)
        fail_dict = load_pts(fail_dict,device)
        save_path = os.path.join(test_root_path,f"fail_{batch_idx}.h5")
        save_h5py(data=fail_dict,path=save_path)
        del fail_dict