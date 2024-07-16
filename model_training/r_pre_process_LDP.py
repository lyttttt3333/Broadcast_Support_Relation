from utils.pre_process_LDP_utils import *
import random
import argparse
import os 

def main(cfg):
    path = os.path.join(cfg.origin_data_path, "data_interaction")
    device = cfg.device
    scene_point_num = cfg.scene_point_num
    subset_num_train = cfg.subset_num_train
    subset_num_test = cfg.subset_num_test
    batch_num_process = cfg.batch_num_process

    output_path = os.path.join(path, "pair")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for train_flag in [True,False]:

        

        if train_flag:
            full_path = os.path.join(output_path,"pair_train")
            succ_path = os.path.join(full_path,"succ")
            fail_path = os.path.join(full_path,"fail")
        else:
            full_path = os.path.join(output_path,"pair_test")
            succ_path = os.path.join(full_path,"succ")
            fail_path = os.path.join(full_path,"fail")

        if not os.path.exists(full_path):
            os.mkdir(full_path)
        if not os.path.exists(succ_path):
            os.mkdir(succ_path)
        if not os.path.exists(fail_path):
            os.mkdir(fail_path)

        all_path=find_batch_outer_env_obj(root_path=path,train=train_flag)
        all_path=data_filter(all_path,device=device)
        all_path=data_valid(all_path)
        
        total_num = subset_num_train if train_flag else subset_num_test
        total_batch = total_num // batch_num_process + 1

        total_path=random.sample(all_path, total_num) if len(all_path) > total_num else all_path

        total_idx = 0

        for i in range(total_batch):
            batch_path = total_path[i*batch_num_process: (i+1)*batch_num_process] if i != total_batch-1 else total_path[i*batch_num_process:]

            point=PreCache(batch_path,scene_point_num,device=device)
            point.preprocess()
            point.get_gt_motion()
            point.reorder()
            point.expand()

            batched_partpoint_list=get_part_point_batch(point.data,point.expand_mask_line,device)
            action_graph,action_index=get_action_info(point.action,point.relation_list,device)
            state_list=get_init_state(action_graph,action_index)
            relation_list=find_relation(state_list,point.expand_mask_piece)

            total_idx = merget_pairs(point.motion,relation_list,batched_partpoint_list,state_list,point.action,False,output_path=path,train=train_flag, idx=total_idx)

            del point

            print(f"{i}/{total_batch}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='add arguments to build clutter solver')

    parser.add_argument('--origin_data_path', type=str, 
                        default="D:\\broadcast_final\\data")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--scene_point_num', type=int, default=10240)
    parser.add_argument('--subset_num_train', type=int, default=360)
    parser.add_argument('--subset_num_test', type=int, default=64)
    parser.add_argument('--batch_num_process', type=int, default=64)

    args = parser.parse_args() 
    
    main(cfg = args)
    


    