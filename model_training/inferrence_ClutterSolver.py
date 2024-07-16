
import argparse
from utils.utils_CS import *
from module import CLS, DIR_Proposal, DIR_Scoring



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--pointcloud_path', type=str, default="D:\\broadcast_final\\data\\data_interaction\\test\\env_1436\\points.npy",
                        help='pointcloud_path in N*3')
    parser.add_argument('--point_mask_path', type=str, default="D:\\broadcast_final\\data\\data_interaction\\test\\env_1436\\mask_line.npy"
                        ,help='pointcloud_mask_path in N')
    parser.add_argument('--image_mask_path', type=str, default="D:\\broadcast_final\\data\\data_interaction\\test\\env_1436\\mask_piece.npy", 
                        help='pointcloud_mask_path in M*pixel_num*pixel_num, if type is 3d, default to None')
    parser.add_argument('--modules_path', type=list, 
                        help='modules save paths in order of [[LDP path], [RDS path], [RDP path]]')
    parser.add_argument('--sample_num', type=int, default=10240)
    parser.add_argument('--resample_num', type=int, default=256)
    parser.add_argument('--max_broadcast_iteration', type=int, default=6)
    parser.add_argument('--broadcast_activate_threshold', type=float, default=0.5)
    parser.add_argument('--adjacent_type', type=str, default="2d_adjacent")
    parser.add_argument('--adjacent_threshold', type=int, default=8)
    parser.add_argument('--point_sample_num', type=int, default=256)
    parser.add_argument('--action_index', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args() 

    pointcloud_path=args.pointcloud_path
    point_mask_path=args.point_mask_path
    image_mask_path=args.image_mask_path
    path_list=args.modules_path
    sample_num=args.sample_num
    resample_num=args.resample_num
    max_broadcast_iteration=args.max_broadcast_iteration
    broadcast_activate_threshold=args.broadcast_activate_threshold
    adjacent_type=args.adjacent_type
    adjacent_threshold=args.adjacent_threshold
    action_index=args.action_index
    device=args.device

    cls=CLS(num=resample_num,device=device)
    cls.load_model(root_dir="D:\\broadcast_final\\Code_BroadcastSupportRelation\\data_model_ckpt")

    dir_scoring=DIR_Scoring(device=device)
    dir_scoring.load_model(root_dir="D:\\broadcast_final\\Code_BroadcastSupportRelation\\data_model_ckpt")

    dir_proposal=DIR_Proposal(device=device)
    dir_proposal.load_model(root_dir="D:\\broadcast_final\\Code_BroadcastSupportRelation\\data_model_ckpt")
    
    pointprocess=PointProcess_eval(sample_num=sample_num,resample_num=resample_num,device=device)
    pointprocess.input_form(pointcloud_path,
                            point_mask_path,
                            image_mask_path)
    pointprocess.sample_point()
    pointprocess.mask_expand()
    pointprocess.get_part_point_batch()
    
    state=get_init_state(pointprocess.total_num,action_index,device)
    action_graph=get_action_graph(pointprocess.total_num,action_index,device)
    order=get_init_order(pointprocess.total_num,device)

    for i in range(max_broadcast_iteration):
        relation=find_relation(state,
                               pointprocess.total_num,
                               pointprocess.init_mask,adjacent_type,
                               adjacent_threshold,
                               device)
        action_graph,order=update_layer(relation=relation,
                                        state=state,
                                        action_graph=action_graph,
                                        part_point=pointprocess.part_point,
                                        model_cls=cls,
                                        model_rds=dir_scoring,
                                        model_rdp=dir_proposal,
                                        order=order,
                                        threshold=broadcast_activate_threshold,
                                        device=device)
        state=update_states(action_graph,
                            state,
                            broadcast_activate_threshold)
        converge=update_stop(state)
        if converge:
            break