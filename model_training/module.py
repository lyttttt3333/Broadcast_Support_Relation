import torch 
import torch.nn as nn
from torch.nn import functional as F
from utils.pointnet2_sem_seg import sem_model,sem_model_full
from utils.pointnet2_cls_ssg import cls_model


class ScoringDecoder(nn.Module):
    def __init__(self, structure1):
        super().__init__()
        self.list1=[]
        last=structure1.pop()
        for i in range(len(structure1)-1):
            self.list1.append(torch.nn.Linear(structure1[i],structure1[i+1]))
            self.list1.append(torch.nn.ELU())
        self.list1.append(torch.nn.Linear(structure1[i+1],last))
        self.list1.append(torch.nn.Sigmoid())
        self.encoder=nn.Sequential(*(self.list1))

    def forward(self, x):
        return self.encoder(x)

class MLP(nn.Module):
    def __init__(self, structure,append_layer=None):
        super().__init__()
        self.list=[]
        last=structure.pop()
        for i in range(len(structure)-1):
            self.list.append(torch.nn.Linear(structure[i],structure[i+1]))
            self.list.append(torch.nn.ELU())
        self.list.append(torch.nn.Linear(structure[i+1],last))
        
        if append_layer is None:
            self.encoder=nn.Sequential(*(self.list))
        else:
            self.list.append(append_layer)
            self.encoder=nn.Sequential(*(self.list))

    def forward(self, x):
        return self.encoder(x)
    
class ClsEncoder(nn.Module):
    def __init__(self, structure1):
        super().__init__()
        self.list1=[]
        last=structure1.pop()
        for i in range(len(structure1)-1):
            self.list1.append(torch.nn.Linear(structure1[i],structure1[i+1]))
            self.list1.append(torch.nn.ELU())
        self.list1.append(torch.nn.Linear(structure1[i+1],last))
        self.encoder=nn.Sequential(*(self.list1))

    def forward(self, x):
        return self.encoder(x)

class ClsDecoder(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.list=[]
        last=structure.pop()
        for i in range(len(structure)-1):
            self.list.append(torch.nn.Linear(structure[i],structure[i+1]))
            self.list.append(torch.nn.ELU())
        self.list.append(torch.nn.Linear(structure[i+1],last))
        self.list.append(torch.nn.Sigmoid())
        self.encoder=nn.Sequential(*(self.list))

    def forward(self, x):
        y=self.encoder(x)
        return y

class CLS():
    def __init__(self,num, device):
        super().__init__()
        self.device =device
        self.pts_num = num
        self.source=sem_model(64).to(device)
        self.encoder=ClsEncoder([131,188,256]).to(device)
        self.decoder=ClsDecoder([64,64,1]).to(device)
        self.init_opt()
        self.init_sch()

    def init_opt(self):
        self.source_opt = torch.optim.Adam(self.source.parameters(),lr=0.001)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),lr=0.001)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=0.001)

    def init_sch(self):
        self.source_sch=torch.optim.lr_scheduler.StepLR(self.source_opt,step_size=36,gamma=0.1)
        self.encoder_sch=torch.optim.lr_scheduler.StepLR(self.encoder_opt,step_size=36,gamma=0.1)
        self.decoder_sch=torch.optim.lr_scheduler.StepLR(self.decoder_opt,step_size=36,gamma=0.1)

    def zero_grad(self):
        self.source_opt.zero_grad()
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

    def step_grad(self):
        self.source_opt.step()
        self.encoder_opt.step()
        self.decoder_opt.step()

    def step_sch(self):
        self.source_sch.step()
        self.encoder_sch.step()
        self.decoder_sch.step()

    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"LDP")
        path = os.path.join(model_dir,"source.ckpt")
        self.source.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"encoder.ckpt")
        self.encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"decoder.ckpt")
        self.decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"LDP")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"source.ckpt")
        torch.save(self.source.state_dict(),path)
        path = os.path.join(model_dir,"encoder.ckpt")
        torch.save(self.encoder.state_dict(),path)
        path = os.path.join(model_dir,"decoder.ckpt")
        torch.save(self.decoder.state_dict(),path)

    def forward(self, source_mat,receive_mat,act,train=True,sample_num=256):
        if train:
            self.source.train()
        else:
            self.source.eval()
        #source_mat=self.farthest_point_sample(partpoint1,num)
        #receive_mat=self.farthest_point_sample(partpoint2,num)
        act=act.unsqueeze(1)
        act=act.repeat(1,sample_num,1)
        signal_source=act
        signal_receive=torch.zeros_like(act)
        source_mat=torch.cat([source_mat,signal_source],dim=-1)
        receive_mat=torch.cat([receive_mat,signal_receive],dim=-1)
        mat=torch.cat([source_mat,receive_mat],dim=1)
        source_ft=self.source(mat)
        source_ft=torch.mean(source_ft,dim=1)
        #receive_ft=self.receive(receive_mat)
        #receive_ft=torch.mean(receive_ft,dim=1)
        output=self.decoder(source_ft)
        return output.squeeze(-1)
    
    def farthest_point_sample(self,xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        centroids=centroids
        new_xyz=self.index_points(xyz,centroids)
        return new_xyz
    
    def index_points(self,points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
    
    def get_loss(self,pred,gt):
        fn=nn.BCELoss(reduction="mean")
        loss=fn(pred,gt)
        judge=torch.zeros_like(pred)
        judge[pred>=0.5]=1
        judge[pred<0.5]=0
        error=judge-gt
        index=torch.where(error!=0)[0].tolist()
        mismatch=torch.norm(judge-gt,p=1,keepdim=False).item()
        return loss,1-mismatch/gt.shape[0]
    
    def train(self,loss):
        self.zero_grad()
        loss.backward()
        self.step_grad()
        self.step_sch()

    
    def _get_motion(self,p,action):
        action=action.unsqueeze(-1).repeat(1,1,p.shape[-1])
        weight=torch.rand((p.shape[-1],)).reshape(1,1,-1).repeat(1,3,1).to(self.device)
        p=p+6*action*weight
        return p

    
    def _get_merged_point(self,p1,p2):
        cent=torch.mean(p1,dim=-1,keepdim=True)
        p1=p1-cent
        p2=p2-cent
        flag_1=torch.mean(torch.zeros_like(p1),dim=1,keepdim=True)
        flag_2=torch.mean(torch.ones_like(p2),dim=1,keepdim=True)
        p1=torch.cat([p1,flag_1],dim=1)
        p2=torch.cat([p2,flag_2],dim=1)
        p=torch.cat([p1,p2],dim=-1)
        return p
    
    def _get_merged_point(self,p1,p2):
        cent=torch.mean(p1,dim=-1,keepdim=True)
        p1=p1-cent
        p2=p2-cent
        flag_1=torch.mean(torch.zeros_like(p1),dim=1,keepdim=True)
        flag_2=torch.mean(torch.ones_like(p2),dim=1,keepdim=True)
        p1=torch.cat([p1,flag_1],dim=1)
        p2=torch.cat([p2,flag_2],dim=1)
        p=torch.cat([p1,p2],dim=-1)
        return p
    

class DIR_Scoring():
    def __init__(self,device):
        super().__init__()
        self.device=device
        self.obj_encoder=cls_model(64,pts_list=[180,128,64]).to(self.device)
        self.env_encoder=cls_model(64,pts_list=[512,256,64]).to(self.device)
        self.decoder=MLP([64 + 64 + 3, 128, 64, 1]).to(self.device)
        self.init_opt()
        self.init_sch()

    def init_opt(self):
        self.obj_encoder_opt = torch.optim.Adam(self.obj_encoder.parameters(),lr=0.001)
        self.env_encoder_opt = torch.optim.Adam(self.env_encoder.parameters(), lr=0.001)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),lr=0.001)

    def init_sch(self):
        self.obj_encoder_sch=torch.optim.lr_scheduler.StepLR(self.obj_encoder_opt,step_size=60,gamma=0.1)
        self.env_encoder_sch=torch.optim.lr_scheduler.StepLR(self.env_encoder_opt,step_size=60,gamma=0.1)
        self.decoder_sch=torch.optim.lr_scheduler.StepLR(self.decoder_opt,step_size=60,gamma=0.1)

    def zero_grad(self):
        self.obj_encoder_opt.zero_grad()
        self.env_encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

    def step_grad(self):
        self.obj_encoder_opt.step()
        self.env_encoder_opt.step()
        self.decoder_opt.step()

    def step_sch(self):
        self.obj_encoder_sch.step()
        self.env_encoder_sch.step()
        self.decoder_sch.step()

    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"RDS")
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        self.obj_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"env_encoder.ckpt")
        self.env_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"decoder.ckpt")
        self.decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"RDS")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        torch.save(self.obj_encoder.state_dict(),path)
        path = os.path.join(model_dir,"env_encoder.ckpt")
        torch.save(self.env_encoder.state_dict(),path)
        path = os.path.join(model_dir,"decoder.ckpt")
        torch.save(self.decoder.state_dict(),path)

    def forward(self, obj_point,env_point,direction):
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)
        input_ft=torch.cat([obj_ft,env_ft,direction],dim=-1)
        scores=self.decoder(input_ft)
        return scores
    
    def predict(self, obj_point,env_point,direction):
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)
        obj_ft=obj_ft.repeat(direction.shape[0],1)
        env_ft=env_ft.repeat(direction.shape[0],1)
        input_ft=torch.cat([obj_ft,env_ft,direction],dim=-1)
        scores=self.decoder(input_ft)
        return scores
    
    def loss(self,score_hat, score_true):
        fn=nn.L1Loss()
        loss=fn(score_hat,score_true)
        return loss
    
    def train(self,loss):
        self.zero_grad()
        loss.backward()
        self.step_grad()
        self.step_sch()
    
    
class DIR_Proposal():
    def __init__(self,device):
        super().__init__()
        self.obj_encoder=cls_model(64).to(device)
        self.env_encoder=cls_model(64).to(device)
        self.encoder_mu=MLP([128 + 3, 64, 32, 16]).to(device)
        self.encoder_var=MLP([128 + 3, 64, 32, 16]).to(device)
        self.decoder=MLP([16 + 128, 128, 64, 3]).to(device)
        self.device=device
        self.init_opt()
        self.init_sch()

    def init_opt(self):
        self.obj_encoder_opt = torch.optim.Adam(self.obj_encoder.parameters(),lr=0.01)
        self.env_encoder_opt = torch.optim.Adam(self.env_encoder.parameters(), lr=0.01)
        self.encoder_mu_opt = torch.optim.Adam(self.encoder_mu.parameters(),lr=0.01)
        self.encoder_var_opt = torch.optim.Adam(self.encoder_var.parameters(),lr=0.01)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),lr=0.001)

    def init_sch(self):
        self.obj_encoder_sch=torch.optim.lr_scheduler.StepLR(self.obj_encoder_opt,step_size=6,gamma=0.1)
        self.env_encoder_sch=torch.optim.lr_scheduler.StepLR(self.env_encoder_opt,step_size=6,gamma=0.1)
        self.encoder_mu_sch=torch.optim.lr_scheduler.StepLR(self.encoder_mu_opt,step_size=6,gamma=0.1)
        self.encoder_var_sch=torch.optim.lr_scheduler.StepLR(self.encoder_var_opt,step_size=6,gamma=0.1)
        self.decoder_sch=torch.optim.lr_scheduler.StepLR(self.decoder_opt,step_size=6,gamma=0.1)

    def zero_grad(self):
        self.obj_encoder_opt.zero_grad()
        self.env_encoder_opt.zero_grad()
        self.encoder_var_opt.zero_grad()
        self.encoder_mu_opt.zero_grad()
        self.decoder_opt.zero_grad()

    def step_grad(self):
        self.obj_encoder_opt.step()
        self.env_encoder_opt.step()
        self.encoder_var_opt.step()
        self.encoder_mu_opt.step()
        self.decoder_opt.step()

    def step_sch(self):
        self.obj_encoder_sch.step()
        self.env_encoder_sch.step()
        self.encoder_mu_sch.step()
        self.encoder_var_sch.step()
        self.decoder_sch.step()

    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"RDP")
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        self.obj_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"env_encoder.ckpt")
        self.env_encoder.load_state_dict(torch.load(path))

        path = os.path.join(model_dir,"encoder_mu.ckpt")
        self.encoder_mu.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"encoder_var.ckpt")
        self.encoder_var.load_state_dict(torch.load(path))

        path = os.path.join(model_dir,"decoder.ckpt")
        self.decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"RDP")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        torch.save(self.obj_encoder.state_dict(),path)
        path = os.path.join(model_dir,"env_encoder.ckpt")
        torch.save(self.env_encoder.state_dict(),path)

        path = os.path.join(model_dir,"encoder_mu.ckpt")
        torch.save(self.encoder_mu.state_dict(),path)
        path = os.path.join(model_dir,"encoder_var.ckpt")
        torch.save(self.encoder_var.state_dict(),path)

        path = os.path.join(model_dir,"decoder.ckpt")
        torch.save(self.decoder.state_dict(),path)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, obj_point, env_point, direction_true):
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)
        condition_ft=torch.cat([obj_ft,env_ft],dim=-1)
        input_ft=torch.cat([condition_ft,direction_true],dim=-1)
        mu=self.encoder_mu(input_ft)
        logvar=self.encoder_var(input_ft)
        latent = self.sample(mu,logvar)
        condition_latent=torch.cat([condition_ft,latent],dim=-1)
        direction_hat=F.normalize(self.decoder(condition_latent),p=2,dim=-1)
        return direction_hat, direction_true, mu, logvar
    
    def loss(self,recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
        return recon_loss + KL_loss
    
    def train(self,loss):
        self.zero_grad()
        loss.backward()
        self.step_grad()
        self.step_sch()

    def propose(self,num,obj_point,env_point):
        input_latent=torch.randn((num,16)).to(self.device)
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)
        condition_ft=torch.cat([obj_ft,env_ft],dim=-1)
        condition_ft=condition_ft.repeat(num,1)
        condition_latent=torch.cat([condition_ft,input_latent],dim=-1)
        return self.decoder(condition_latent)

class Affordance():
    def __init__(self,device):
        super().__init__()
        self.encoder=sem_model(128, init_pts=257).to(device)
        self.obj_decoder=ScoringDecoder([128,64,1]).to(device)
        self.env_decoder=ScoringDecoder([128 + 3 + 1, 64, 1]).to(device)
        self._init_opt()
        self._init_sch()

    def _init_opt(self):
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),lr=0.002)
        self.obj_decoder_opt = torch.optim.Adam(self.obj_decoder.parameters(), lr=0.002)
        self.env_decoder_opt = torch.optim.Adam(self.env_decoder.parameters(), lr=0.002)
    
    def _init_sch(self):
        self.encoder_sch=torch.optim.lr_scheduler.StepLR(self.encoder_opt,step_size=6,gamma=0.66)
        self.obj_decoder_sch=torch.optim.lr_scheduler.StepLR(self.obj_decoder_opt,step_size=6,gamma=0.66)
        self.env_decoder_sch=torch.optim.lr_scheduler.StepLR(self.env_decoder_opt,step_size=6,gamma=0.66)

    def _zero_grad(self):
        self.encoder_opt.zero_grad()
        self.obj_decoder_opt.zero_grad()
        self.env_decoder_opt.zero_grad()

    def _step_grad(self):
        self.encoder_opt.step()
        self.obj_decoder_opt.step()
        self.env_decoder_opt.step()
    
    def _step_sch(self):
        self.encoder_sch.step()
        self.obj_decoder_sch.step()
        self.env_decoder_sch.step()
    
    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"GA")
        path = os.path.join(model_dir,"encoder.ckpt")
        self.encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"env_decoder.ckpt")
        self.env_decoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"obj_decoder.ckpt")
        self.obj_decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"GA")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"encoder.ckpt")
        torch.save(self.encoder.state_dict(),path)
        path = os.path.join(model_dir,"env_decoder.ckpt")
        torch.save(self.env_decoder.state_dict(),path)
        path = os.path.join(model_dir,"obj_decoder.ckpt")
        torch.save(self.obj_decoder.state_dict(),path)
        

    def forward(self,obj_point,env_point,grasp_point):
        relu = nn.ReLU()
        geom_ft = self.geom_ft(obj_pts=obj_point,grasp_pts=grasp_point)
        geom_score = self.geom_score(geom_ft=geom_ft)
        env_score = self.env_score(geom_ft=geom_ft,env_pts=env_point)
        final_score = relu(env_score)
        return final_score
        
    def geom_ft(self,obj_pts, grasp_pts):
        grasp_pts = grasp_pts[:,None,:]
        obj_pts = torch.cat([grasp_pts,obj_pts],dim=1)
        obj_pts_ft = self.encoder(obj_pts)[:,-1,:]
        return obj_pts_ft
    
    def env_score(self,geom_ft, env_pts):
        relu=nn.ReLU()
        env_point_num=env_pts.shape[1]
        geom_ft = geom_ft[:,None,:].repeat(1,env_point_num,1)
        cat_ft = torch.cat([geom_ft, env_pts],dim=-1)
        env_score = self.env_decoder(cat_ft)
        env_score = env_score * (1 - env_pts[:,:,-1][:,:,None])
        #env_score = relu(env_score) - 0.5
        env_score = torch.max(env_score,dim=1)[0]
        return env_score
        

    
    def geom_score(self,geom_ft):
        return self.obj_decoder(geom_ft)
        
    
    def train(self,pred,gt,train:bool):
        if train:
            fn=torch.nn.BCELoss(reduction="mean")
            loss=fn(pred,gt)
            self._zero_grad()
            loss.backward()
            self._step_grad()
            self._step_sch()
        error=self._eval(pred.detach(),gt.detach())
        return loss.detach().item(),1-error/pred.shape[0]
    
    def _eval(self,pred,gt):
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
        error=torch.abs(pred-gt)
        error_num=torch.sum(error).item()
        return error_num
    

class Pose_Scoring():

    def __init__(self,device):
        super().__init__()
        self.device=device
        self.obj_encoder=cls_model(64,pts_list=[180,128,64]).to(self.device)
        self.env_encoder=cls_model(64,pts_list=[512,256,64],normal_channel=True).to(self.device)
        self.decoder=MLP([64 + 64 + 3 + 4, 128, 64, 1]).to(self.device)
        self.init_opt()
        self.init_sch()

    def init_opt(self):
        self.obj_encoder_opt = torch.optim.Adam(self.obj_encoder.parameters(),lr=0.001)
        self.env_encoder_opt = torch.optim.Adam(self.env_encoder.parameters(), lr=0.001)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),lr=0.001)

    def init_sch(self):
        self.obj_encoder_sch=torch.optim.lr_scheduler.StepLR(self.obj_encoder_opt,step_size=60,gamma=0.1)
        self.env_encoder_sch=torch.optim.lr_scheduler.StepLR(self.env_encoder_opt,step_size=60,gamma=0.1)
        self.decoder_sch=torch.optim.lr_scheduler.StepLR(self.decoder_opt,step_size=60,gamma=0.1)

    def zero_grad(self):
        self.obj_encoder_opt.zero_grad()
        self.env_encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

    def step_grad(self):
        self.obj_encoder_opt.step()
        self.env_encoder_opt.step()
        self.decoder_opt.step()

    def step_sch(self):
        self.obj_encoder_sch.step()
        self.env_encoder_sch.step()
        self.decoder_sch.step()

    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"GPS")
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        self.obj_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"env_encoder.ckpt")
        self.env_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"decoder.ckpt")
        self.decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"GPS")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        torch.save(self.obj_encoder.state_dict(),path)
        path = os.path.join(model_dir,"env_encoder.ckpt")
        torch.save(self.env_encoder.state_dict(),path)
        path = os.path.join(model_dir,"decoder.ckpt")
        torch.save(self.decoder.state_dict(),path)

    def forward(self, obj_point,env_point,grasp_pt,grasp_pose):
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)        
        obj_ft = obj_ft[:,None,:]
        obj_ft = obj_ft.repeat(1,grasp_pose.shape[1],1)
        env_ft = env_ft[:,None,:]
        env_ft = env_ft.repeat(1,grasp_pose.shape[1],1)
        input_ft=torch.cat([obj_ft,env_ft,grasp_pt,grasp_pose],dim=-1)
        scores=self.decoder(input_ft)
        return scores
    
    def loss(self,score_hat, score_true):
        fn=nn.BCELoss(reduction="mean")
        loss=fn(score_hat,score_true)
        return loss
    
    def train(self,loss):
        self.zero_grad()
        loss.backward()
        self.step_grad()
        self.step_sch()

class Pose_Proposal():
    def __init__(self,device):
        super().__init__()
        self.obj_encoder=cls_model(64,pts_list=[128,64,32]).to(device)
        self.env_encoder=cls_model(64,normal_channel=True,pts_list=[128,64,32]).to(device)
        self.encoder_mu=MLP([128 + 7, 64, 32, 16]).to(device)
        self.encoder_var=MLP([128 + 7, 64, 32, 16]).to(device)
        self.decoder=MLP([16 + + 3 + 128, 128, 64, 4]).to(device)
        self.device=device
        self.init_opt()
        self.init_sch()

    def init_opt(self):
        self.obj_encoder_opt = torch.optim.Adam(self.obj_encoder.parameters(),lr=0.01)
        self.env_encoder_opt = torch.optim.Adam(self.env_encoder.parameters(), lr=0.01)
        self.encoder_mu_opt = torch.optim.Adam(self.encoder_mu.parameters(),lr=0.01)
        self.encoder_var_opt = torch.optim.Adam(self.encoder_var.parameters(),lr=0.01)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),lr=0.001)

    def init_sch(self):
        self.obj_encoder_sch=torch.optim.lr_scheduler.StepLR(self.obj_encoder_opt,step_size=6,gamma=0.1)
        self.env_encoder_sch=torch.optim.lr_scheduler.StepLR(self.env_encoder_opt,step_size=6,gamma=0.1)
        self.encoder_mu_sch=torch.optim.lr_scheduler.StepLR(self.encoder_mu_opt,step_size=6,gamma=0.1)
        self.encoder_var_sch=torch.optim.lr_scheduler.StepLR(self.encoder_var_opt,step_size=6,gamma=0.1)
        self.decoder_sch=torch.optim.lr_scheduler.StepLR(self.decoder_opt,step_size=6,gamma=0.1)

    def zero_grad(self):
        self.obj_encoder_opt.zero_grad()
        self.env_encoder_opt.zero_grad()
        self.encoder_var_opt.zero_grad()
        self.encoder_mu_opt.zero_grad()
        self.decoder_opt.zero_grad()

    def step_grad(self):
        self.obj_encoder_opt.step()
        self.env_encoder_opt.step()
        self.encoder_var_opt.step()
        self.encoder_mu_opt.step()
        self.decoder_opt.step()

    def step_sch(self):
        self.obj_encoder_sch.step()
        self.env_encoder_sch.step()
        self.encoder_mu_sch.step()
        self.encoder_var_sch.step()
        self.decoder_sch.step()

    def load_model(self,root_dir):
        import os
        model_dir=os.path.join(root_dir,"GPP")
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        self.obj_encoder.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"env_encoder.ckpt")
        self.env_encoder.load_state_dict(torch.load(path))

        path = os.path.join(model_dir,"encoder_mu.ckpt")
        self.encoder_mu.load_state_dict(torch.load(path))
        path = os.path.join(model_dir,"encoder_var.ckpt")
        self.encoder_var.load_state_dict(torch.load(path))

        path = os.path.join(model_dir,"decoder.ckpt")
        self.decoder.load_state_dict(torch.load(path))

    def save_model(self,root_dir):
        import os
        import shutil
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        model_dir=os.path.join(root_dir,"GPP")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        path = os.path.join(model_dir,"obj_encoder.ckpt")
        torch.save(self.obj_encoder.state_dict(),path)
        path = os.path.join(model_dir,"env_encoder.ckpt")
        torch.save(self.env_encoder.state_dict(),path)

        path = os.path.join(model_dir,"encoder_mu.ckpt")
        torch.save(self.encoder_mu.state_dict(),path)
        path = os.path.join(model_dir,"encoder_var.ckpt")
        torch.save(self.encoder_var.state_dict(),path)

        path = os.path.join(model_dir,"decoder.ckpt")
        torch.save(self.decoder.state_dict(),path)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, obj_point, env_point, grasp_pt, grasp_pose_true):
        obj_ft=self.obj_encoder(obj_point)
        env_ft=self.env_encoder(env_point)
        condition_ft=torch.cat([obj_ft,env_ft],dim=-1)
        condition_ft = condition_ft[:,None,:]
        condition_ft = condition_ft.repeat(1,grasp_pose_true.shape[1],1)
        input_ft=torch.cat([condition_ft, grasp_pt, grasp_pose_true],dim=-1)
        mu=self.encoder_mu(input_ft)
        logvar=self.encoder_var(input_ft)
        latent = self.sample(mu,logvar)
        condition_latent=torch.cat([condition_ft, grasp_pt, latent],dim=-1)
        grasp_pose_hat=self.decoder(condition_latent)
        return grasp_pose_hat, grasp_pose_true, mu, logvar
    
    def loss(self,recon_x, x, mu, logvar,result):
        recon_x = recon_x * result
        x = x *result
        mu = mu * result
        logvar = logvar * result + (1 - result)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  
        return recon_loss + KL_loss
    
    def train(self,loss):
        self.zero_grad()
        loss.backward()
        self.step_grad()
        self.step_sch()

    def propose(self,num,obj_point,env_point):
        input_latent=torch.randn((num,16))
        obj_ft=self.obj_encoder(obj_point.unsqueeze(0))
        env_ft=self.env_encoder(env_point.unsqueeze(0))
        condition_ft=torch.cat([obj_ft,env_ft],dim=-1)
        condition_ft=condition_ft.repeat(num,1)
        condition_latent=torch.cat([condition_ft,input_latent])
        return self.decoder(condition_latent)
