import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Kitti, get_dataloader
from network import PointPillars
from fusion_net import FusionNet
from losses import Loss
from torch.utils.tensorboard import SummaryWriter
import statistics


def main(args):
    setup_seed()

    #detection_2d_path = config.train_config.detection_2d_path
    train_dataset = Kitti(data_root=args.data_root,
                          split='train')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)


    pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    pointpillars.load_state_dict(torch.load("/home/loahit/PointPillars-Camera-LiDAR-Fusion/pillar_logs/checkpoints/epoch_5.pth"))
    pointpillars.eval()
    #loss_func = Loss()
    fusion=FusionNet()
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.Adam(fusion.parameters(),lr = 3e-3, betas=(0.9, 0.99),weight_decay=0.01)


    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    Train_Losses=[]
    Val_Losses=[]

 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
