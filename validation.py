import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from dataset.VIGOR import VIGOR
from dataset.CVUSA import CVUSA
from dataset.CVACT import CVACT
from dataset.University1652 import University1652
from dataset.University1652701 import University1652701
from model.net import CCA
from criterion.soft_triplet import SoftTripletBiLoss
from dataset.global_sampler import DistributedMiningSampler,DistributedMiningSamplerVigor
from criterion.sam import SAM
from ptflops import get_model_complexity_info

from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--dim_out', default=1000, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--cross', action='store_true',
                    help='use cross area')

parser.add_argument('--dataset', default='university1652', type=str,
                    help='university1652, university1652701')
parser.add_argument('--loss_func', default='cosine', type=str,
                    help='consine,euclid')
parser.add_argument('--op', default='adam', type=str,
                    help='sgd, adam, adamw')

parser.add_argument('--share', action='store_true',
                    help='share fc')

parser.add_argument('--mining', action='store_true',
                    help='mining')
parser.add_argument('--asam', action='store_true',
                    help='asam')

parser.add_argument('--rho', default=0.05, type=float,
                    help='rho for sam')
parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')



best_acc1 = 0


def compute_complexity(model, args):
    if args.dataset == 'university1652':
        size_sat = [256, 256]
        size_sat_default = [256, 256]
        size_grd = [256, 256]
    elif args.dataset == 'university701':
        size_sat = [256, 256]
        size_sat_default = [256, 256]
        size_grd = [256, 256]

    if args.sat_res != 0:
        size_sat = [args.sat_res, args.sat_res]

    if args.fov != 0:
        size_grd[1] = int(args.fov /360. * size_grd[1])

    with torch.cuda.device(0):
        macs_1, params_1 = get_model_complexity_info(model.module.query_net, (3, size_grd[0], size_grd[1]), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        macs_2, params_2 = get_model_complexity_info(model.module.reference_net, (3, size_sat[0] , size_sat[1] ),
                                                     as_strings=False,
                                                     print_per_layer_stat=True, verbose=True)

        print('flops:', (macs_1+macs_2)/1e9, 'params:', (params_1+params_2)/1e6)



def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.distributed = False

    main_worker(args.gpu, args)


def main_worker(gpu,args):
    global best_acc1
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'")

    model = CCA(args=args)


    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    criterion = SoftTripletBiLoss().cuda(args.gpu)

    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.op == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.op == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.op == 'sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(parameters, base_optimizer,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False, rho=args.rho, adaptive=args.asam)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if not args.multiprocessing_distributed or args.gpu == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, 'attention'))
            os.mkdir(os.path.join(args.save_path, 'attention','train'))
            os.mkdir(os.path.join(args.save_path, 'attention','val'))

    if args.dataset == 'university1652':
        dataset = University1652
        mining_sampler = DistributedMiningSampler
    elif args.dataset == 'university1652701':
        dataset = University1652701
        mining_sampler = DistributedMiningSampler

 
    train_dataset = dataset(mode='train', print_bool=True, same_area=(not args.cross),args=args)
    val_scan_dataset = dataset(mode='scan_val', same_area=(not args.cross), args=args)
    val_query_dataset = dataset(mode='test_query', same_area=(not args.cross), args=args)
    val_reference_dataset = dataset(mode='test_reference', same_area=(not args.cross), args=args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_scan_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        drop_last=False)

    val_scan_loader = torch.utils.data.DataLoader(
        val_scan_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        drop_last=False)

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True) # 512, 64
    val_reference_loader = torch.utils.data.DataLoader(
        val_reference_dataset, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True) # 80, 128


    if args.evaluate:
        if not args.multiprocessing_distributed or args.gpu == 0:
            validate(val_query_loader, val_reference_loader, model, args)
        return

 

def scan(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_q, images_k, _, indexes, delta, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output, sav satellite only
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


def validate(val_query_loader, val_reference_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress_q = ProgressMeter(
        len(val_query_loader),
        [batch_time],
        prefix='Test_query: ')
    progress_k = ProgressMeter(
        len(val_reference_loader),
        [batch_time],
        prefix='Test_reference: ')

    
    model_query = model.query_net
    model_reference = model.reference_net
    model_head = model.mlphead


    model_query.eval()
    model_reference.eval()
    model_head.eval()
    print('model validate on cuda', args.gpu)

    query_features = np.zeros([len(val_query_loader.dataset), args.dim_out])
    query_labels = np.zeros([len(val_query_loader.dataset)])
    reference_features = np.zeros([len(val_reference_loader.dataset), args.dim_out])
    reference_labels = np.zeros([len(val_reference_loader.dataset)])

    with torch.no_grad():
        end = time.time()
        for i, (images, indexes,sat_name,uav_ids) in enumerate(val_reference_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            indexes = indexes.cuda(args.gpu, non_blocking=True)

            reference_embed = model_head(model_reference(x=images, indexes=indexes))  # delta

            reference_features[indexes.cpu().numpy().astype(int), :] = reference_embed.detach().cpu().numpy()
            

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_k.display(i)
        end = time.time()
        for i, (images, indexes, labels,cls_name_lst) in enumerate(val_query_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            indexes = indexes.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

            query_embed = model_head(model_query(images))

            query_features[indexes.cpu().numpy(), :] = query_embed.cpu().numpy()
            query_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_q.display(i)
        [top1, top5] = accuracy(args,query_features, reference_features, query_labels.astype(int))
        [top1_inverse, top5_inverse] = accuracy_inverse(args,query_features, reference_features, reference_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'uav_global_descriptor.npy'), query_features)
        np.save('sat_global_descriptor.npy', reference_features)

    return top1_inverse


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    torch.save(state, os.path.join(args.save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(args.save_path,filename), os.path.join(args.save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '//' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# d->s
def accuracy(args, query_features, reference_features, query_labels, topk=[1,5,10]): 
# def accuracy(args, reference_features, query_features, query_labels, topk=[1,5,10]): # s
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    sum_max_prec = 0
    
    if args.loss_func == 'cosine':
        # cosine similarity
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())
        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
            if ranking == 0:
                max_prec = 1
            else:
                max_prec = (0+1/(ranking+1))/2
            sum_max_prec += max_prec
            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
                    # results_ap[j] += 1/
        
    elif args.loss_func == 'euclid':
        # euclid similarity
        dist_mat = np.zeros([N, M])
        query_features = query_features/np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features = reference_features/np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        for i in range(N):
            for j in range(M):
                dist_mat[i,j] = np.linalg.norm(query_features[i,:] - reference_features[j,:])
        for i in range(N):
            ranking = np.sum((dist_mat[i,:]<dist_mat[i,query_labels[i]])*1.)
            if ranking == 0:
                max_prec = 1
            else:
                max_prec = (0+1/(ranking+1))/2
            sum_max_prec += max_prec
            for j, k in enumerate(topk): 
                if ranking < k:
                    results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    ap=sum_max_prec/N*100
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, AP:{}, time:{}'.format(results[0], results[1], results[2], results[-1], ap, time.time() - ts))
    # print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, AP:{}, time:{}'.format(results[0], results[1], results[2], results[-1], 0, time.time() - ts))
    return results[:2]

# s->d
def accuracy_inverse(args, query_features, reference_features, ref_labels, topk=[1,5,10]): 
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(N//100)
    results = np.zeros(len(topk))
    sum_max_prec = 0
    
    if args.loss_func == 'cosine':
        # cosine similarity
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())
        for class_id in range(M):
            # for recall
            temp = np.max(similarity[class_id*54:(class_id+1)*54, class_id])
            ranking = np.sum((similarity[:class_id*54,class_id]>temp)*1.)+np.sum((similarity[(class_id+1)*54:,class_id]>temp)*1.)
            for j, k in enumerate(topk): #[1,5,10,1%]
                if ranking < k:
                    results[j] += 1.
            # for ap
            temp_list = np.sort(similarity[class_id*54:(class_id+1)*54, class_id])
            for i,elem in enumerate(temp_list):
                rank_elem = np.sum((similarity[:class_id*54,class_id]>elem)*1.)+np.sum((similarity[(class_id+1)*54:,class_id]>elem)*1.)
                sum_max_prec+=((i+1)/(rank_elem+1))/54


    elif args.loss_func == 'euclid':
        # euclid similarity
        dist_mat = np.zeros([N, M])
        query_features = query_features/np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features = reference_features/np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        for i in range(N):
            for j in range(M):
                dist_mat[i,j] = np.linalg.norm(query_features[i,:] - reference_features[j,:])
        np.save('dist.npy', dist_mat)
        for class_id in range(M):
            temp = np.min(dist_mat[class_id*54:(class_id+1)*54, class_id])
            ranking = np.sum((dist_mat[:class_id*54,class_id]<temp)*1.)+np.sum((dist_mat[(class_id+1)*54:,class_id]<temp)*1.)
            for j, k in enumerate(topk): #[1,5,10,1%]
                if ranking < k:
                    results[j] += 1.
            # for ap
            temp_list = np.sort(dist_mat[class_id*54:(class_id+1)*54, class_id])
            for i,elem in enumerate(temp_list):
                rank_elem = np.sum((dist_mat[:,class_id]<elem)*1.)
                sum_max_prec+=((i+1)/(rank_elem+1))/54

    results = results/ M * 100.
    ap = sum_max_prec/ M * 100
    print('inverse%-top1:{}, top5:{}, top10:{}, top1%:{}, AP:{}, time:{}'.format(results[0], results[1], results[2], results[-1], ap, time.time() - ts))
    # print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, AP:{}, time:{}'.format(results[0], results[1], results[2], results[-1], 0, time.time() - ts))
    return results[:2]

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    main()
