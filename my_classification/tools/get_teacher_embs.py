import torch
import torch.nn as nn
import os
import logging
import numpy as np
import json
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from functools import partial

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def emb_fea_resnet34(model, dataloader, args):
    # model to evaluate mode
    model.eval()

    EMB = {}

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(), labels.cuda()

            # compute output
            teacher_out = {}
            def forward_hook(module, input, output):
                avg_out = output[0] if len(output) == 1 else output
                teacher_out['avgpool'] = torch.flatten(avg_out, 1)

            module = None
            for k, m in model.named_modules():
                if k == 'avgpool':
                    module = m
                    # print(f"now k is {k}")
                    break
            module.register_forward_hook(forward_hook)

            logits = model(images)
            # print(f"the shape of teacher output is {teacher_out['avgpool'].shape}")

            for emb, i in zip(teacher_out['avgpool'], labels):
                i = i.item()
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    for key, value in EMB.items():
        for i in range(512):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB

def emb_fea_resnet56(model, dataloader, args):
    # model to evaluate mode
    model.eval()

    EMB = {}

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(), labels.cuda()

            # compute output
            
            [f0, f1_pre, f2_pre, f3_pre, f4], x = model(images,is_feat=True)
            # print(f"the shape of f4 is {f4.shape}") 16 x 64

            for emb, i in zip(f4, labels):
                i = i.item()
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    for key, value in EMB.items():
        for i in range(64):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB

def emb_fea_wrn(model, dataloader, args):
    # model to evaluate mode
    model.eval()

    EMB = {}

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(), labels.cuda()

            [f0, f1, f2, f3, f4], out = model(images,is_feat=True)
            # print(f"the shape of f3 is {f3.shape}")
            # print(f"the shape of f4 is {f4.shape}")

            for emb, i in zip(f4, labels):
                i = i.item()
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    for key, value in EMB.items():
        for i in range(128):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB

def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)


    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)
    

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        teacher_model.cuda()
    if args.teacher_model == 'tv_resnet34':    
        emb = emb_fea_resnet34(teacher_model, train_loader, args)
        emb_json = json.dumps(emb, indent=4)
        directory = "./ckpt/cifar100_embedding_fea/"
        file_path = os.path.join(directory, "resnet34.json")
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(emb_json)
        f.close()
    elif args.teacher_model == 'cifar_wrn_40_2':
        emb = emb_fea_wrn(teacher_model, train_loader, args)
        emb_json = json.dumps(emb, indent=4)
        directory = "./ckpt/cifar100_embedding_fea/"
        file_path = os.path.join(directory, "wrn_40_2.json")
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(emb_json)
        f.close()
    elif args.teacher_model == 'cifar_resnet56':
        emb = emb_fea_resnet56(teacher_model, train_loader, args)
        emb_json = json.dumps(emb, indent=4)
        directory = "./ckpt/cifar100_embedding_fea/"
        file_path = os.path.join(directory, "resnet56.json")
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(emb_json)
        f.close()


if __name__ == '__main__':
    main()
