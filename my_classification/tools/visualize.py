import torch
import torch.nn.functional as F
import numpy as np
import random
from lib.utils.args import parse_args
from datetime import datetime
from lib.utils.dist_utils import init_dist, init_logger
from lib.dataset.builder import build_dataloader
from lib.models.builder import build_model
import matplotlib.pyplot as plt
from lib.utils.misc import CheckpointManager
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

class_colors_rgb = [
    [230, 25, 75],   # Red
    [60, 180, 75],   # Green
    [255, 225, 25],  # Yellow
    [0, 130, 200],   # Blue
    [245, 130, 48],  # Orange
    [145, 30, 180],  # Purple
    [70, 240, 240],  # Cyan
    [240, 50, 230],  # Magenta
    [210, 245, 60],  # Lime
    [250, 190, 212],  # Pink
    [0, 128, 128],   # Teal
    [220, 190, 255],  # Lavender
    [170, 110, 40],  # Brown
    [255, 250, 200],  # Beige
    [128, 0, 0],     # Maroon
    [170, 255, 195],  # Mint
    [128, 128, 0],   # Olive
    [255, 215, 180],  # Coral
    [0, 0, 128],     # Navy
    [128, 128, 128]  # Grey
]

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

denormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

def get_attn_map(feature):
    resized = F.interpolate(feature, size=(32, 32), mode='bilinear', align_corners=False)
    avg = resized.mean(dim=1).view(resized.size(0), -1)
    attn = (avg / 0.5).softmax(dim=-1)
    return attn[0].view(32, 32)


def tsne_visualization(model, device, test_loader):
    test_loader.shuffle = True
    model.eval()
    model = model.to(device)
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature, _ = model(data, is_feat=True)
            feature = feature[-1]
            features.append(feature.cpu().detach().numpy())
            labels.append(target.cpu().detach().numpy())
            break
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.eye(100)[labels]
    tsne = TSNE(n_components=2, random_state=0)
    features = tsne.fit_transform(features)
    plt.figure()
    for idx, i in enumerate([79, 99,  1, 34, 69, 14, 98, 17, 36, 73, 72, 19, 71,  5, 59, 67,  7,
                             26, 50, 93]):
        class_indices = np.where(labels[:, i] == 1)[0]
        plt.scatter(features[class_indices, 0], features[class_indices, 1], c=[
                    np.array(class_colors_rgb[idx])/255], s=20)
    plt.title('t-SNE visualization')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig('tsne.png')

def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}/reproduce'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)

    student_model = build_model(args, args.model, args.student_pretrained, args.student_ckpt)
    student_model.cuda()

    teacher_model = build_model(
        args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
    teacher_model.cuda()

    # build kd loss
    from lib.models.losses.diffkd.diffkd import DiffKD
    diffkd = DiffKD(960, 2048, kernel_size=3, use_ae=True)
    ckpt_manager = CheckpointManager(diffkd.model,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': None,
                                         'dyrep': None
                                     })
    ckpt_manager.load(args.resume)
    diffkd.cuda()

    # tsne_visualization(student_model, 'cuda', val_loader)

    for batch_idx, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        student_feat, _ = student_model(input, is_feat=True)
        teacher_feat, _ = teacher_model(input, is_feat=True)
        student_feat = student_feat[-2]
        teacher_feat = teacher_feat[-2]

        refined_feat, teacher_feat, ddim_loss, _ = diffkd(
            student_feat, teacher_feat)
        attn_teacher, attn_student, attn_refined_student = get_attn_map(teacher_feat), get_attn_map(student_feat), get_attn_map(refined_feat)

        input = denormalize(input)
        input = torch.clamp(input, 0, 1)

        plt.figure()
        plt.subplot(141)
        plt.imshow(input[0].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(attn_teacher.detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(attn_student.detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(attn_refined_student.detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f'visualization/attn_map_{batch_idx}.png')


if __name__ == '__main__':
    main()