from torch.utils.tensorboard import SummaryWriter
import math
import torch
import torch.nn as nn
from functools import partial
import json

from .kl_div import KLDivergence
from .dist_kd import DIST
from .diffkd import DiffKD

import logging
logger = logging.getLogger()



KD_MODULES = {
    'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
    'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),
    'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_ResNet50': dict(modules=['layer4', 'linear'], channels=[2048, 100]), # N*2048*4*4
    'cifar_resnet8x4': dict(modules=['layer3', 'fc'], channels=[256, 100]), # N*256*8*8
    'cifar_resnet32x4': dict(modules=['layer3', 'fc'], channels=[256, 100]), # N*256*8*8
    'cifar_MobileNetV2': dict(modules=['conv2', 'classifier'], channels=[1280, 100]), # N*1280*2*2
    'cifar_ShuffleV1': dict(modules=['layer3', 'linear'], channels=[960, 100]), # N*960*4*4
    'cifar_ShuffleV2': dict(modules=['relu', 'linear'], channels=[1024, 100]), # N*1024*4*4
    'tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 1000]),
    'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  # mbv2
    'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
    'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),
}


class DirectNormLoss(nn.Module):

    def __init__(self, num_class=100, nd_loss_factor=1.0):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        self.nd_loss_factor = nd_loss_factor

    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss

    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(
            s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor

        return nd_loss / len(labels)


class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(
        self,
        student,
        teacher,
        student_name,
        teacher_name,
        ori_loss,
        kd_method='kdt4',
        ori_loss_weight=1.0,
        nd_loss_factor=1.0,
        kd_loss_weight=1.0,
        kd_loss_kwargs={},
        tensorboard_writer=None
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        self.nd_loss_factor = nd_loss_factor
        self.tensorboard_writer = tensorboard_writer

        self._teacher_out = None
        self._student_out = None

        # init kd loss
        # module keys for distillation. '': output logits
        teacher_modules = ['',]
        student_modules = ['',]
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        elif kd_method == 'diffkd':
            # get configs
            ae_channels = kd_loss_kwargs.get('ae_channels', 1024)
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)

            # print(kd_loss_kwargs)
            kernel_sizes = [3, 1]  # distillation on feature and logits
            student_modules = KD_MODULES[student_name]['modules']
            student_channels = KD_MODULES[student_name]['channels']
            teacher_modules = KD_MODULES[teacher_name]['modules']
            teacher_channels = KD_MODULES[teacher_name]['channels']

            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            self.nd_loss = DirectNormLoss(
                num_class=100, nd_loss_factor=nd_loss_factor).cuda()
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
                self.diff[tm] = DiffKD(sc, tc, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels)
                self.kd_loss[tm] = nn.MSELoss() if ks != 1 else KLDivergence(tau=tau)
            self.diff.cuda()
            # add diff module to student for optimization
            self.student._diff = self.diff
        elif kd_method == 'mse':
            # distillation on feature
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
            self.align.cuda()
            # add align module to student for optimization
            self.student._align = self.align
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # # register forward hook
        # # dicts that store distillation outputs of student and teacher
        self._teacher_out = {}
        self._student_out = {}

        for student_module, teacher_module in zip(student_modules, teacher_modules):
            self._register_forward_hook(student, student_module, teacher=False)
            self._register_forward_hook(teacher, teacher_module, teacher=True)
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules

        teacher.eval()
        self._iter = 0

    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits = self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        kd_loss = 0
        nd_loss = 0

        for tm, sm in zip(self.teacher_modules, self.student_modules):

            # nd_loss for resnet56 --> resnet20
            if tm == 'layer3':
                with open("teacher_embedding/cifar100_embedding_fea/resnet56.json", 'r') as f:
                    T_EMB = json.load(f)
                f.close()
                avg_pool = nn.AvgPool2d(8)
                s_emb = avg_pool(self._student_out[sm][0])
                s_emb = s_emb.view(s_emb.size(0), -1)
                t_emb = avg_pool(self._teacher_out[sm][0])
                t_emb = t_emb.view(t_emb.size(0), -1)

                nd_loss = self.nd_loss(
                    s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=targets)

            # transform student feature
            if self.kd_method == 'diffkd':
                if tm in ['layer3', 'layer4']:
                    self._student_out[sm], self._teacher_out[tm], diff_loss, ae_loss = \
                        self.diff[tm](self._reshape_BCHW(
                            self._student_out[sm][0]), self._reshape_BCHW(self._teacher_out[tm][0]))
                else:
                    self._student_out[sm], self._teacher_out[tm], diff_loss, ae_loss = \
                        self.diff[tm](self._reshape_BCHW(self._student_out[sm]), self._reshape_BCHW(self._teacher_out[tm]))
            if hasattr(self, 'align'):
                self._student_out[sm] = self.align(self._student_out[sm])

            # compute kd loss
            if isinstance(self.kd_loss, nn.ModuleDict):
                kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm])
            else:
                kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm])

            if self.kd_method == 'diffkd':
                # add additional losses in DiffKD
                if ae_loss is not None:
                    kd_loss += diff_loss + ae_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f} AE loss: {ae_loss.item():.4f}')
                        self.tensorboard_writer.add_scalar(
                            'Loss/kd_loss', kd_loss_.item(), self._iter)
                        self.tensorboard_writer.add_scalar(
                            'Loss/diff_loss', diff_loss.item(), self._iter)
                        self.tensorboard_writer.add_scalar(
                            'Loss/ae_loss', ae_loss.item(), self._iter)
                else:
                    kd_loss += diff_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f}')
                        self.tensorboard_writer.add_scalar(
                            'Loss/kd_loss', kd_loss_.item(), self._iter)
                        self.tensorboard_writer.add_scalar(
                            'Loss/diff_loss', diff_loss.item(), self._iter)
            else:
                if self._iter % 50 == 0:
                    logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f}')
                    self.tensorboard_writer.add_scalar(
                        'Loss/kd_loss', kd_loss_.item(), self._iter)
            kd_loss += kd_loss_

        kd_loss += nd_loss
        self.tensorboard_writer.add_scalar(
            'Loss/nd_loss', nd_loss.item(), self._iter)
        
        self._teacher_out = {}
        self._student_out = {}

        self._iter += 1
        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight

    def _register_forward_hook(self, model, name, teacher=False):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    def _forward_hook(self, module, input, output, name, teacher=False):
        if teacher:
            self._teacher_out[name] = output[0] if len(output) == 1 else output
        else:
            self._student_out[name] = output[0] if len(output) == 1 else output

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x