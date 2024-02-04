import os, shutil
import pdb
import torch, math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def Save_Checkpoint(state, last, last_path, best, best_path, is_best):
    if os.path.exists(last):
        shutil.rmtree(last)
    last_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(last_path, 'ckpt.pth'))

    if is_best:
        if os.path.exists(best):
            shutil.rmtree(best)
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(state, os.path.join(best_path, 'ckpt.pth'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, weight_decay_head=1.0, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    head = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            if 'head' in name:
                head.append(param)
            else:
                decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': head, 'weight_decay': weight_decay_head}]


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
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


class _Catcher():
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = {}

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def register_model_hooks(self, catch_dic):
        for name, module in self.model.named_modules():
            if name in catch_dic:
                k = catch_dic[name]
                self.features[k] = None
                self.hooks[k] = module.register_forward_hook(self._get_hook(k))

        if not self.hooks.keys() == catch_dic.values():
            # print(self.hooks.keys())
            print("unfound features: {}".format(catch_dic.values() - self.hooks.keys()))

    def get_features(self, key):
        return self.features[key]
 

class RepCatcher(_Catcher):
    def __init__(self, model, layers_idx):
        super(RepCatcher, self).__init__(model)
        self.layers_idx = layers_idx
        # assert len(layers_idx) == self.model.depth
        catch_dic = self._get_rep_catch_dic(layers_idx)
        self.register_model_hooks(catch_dic)

    def _get_rep_catch_dic(self, idx):
        out = {}
        # [0, 1, 2, ..., 11]
        for i in idx:
            if i == 0:
                out["patch_embed"] = "rep{}".format(i)
            else:
                out["blocks.{}".format(i-1)] = "rep{}".format(i)
        return out
    
    def get_features(self, idx=None):
        # pdb.set_trace()
        if idx is None:
            idx = self.layers_idx
        return_list = True
        if isinstance(idx, int):
            idx = [idx]
            return_list = False
        reps = []
        for i in idx:
            rep = self.features["rep{}".format(i)]
            if i == 0:
                # patch embed
                reps.append(rep)
            else:
                rep = rep[:, 1:, :]   # remove cls token
                reps.append(rep)

        return reps if return_list else reps[0]
