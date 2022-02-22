import os.path as osp
import platform
import shutil
import time
import warnings
import copy
from collections import OrderedDict

import torch
import torch.distributed as dist

import mmcv
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from torch.utils import data


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            for name, value in loss_value.items():
                log_vars[name] = value
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

def parse_losses_sim(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            for name, value in loss_value.items():
                log_vars[name] = value
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    # 计算包含特征相似度的loss
    loss = log_vars['loss_cls'] + log_vars['sim']

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

def get_weights(model):
    for name, param in model.named_parameters():
        if '.thres' not in name:
            yield param

def get_thres(model):
    for name, param in model.named_parameters():
        if '.thres' in name:
            yield param

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        # loss = self.net.loss(trn_X, trn_y) # L_trn(w)
        losses = self.net(img=trn_X, gt_label=trn_y)
        loss, _ = parse_losses(losses)

        # compute gradient
        gradients = torch.autograd.grad(loss, get_weights(self.net))

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(get_weights(self.net), get_weights(self.v_net), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(get_thres(self.net), get_thres(self.v_net)):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        # loss = self.v_net.loss(val_X, val_y)
        # # L_val(w`)
        losses = self.v_net(img=val_X, gt_label=val_y)
        loss, _ = parse_losses_sim(losses)

        # compute gradient
        v_alphas = tuple(get_thres(self.v_net))
        v_weights = tuple(get_weights(self.v_net))
        # breakpoint()
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(get_thres(self.net), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p += eps * d
        losses = self.net(img=trn_X, gt_label=trn_y)
        loss, _ = parse_losses(losses)
        dalpha_pos = torch.autograd.grad(loss, get_thres(self.net)) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p -= 2. * eps * d
        losses = self.net(img=trn_X, gt_label=trn_y)
        loss, _ = parse_losses(losses)
        dalpha_neg = torch.autograd.grad(loss, get_thres(self.net)) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(get_weights(self.net), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


@RUNNERS.register_module()
class BilevelEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, train_data_batch, val_data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            pass
            # outputs = self.batch_processor(
            #     self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # breakpoint()
            # 优化阈值
            self.optimizer['thres'].zero_grad()
            self.architect.unrolled_backward(train_data_batch['img'], train_data_batch['gt_label'],
                                             val_data_batch['img'], val_data_batch['gt_label'],
                                             self.optimizer['weight'].param_groups[0]['lr'], self.optimizer['weight'])
            self.optimizer['thres'].step()

            # 优化权重
            self.optimizer['weight'].zero_grad()
            outputs = self.model.train_step(train_data_batch, self.optimizer['weight'],
                                            **kwargs)
            outputs['loss'].backward()
            self.optimizer['weight'].step()
        else:
            outputs = self.model.val_step(train_data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        # 指定runner.data_loader是优化权重使用的data_loader
        self.data_loader = data_loaders[0]
        self._max_iters = self._max_epochs * len(data_loaders[0])
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, (train_data_batch, val_data_batch) in enumerate(zip(*data_loaders)):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(train_data_batch, val_data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args: 
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                of alphas and weights
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        # 指定2个dataloader
        assert len(data_loaders) == 2
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                # iter数由训练weight的dataloader决定
                self._max_iters = self._max_epochs * len(data_loaders[0])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)

        # 定义用来计算thres梯度的类
        self.architect = Architect(self.model, 0.9, 1e-5)

        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    # give self.train all two data loaders
                    epoch_runner(data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
