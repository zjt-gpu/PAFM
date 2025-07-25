import math
from torch import inf
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateauWithWarmup(object):
    

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, warmup_lr=None,
                 warmup=0):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.warmup_lr = warmup_lr
        self.warmup = warmup
        
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _prepare_for_warmup(self):
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                if len(self.warmup_lr) != len(self.optimizer.param_groups):
                    raise ValueError("expected {} warmup_lrs, got {}".format(
                        len(self.optimizer.param_groups), len(self.warmup_lr)))
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None
        if self.warmup > self.last_epoch:
            curr_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i])/float(self.warmup)) for i in range(len(curr_lrs))]
        else:
            self.warmup_lr_steps = None

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr + self.warmup_lr_steps[i], self.min_lrs[i])
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: increasing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else: 
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self._prepare_for_warmup()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class CosineAnnealingLRWithWarmup(object):
    

    def __init__(self, optimizer, T_max, last_epoch=-1, verbose=False,
                 min_lr=0, warmup_lr=None, warmup=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.warmup_lr = warmup_lr
        self.warmup = warmup

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.max_lrs = [lr for lr in self.min_lrs]
        
        self._prepare_for_warmup()

    def step(self):
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = float(epoch - self.warmup) / float(max(1, self.T_max - self.warmup))
            factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            old_lr = float(param_group['lr'])
            new_lr = max(self.max_lrs[i] * factor, self.min_lrs[i])
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: reducing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _increase_lr(self, epoch):
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr + self.warmup_lr_steps[i]
            param_group['lr'] = new_lr
            self.max_lrs[i] = max(self.max_lrs[i], new_lr)
            if self.verbose:
                print('Epoch {:5d}: increasing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _prepare_for_warmup(self):
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                if len(self.warmup_lr) != len(self.optimizer.param_groups):
                    raise ValueError("expected {} warmup_lrs, got {}".format(
                        len(self.optimizer.param_groups), len(self.warmup_lr)))
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None
        if self.warmup > self.last_epoch:
            curr_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i])/float(self.warmup)) for i in range(len(curr_lrs))]
        else:
            self.warmup_lr_steps = None


    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._prepare_for_warmup()