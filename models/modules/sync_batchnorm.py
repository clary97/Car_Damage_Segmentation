# models/modules/sync_batchnorm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
import collections
import queue

_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])

class FutureResult(object):
    def __init__(self):
        self._event = torch.multiprocessing.Event()
        self._result = None

    def put(self, result):
        self._result = result
        self._event.set()

    def get(self):
        self._event.wait()
        return self._result

class _MasterRegistry(object):
    def __init__(self, future_result):
        self.result = future_result

class SlavePipe(object):
    def __init__(self, identifier, queue, future):
        self.identifier = identifier
        self.queue = queue
        self.future = future

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        return self.future.get()

class SyncMaster(object):
    def __init__(self, master_callback):
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        self._activated = True
        intermediates = [(0, master_msg)]
        for _ in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        for i, res in results:
            if i != 0:
                self._registry[i].result.put(res)

        for _ in range(self.nr_slaves):
            self._queue.get()

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                                self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = input.sum(dim=0).sum(dim=-1)
        input_ssum = (input ** 2).sum(dim=0).sum(dim=-1)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - mean[None, :, None]) * (inv_std[None, :, None] * self.weight[:, None]) + self.bias[:, None]
        else:
            output = (input - mean[None, :, None]) * inv_std[None, :, None]
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if copy_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [j for i in intermediates for j in (i[1].sum, i[1].ssum)]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum(i[1].sum_size for i in intermediates)
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean = sum_ / sum_size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (sum_size - 1)
        bias_var = sumvar / sum_size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        inv_std = bias_var.clamp(self.eps).rsqrt()
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        return [(rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])) for i, rec in enumerate(intermediates)]

class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super()._check_input_dim(input)

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super()._check_input_dim(input)
