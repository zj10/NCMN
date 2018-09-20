import torch
import torch.cuda.comm as comm
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return cast(kaiming_normal_(torch.Tensor(no, ni, k, k)))


def linear_params(ni, no):
    return cast({'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})


def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})


def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})


def data_parallel(f, input, params, stats, ncmn, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, ncmn, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k, v in param_dict.items():
            for i, u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast.apply(device_ids, x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p, s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode, const_scale=None, noise_std=0, ema=True):
    x = F.batch_norm(x, weight=None,
                     bias=None,
                     running_mean=stats[base + '.running_mean'] if ema else None,
                     running_var=stats[base + '.running_var'] if ema else None,
                     training=mode)
    Tensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    weight = params[base + '.weight'] if const_scale is None else Variable(Tensor(x.size(1)).fill_(const_scale))
    if mode and noise_std:
        noise = Tensor(*x.size()[0:2], 1, 1).uniform_(-noise_std * 3 ** 0.5, noise_std * 3 ** 0.5)
        x = x.add(x.mul(noise).detach_())
    return x.mul(weight.view(x.size(1), 1, 1)).add(params[base + '.bias'].view(x.size(1), 1, 1))


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3),
              str(tuple(v.size())).ljust(23), torch.typename(v.data if isinstance(v, Variable) else v))
