import torch
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
    flatten_params, flatten_stats, batch_norm


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = torch.Tensor([16, 32, 64]).mul(width).int().numpy().tolist()

    def gen_block_params(ni, no, scalar):
        if scalar:
            return {
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'bn2': bnparams(no),
            }
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count, bias=False):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no, bias)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no), 'bn2': bnstats(no)}
                for i in range(count)}

    flat_vectors = flatten_params({
        'conv0': conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'conv1': conv_params(widths[2], num_classes, 1),
    })

    flat_scalars = flatten_params({
        'group0': gen_group_params(16, widths[0], n, True),
        'group1': gen_group_params(widths[0], widths[1], n, True),
        'group2': gen_group_params(widths[1], widths[2], n, True),
        'bn': bnparams(widths[2]),
    })

    flat_stats = flatten_stats({
        'group0': gen_group_stats(16, widths[0], n),
        'group1': gen_group_stats(widths[0], widths[1], n),
        'group2': gen_group_stats(widths[1], widths[2], n),
        'bn': bnstats(widths[2]),
    })

    def block_ncmn0(x, params, stats, base, mode, stride, noise_std):
        o1 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode, 1., noise_std), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, stats, base + '.bn1', mode, 1., noise_std), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def block_ncmn1(x, params, stats, base, mode, stride, noise_std):
        o1 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode, 1.), inplace=True)
        y = lin_layer(o1, params, stats, base, mode, stride, 0)
        if mode:
            y_n = lin_layer(o1, params, stats, base, mode, stride, 0, noise_std)
            y = y + y_n.sub_(y).detach_()
        o2 = F.relu(y, inplace=True)
        z = lin_layer(o2, params, stats, base, mode, 1, 1)
        if mode:
            z_n = lin_layer(o2, params, stats, base, mode, 1, 1, noise_std)
            z = z + z_n.sub_(z).detach_()
        if base + '.convdim' in params:
            return z + F.conv2d(o1 if stride == 1 else x, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def lin_layer(x, params, stats, base, mode, stride, layer, noise_std=0):
        x = corrupt(x, noise_std)
        z = F.conv2d(x, params[base + '.conv' + str(layer)], stride=stride, padding=1)
        return batch_norm(z, params, stats, base + '.bn' + str(layer + 1), mode, 1., ema=noise_std or not mode)

    def block_ncmn2(x, params, stats, base, mode, stride, noise_std):
        o1 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode, 1.), inplace=True)
        z = res_branch(o1, params, stats, base, mode, stride)
        if mode:
            z_n = res_branch(o1, params, stats, base, mode, stride, noise_std)
            z = z + z_n.sub_(z).detach_()
        if base + '.convdim' in params:
            return z + F.conv2d(o1 if stride == 1 else x, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def res_branch(o1, params, stats, base, mode, stride, noise_std=0):
        o1 = corrupt(o1, noise_std)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, stats, base + '.bn1', mode, 1.), inplace=True)
        o2 = corrupt(o2, noise_std)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        return batch_norm(z, params, stats, base + '.bn2', mode, 1., ema=noise_std or not mode)

    def corrupt(x, noise_std):
        if not noise_std:
            return x
        Tensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        noise = Tensor(*x.size()[0:2], 1, 1).uniform_(1 - noise_std * 3 ** 0.5, 1 + noise_std * 3 ** 0.5)
        return x * noise

    def group(o, params, stats, base, ncmn, mode, stride):
        assert ncmn[0] in (0, 1, 2)
        if ncmn[0] == 0 or ncmn[1] == 0:
            block = block_ncmn0
        elif ncmn[0] == 1:
            block = block_ncmn1
        elif ncmn[0] == 2:
            block = block_ncmn2
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base, i), mode, stride if i == 0 else 1, ncmn[1])
        return o

    def f(input, params, stats, ncmn, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', ncmn, mode, 1)
        g1 = group(g0, params, stats, 'group1', ncmn, mode, 2)
        g2 = group(g1, params, stats, 'group2', ncmn, mode, 2)
        o = F.relu(batch_norm(g2, params, stats, 'bn', mode, 1.))
        o = F.conv2d(o, params['conv1'])
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        return o

    return f, flat_vectors, flat_scalars, flat_stats
