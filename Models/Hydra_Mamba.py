import torch
import pointops

from copy import deepcopy
from functools import partial
from knn_cuda import KNN
from pointops import offset2batch, batch2offset
from timm.layers import trunc_normal_
from torch import nn
from einops import rearrange
from timm.models.layers import DropPath
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

from Models.Multihead_SSM import MultiheadSSM
from Utils import Misc
from Utils.Serialization.default import hilbert_encode
from Utils.Tool import index_points
from Models.Build_Model import models
from Utils.pointnet_util import PointNetFeaturePropagation


class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class Serialization(object):
    def __init__(
            self,
            coord,
            serial_mode,
            order,
            grid_size,
            assign=None
    ):
        super(Serialization, self).__init__()
        self.serial_mode = serial_mode
        self.grid_size = grid_size
        self.order = order
        self.order_index = -1
        self.assign = assign
        self.code = self.hilbert(coord)

    def __call__(self):
        if self.serial_mode == 'single':
            self.order_index = 0 if not self.assign else self.assign % len(self.order)
        elif self.serial_mode == 'cyclic':
            self.order_index = (self.order_index + 1) % len(self.order)
        elif self.serial_mode == 'random':
            self.order_index = torch.randperm(len(self.order))[0]
        else:
            raise NotImplementedError

        return self.code[self.order[self.order_index]]

    def update(self, index):
        assert len(index.shape) == 2
        self.code = {key: torch.gather(value, 1, index.long()) for key, value in self.code.items()}

    def retrieve(self, code):
        self.code = code

    def hilbert(self, coord):
        assert len(coord.shape) == 2 or len(coord.shape) == 3
        batch = None if len(coord.shape) < 3 else self.offset2batch(coord)
        if batch is not None:
            coord = rearrange(coord, 'b n d -> (b n) d')

        grid_coord = torch.div(
            coord - coord.min(0)[0], self.grid_size, rounding_mode="trunc"
        ).int()
        depth = int(grid_coord.max()).bit_length()
        assert (depth * 3 + len(batch).bit_length() <= 63) & (depth <= 16)

        return {order: self.hilbert_(order, grid_coord, depth, batch) for order in self.order}

    @torch.inference_mode()
    def hilbert_(self, order, grid_coord, depth, batch):
        # assert order in {'xyz, xzy, yxz, yzx, zxy, zyx'}
        if order == 'hilbert_xyz':
            code = hilbert_encode(grid_coord, depth=depth)
        elif order == 'hilbert_xzy':
            code = hilbert_encode(grid_coord[:, [0, 2, 1]], depth=depth)
        elif order == 'hilbert_yxz':
            code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
        elif order == 'hilbert_yzx':
            code = hilbert_encode(grid_coord[:, [1, 2, 0]], depth=depth)
        elif order == 'hilbert_zxy':
            code = hilbert_encode(grid_coord[:, [2, 0, 1]], depth=depth)
        elif order == 'hilbert_zyx':
            code = hilbert_encode(grid_coord[:, [2, 1, 0]], depth=depth)
        else:
            raise NotImplementedError

        if batch is not None:
            batch = batch.long()
            code = rearrange(batch << depth * 3 | code, '(b n) -> b n', b=len(torch.unique(batch)))

        return code

    @torch.inference_mode()
    def offset2batch(self, coord):
        offset = coord.shape[1] * (torch.arange(coord.shape[0]) + 1)
        bincount = torch.diff(
            offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
        )
        return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount).to(coord.device)


class TokenMixer(nn.Module):
    def __init__(
            self,
            ssm_mode: str,
            locality,
            in_channels: int,
            num_heads: int,
            d_state,
            expand,
            drop_path,
            kernel_size=None,
            conv_stride=None,
            padding=None,
    ):
        super(TokenMixer, self).__init__()
        self.ssm_mode = ssm_mode
        self.locality = locality
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.padding = padding

        self.forwardSSM, self.backwardSSM = [
            MultiheadSSM(
                in_channels,
                num_heads,
                d_state,
                expand
            ) for _ in range(2)
        ]

        if locality:
            self.conv = nn.Sequential(
                SwapAxes(),
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    conv_stride,
                    padding,
                    groups=1 if locality == 'Conv'
                    else in_channels
                ),
                nn.GELU(),
                SwapAxes(),
            )

        self.norm = nn.LayerNorm(in_channels)

    def cascaded(self, inputs):
        return self.backwardSSM(
            self.norm(self.forwardSSM(inputs)).flip(dims=(-2,))
        ).flip(dims=(-2,))

    def parallel(self, inputs):
        return self.forwardSSM(inputs) + self.backwardSSM(inputs.flip(dims=(-2,))).flip(dims=(-2,))

    def forward(self, inputs):
        """
            inputs: (batch, seqlen, in_channels_dec)
        """
        assert self.ssm_mode in {'Cascaded', 'Parallel'}
        assert self.locality in {None, 'Conv', 'DWConv'}

        outputs = self.cascaded(inputs) if self.ssm_mode == 'Cascaded' else self.parallel(inputs)
        if self.locality:
            assert (inputs.shape[-2] + 2 * self.padding - self.kernel_size) // self.conv_stride + 1 == inputs.shape[-2]
            outputs = outputs + self.conv(inputs)

        return outputs


class ChannelMixer(nn.Module):
    def __init__(
            self,
            in_channels,
            mlp_ratio=0.0,
    ):
        super(ChannelMixer, self).__init__()
        out_channels = in_channels
        hidden_channels = in_channels * int(mlp_ratio)

        self.mlp = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, 1),
            SwapAxes(),
        )

    def forward(self, x):
        return self.mlp(x)


class HydraMambaBlock(nn.Module):
    def __init__(
            self,
            ssm_mode: str,
            locality,
            in_channels: int,
            num_heads: int,
            d_state,
            expand,
            drop_path,
            kernel_size=None,
            conv_stride=None,
            padding=None,
            mlp_ratio=0.0,
            pre_norm=True
    ):
        super(HydraMambaBlock, self).__init__()
        self.pre_norm = pre_norm

        self.norm1 = nn.LayerNorm(in_channels)
        self.biSSM = TokenMixer(
            ssm_mode,
            locality,
            in_channels,
            num_heads,
            d_state,
            expand,
            drop_path,
            kernel_size,
            conv_stride,
            padding
        )

        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = ChannelMixer(
            in_channels,
            mlp_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, points):
        shortcut = points
        if self.pre_norm:
            points = self.norm1(points)
        points = self.biSSM(points)
        points = self.drop_path(points) + shortcut
        if not self.pre_norm:
            points = self.norm1(points)

        shortcut = points
        if self.pre_norm:
            points = self.norm2(points)
        points = self.mlp(points)
        points = self.drop_path(points) + shortcut
        if not self.pre_norm:
            points = self.norm2(points)

        return points


class Group(nn.Module):
    def __init__(self, group_size, num_group=None):
        super().__init__()
        self.group_size = group_size
        self.num_group = num_group
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center_idx, center = Misc.fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        return idx, center_idx, center


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, inputs):
        '''
            inputs : B G N D
            -----------------
            feature_global : B G C
        '''
        b, g, n, d = inputs.shape
        inputs = rearrange(inputs, 'b g n d -> (b g) n d')
        # encoder
        feature = self.first_conv(inputs.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(b, g, -1)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, num_group):
        super(DownSampling, self).__init__()
        self.group_divider = Group(group_size=group_size, num_group=num_group)
        self.encoder = Encoder(in_channels, out_channels)

    def forward(self, xyz, features):
        idx, center_idx, center = self.group_divider(xyz)
        # center point features plus pos info
        group_input = self.encoder(index_points(features, idx))
        return center_idx, center, group_input


class UpSampling(nn.Module):
    def __init__(self, in_channels_dec, in_channels_enc, out_channels):
        super(UpSampling, self).__init__()

        self.fc1 = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels_dec, out_channels, 1),
            nn.BatchNorm1d(out_channels),  # TODO
            nn.ReLU(inplace=True),
            SwapAxes(),
        )
        self.fc2 = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels_enc, out_channels, 1),
            nn.BatchNorm1d(out_channels),  # TODO
            nn.ReLU(inplace=True),
            SwapAxes(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        feats2 = self.fc2(points2)
        return xyz2, feats1 + feats2


class GridPooling(nn.Module):
    """
        Partition-based Pooling (Grid Pooling)
    """
    def __init__(self, in_channels, out_channels, grid_size):
        super(GridPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            SwapAxes(),
        )

    def forward(self, points, start=None):
        xyz, features, offset = points
        batch = offset2batch(offset)
        features = self.fc(features)
        xyz, features = xyz.squeeze(0), features.squeeze(0)
        # min value per segment
        start = (
            segment_csr(
                xyz,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            ) if start is None else start
        )
        cluster = voxel_grid(pos=xyz - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        xyz = segment_csr(xyz[sorted_cluster_indices], idx_ptr, reduce="mean")
        features = segment_csr(features[sorted_cluster_indices], idx_ptr, reduce="max")

        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [xyz.unsqueeze(0), features.unsqueeze(0), offset], cluster


class GridUnpooling(nn.Module):
    """
        Map Unpooling with skip connection
    """

    def __init__(self, in_channels_dec, in_channels_enc, out_channels, backend="map"):
        super(GridUnpooling, self).__init__()
        self.in_channels = in_channels_dec
        self.skip_channels = in_channels_enc
        self.out_channels = out_channels
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.fc1 = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels_dec, out_channels, 1),
            nn.BatchNorm1d(out_channels),  # TODO
            nn.ReLU(inplace=True),
            SwapAxes()
        )
        self.fc2 = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels_enc, out_channels, 1),
            nn.BatchNorm1d(out_channels),  # TODO
            nn.ReLU(inplace=True),
            SwapAxes(),
        )

    def forward(self, points, skip_points, cluster=None):
        xyz, features, offset = points
        skip_xyz, skip_features, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            features = self.fc1(features)[:, cluster]
        elif self.backend == "interp":
            features = pointops.interpolation(xyz, skip_xyz, self.fc1(features), offset, skip_offset).unsqueeze(0)
        else:
            raise NotImplementedError

        features = features + self.fc2(skip_features)
        return [skip_xyz, features, skip_offset]


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Embedding, self).__init__()

        self.stem = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            SwapAxes()
        )

    def forward(self, inputs):
        return self.stem(inputs)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


@models.register_module()
class HydraMambaReg(nn.Module):
    def __init__(
            self,
            in_channels,
            serial_mode,
            grid_size,
            order,
            stride,
            group_size,
            enc_depths,
            enc_channels,
            enc_num_head,
            mlp_ratio,
            pre_norm,
            d_state,
            expand,
            ssm_mode,
            locality,
            kernel_size,
            conv_stride,
            padding,
            drop_path,
            num_points,
            num_classes
    ):
        super(HydraMambaReg, self).__init__()
        self.num_points = num_points
        self.enc_depths = enc_depths

        self.emb = Embedding(in_channels, enc_channels[0])

        self.serial = partial(
            Serialization,
            serial_mode=serial_mode,
            order=order,
            grid_size=grid_size
        )

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.down_sampling, self.blocks_enc = nn.ModuleList(), nn.ModuleList()
        for i in range(len(enc_depths)):
            enc_drop_path_stage = enc_drop_path[sum(enc_depths[:i]): sum(enc_depths[: i + 1])]
            if i > 0:
                self.num_points //= stride[i - 1]
                self.down_sampling.append(
                    DownSampling(enc_channels[i - 1], enc_channels[i], group_size, self.num_points)
                )

            blocks = nn.ModuleList()
            for j in range(enc_depths[i]):
                blocks.append(
                    HydraMambaBlock(
                        ssm_mode,
                        locality,
                        enc_channels[i],
                        enc_num_head[i],
                        d_state,
                        expand,
                        enc_drop_path_stage[j],
                        kernel_size,
                        conv_stride,
                        padding,
                        mlp_ratio,
                        pre_norm,
                    )
                )
            self.blocks_enc.append(blocks)

        self.cls_head = nn.Sequential(
            nn.Linear(enc_channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        self.apply(_init_weights)

    def forward(self, inputs):
        xyz = inputs[..., 0:3]
        features = self.emb(inputs)
        serial = self.serial(xyz)

        for i in range(len(self.enc_depths)):
            if i > 0:
                center_idx, xyz, group_input = self.down_sampling[i - 1](xyz, features)
                serial.update(center_idx)
            else:
                group_input = features

            for j in range(self.enc_depths[i]):
                serial_code = serial()
                sort = torch.argsort(serial_code, dim=-1)
                retrieve = torch.argsort(sort, dim=-1)
                features = index_points(self.blocks_enc[i][j](
                    index_points(group_input, sort)
                ), retrieve)

        return self.cls_head(features.mean(dim=1))


@models.register_module()
class HydraMambaSegV1(nn.Module):
    def __init__(
            self,
            in_channels,
            serial_mode,
            grid_size,
            order,
            stride,
            group_size,
            enc_depths,
            enc_channels,
            enc_num_head,
            dec_depths,
            dec_channels,
            dec_num_head,
            mlp_ratio,
            pre_norm,
            d_state,
            expand,
            ssm_mode,
            locality,
            kernel_size,
            conv_stride,
            padding,
            drop_path,
            num_points,
            num_category,
            num_parts
    ):
        super(HydraMambaSegV1, self).__init__()
        assert len(enc_depths) == len(dec_depths)

        self.num_points = num_points
        self.enc_depths = enc_depths
        self.dec_depths = dec_depths

        self.emb = Embedding(in_channels, enc_channels[0])
        self.serial = partial(
            Serialization,
            serial_mode=serial_mode,
            order=order,
            grid_size=grid_size
        )

        self.down_sampling, self.blocks_enc = nn.ModuleList(), nn.ModuleList()
        for i in range(len(enc_depths)):
            if i > 0:
                self.num_points //= stride[i - 1]
                self.down_sampling.append(
                    DownSampling(enc_channels[i - 1], enc_channels[i], group_size, self.num_points)
                )

            blocks = nn.ModuleList()
            for _ in range(enc_depths[i]):
                blocks.append(
                    HydraMambaBlock(
                        ssm_mode,
                        locality,
                        enc_channels[i],
                        enc_num_head[i],
                        d_state,
                        expand,
                        drop_path,
                        kernel_size,
                        conv_stride,
                        padding,
                        mlp_ratio,
                        pre_norm,
                    )
                )
            self.blocks_enc.append(blocks)

        self.up_sampling, self.blocks_dec = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dec_depths)):
            if i > 0:
                self.up_sampling.append(
                    UpSampling(dec_channels[i - 1], enc_channels[-i - 1], dec_channels[i])
                )

            blocks = nn.ModuleList()
            for _ in range(dec_depths[i]):
                blocks.append(
                    HydraMambaBlock(
                        ssm_mode,
                        locality,
                        dec_channels[i],
                        dec_num_head[i],
                        d_state,
                        expand,
                        drop_path,
                        kernel_size,
                        conv_stride,
                        padding,
                        mlp_ratio,
                        pre_norm,
                    )
                )
            self.blocks_dec.append(blocks)

        self.seg_head = nn.Sequential(
            SwapAxes(),
            nn.Conv1d(dec_channels[-1] + num_category, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_parts, 1),
            SwapAxes(),
        )

        self.apply(_init_weights)

    def forward(self, inputs, cat_prompt):
        xyz = inputs[..., 0:3]
        features = self.emb(inputs)
        serial = self.serial(xyz)

        # encoding
        xyz_features_code = []
        for i in range(len(self.enc_depths)):
            if i > 0:
                center_idx, xyz, group_input = self.down_sampling[i - 1](xyz, features)
                serial.update(center_idx)
            else:
                group_input = features

            for j in range(self.enc_depths[i]):
                serial_code = serial()
                sort = torch.argsort(serial_code, dim=-1)
                retrieve = torch.argsort(sort, dim=-1)
                features = index_points(self.blocks_enc[i][j](
                    index_points(group_input, sort)
                ), retrieve)
            xyz_features_code.append((xyz, features, deepcopy(serial.code)))

        # decoding
        xyz, features, _ = xyz_features_code[-1]
        for i in range(len(self.dec_depths)):
            if i > 0:
                xyz, group_input = self.up_sampling[i - 1](
                    xyz,
                    features,
                    xyz_features_code[-i - 1][0],
                    xyz_features_code[-i - 1][1]
                )
                serial.retrieve(xyz_features_code[-i - 1][-1])
            else:
                group_input = features

            for j in range(self.dec_depths[i]):
                serial_code = serial()
                sort = torch.argsort(serial_code, dim=-1)
                retrieve = torch.argsort(sort, dim=-1)
                features = index_points(self.blocks_dec[i][j](
                    index_points(group_input, sort)
                ), retrieve)

        cat_prompt = cat_prompt.unsqueeze(dim=1).repeat(1, features.shape[1], 1)

        return self.seg_head(torch.cat([features, cat_prompt], -1))
