import numpy as np
import torch
import random
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats

from collections.abc import Sequence, Mapping
from .Build_Augmentation import augmentations


@augmentations.register_module()
class NormalizeColor(object):
    def __call__(self, data, labels=None):
        data[:, 6:9] = data[:, 6:9] / 127.5 - 1

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center"]
        self.mode = mode

    def __call__(self, data, labels=None):
        coord = data[:, 0:3]
        point_max = (
            int(self.sample_rate * coord.shape[0])
            if self.sample_rate is not None
            else self.point_max
        )
        # mode is "random" or "center"
        if coord.shape[0] > point_max:
            if self.mode == "random":
                center = coord[np.random.randint(coord.shape[0])]
            elif self.mode == "center":
                center = coord[coord.shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(coord - center), 1))[:point_max]
            data = data[idx_crop]
            if labels is not None:
                labels = labels[idx_crop]

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class GridSample(object):
    def __init__(self, grid_size=0.05, hash_type="fnv",):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec

    def __call__(self, data, labels=None):
        coord = data[:, 0:3]
        scaled_coord = coord / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        data = data[idx_unique]

        return (data, labels[idx_unique]) if labels is not None else data

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@augmentations.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data, labels=None):
        if np.random.rand() < self.p:
            noise = np.random.randn(data.shape[0], 3)
            noise *= self.std * 255
            data[:, 6:9] = np.clip(
                noise + data[:, 6:9], 0, 255
            )

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data, labels=None):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data[:, 6:9] = np.clip(tr + data[:, 6:9], 0, 255)

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data, labels=None):
        if np.random.rand() < self.p:
            lo = np.min(data[:, 6:9], 0, keepdims=True)
            hi = np.max(data[:, 6:9], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data[:, 6:9] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data[:, 6:9] = (1 - blend_factor) * data[:, 6:9] + blend_factor * contrast_feat

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data, labels=None):
        if random.random() < self.dropout_application_ratio:
            n = len(data)
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            data = data[idx]
            if labels:
                labels = labels[idx]

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class NormalizeCoord(object):
    def __call__(self, data, labels=None):
        coord = data[:, 0:3]
        centroid = np.mean(coord, axis=0)
        coord = coord - centroid
        m = np.max(np.sqrt(np.sum(coord ** 2, axis=1)))
        data[:, 0:3] = coord / m

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data, labels=None):
        coord = data[:, 0:3]
        x_min, y_min, z_min = coord.min(axis=0)
        x_max, y_max, _ = coord.max(axis=0)
        if self.apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data[:, 0:3] = coord - shift

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data, labels=None):
        if random.random() > self.p:
            return (data, labels) if labels is not None else data
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError

        coord = data[:, 0:3]
        if self.center is None:
            x_min, y_min, z_min = coord.min(axis=0)
            x_max, y_max, z_max = coord.max(axis=0)
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        else:
            center = self.center

        coord = np.dot(coord - center, np.transpose(rot_t))
        data[:, 0:3] = coord + center

        if data.shape[-1] > 3:
            data[:, 3:6] = np.dot(data[:, 3:6], np.transpose(rot_t))

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data, labels=None):
        scale = np.random.uniform(
            self.scale[0], self.scale[1], 3 if self.anisotropic else 1
        )
        data[:, 0:3] *= scale

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data, labels=None):
        shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
        shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
        shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
        data[:, 0:3] += [shift_x, shift_y, shift_z]

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data, labels=None):
        coord = data[:, 0:3]
        if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]
            if data.shape[-1] > 3:
                data[:, 3:6][:, 0] = -data[:, 3:6][:, 0]
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
            if data.shape[-1] > 3:
                data[:, 3:6][:, 1] = -data[:, 3:6][:, 1]
        data[:, 0:3] = coord

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data, labels=None):
        jitter = np.clip(
            self.sigma * np.random.randn(data[:, 0:3].shape[0], 3),
            -self.clip,
            self.clip,
        )
        data[:, 0:3] += jitter

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data, labels=None):
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                data[:, 0:3] = self.elastic_distortion(data[:, 0:3], granularity, magnitude)

        return (data, labels) if labels is not None else data


@augmentations.register_module()
class ShufflePoint(object):
    def __call__(self, data, labels=None):
        shuffle_index = np.arange(data.shape[0])
        np.random.shuffle(shuffle_index)
        data = data[shuffle_index]

        return (data, labels[shuffle_index]) if labels is not None else data


@augmentations.register_module()
class ToTensor(object):
    def __call__(self, data, labels=None):
        if isinstance(data, torch.Tensor):
            return (data, labels) if labels is not None else data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return (data, labels) if labels is not None else data
        elif isinstance(data, int):
            return (torch.LongTensor([data]), labels) if labels is not None else torch.LongTensor([data])
        elif isinstance(data, float):
            return (torch.FloatTensor([data]), labels) if labels is not None else torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return (torch.from_numpy(data), labels) if labels is not None else torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return (torch.from_numpy(data).long(), labels) if labels is not None else torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return (torch.from_numpy(data).float(), labels) if labels is not None else torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            return ({sub_key: self(item, labels) for sub_key, item in data.items()}, labels) if labels is not None else {sub_key: self(item, labels) for sub_key, item in data.items()}
        elif isinstance(data, Sequence):
            return ([self(item, labels) for item in data], labels) if labels is not None else [self(item, labels) for item in data]
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


