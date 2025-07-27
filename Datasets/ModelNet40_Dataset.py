import os
import glob
import h5py
import pickle
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from Augmentations.Build_Augmentation import build_augmentation
from Utils.Logger import print_log
from Datasets.Build_Dataloader import datasets


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def load_modelnet_data(partition):
    BASE_DIR = './'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


@datasets.register_module()
class ModelNet40SVM(Dataset):
    def __init__(self, config):
        self.num_points = config.N_POINTS
        self.partition = config.partition
        self.data, self.label = load_modelnet_data(config.partition)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


@datasets.register_module()
class ModelNet(Dataset):
    def __init__(self, data_path, use_normals, num_points, num_category, mode, transform, loop=1):
        self.data_path = data_path
        self.use_normals = use_normals
        self.num_points = num_points
        self.num_category = num_category
        self.process_data = True
        self.uniform = True
        self.loop = loop
        split = mode
        self.process_points = 8192
        self.transform = build_augmentation(transform)

        if self.num_category == 10:
            self.catfile = os.path.join(self.data_path, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.data_path, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.data_path, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is {%d} x {%d}' % (split, len(self.datapath), self.loop), logger='ModelNet')

        if self.process_data:
            if self.uniform:
                self.save_path = os.path.join(self.data_path, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.process_points))
            else:
                self.save_path = os.path.join(self.data_path, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.process_points))

            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[fn[0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.process_points)
                    else:
                        point_set = point_set[0:self.process_points, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath) * self.loop

    def __getitem__(self, index):
        index = index % len(self.datapath)
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            label = np.array(cls).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.num_points)
        else:
            point_set = point_set[0:self.num_points, :]

        point_set = self.transform(point_set)

        return point_set, label[0]

# class ModelNet(Dataset):
#     def __init__(self, data_path, use_normals, num_points, num_category, mode, transform, loop=1):
#         self.data_path = data_path
#         self.use_normals = use_normals
#         self.num_points = num_points
#         self.num_category = num_category
#         self.process_data = True
#         self.uniform = True
#         self.loop = loop
#         split = mode
#         self.transform = build_augmentation(transform)
#
#         if self.num_category == 10:
#             self.catfile = os.path.join(self.data_path, 'modelnet10_shape_names.txt')
#         else:
#             self.catfile = os.path.join(self.data_path, 'modelnet40_shape_names.txt')
#
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#
#         shape_ids = {}
#         if self.num_category == 10:
#             shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet10_train.txt'))]
#             shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet10_test.txt'))]
#         else:
#             shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet40_train.txt'))]
#             shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.data_path, 'modelnet40_test.txt'))]
#
#         assert (split == 'train' or split == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
#         self.datapath = [(shape_names[i], os.path.join(self.data_path, shape_names[i], shape_ids[split][i]) + '.txt') for i
#                          in range(len(shape_ids[split]))]
#         print_log('The size of %s data is {%d} x {%d}' % (split, len(self.datapath), self.loop), logger='ModelNet')
#
#         self.cache = {}
#
#     def __len__(self):
#         return len(self.datapath) * self.loop
#
#     def __getitem__(self, index):
#         index = index % len(self.datapath)
#         if index in self.cache:
#             point_set, label = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cls = self.classes[fn[0]]
#             label = np.array(cls).astype(np.int32)
#             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
#             if not self.use_normals:
#                 point_set = point_set[:, 0:3]
#             point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#             self.cache[index] = (point_set, label)
#
#         if self.uniform:
#             point_set = farthest_point_sample(point_set, self.num_points)
#         else:
#             point_set = point_set[0:self.num_points, :]
#
#         point_set = self.transform(point_set)
#
#         return point_set, label
