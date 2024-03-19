import glob
import os
import os.path as osp

import numpy as np
import torch.utils.data as data

__all__ = ["FlyingThings3DSubset","FlyingThings3DSubset_Occlusion"]

class FlyingThings3DSubset(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 full = True):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        print(self.root)
        self.train = train
        self.transform = transform
        self.num_points = num_points

        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
      
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')
        print(osp.exists(root))
        print(root)
        

        all_paths = os.walk(root)
        
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        return pc1, pc2

class FlyingThings3DSubset_Occlusion(data.Dataset):
    def __init__(self, train, transform, num_points, data_root, full=True):
        self.root = osp.join(data_root, "data_processed_maxcut_35_20k_2k_8192")
        print(self.root)
        self.train = train
        self.transform = transform
        self.num_points = num_points

        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, "TRAIN*.npz"))
        else:
            self.datapath = glob.glob(os.path.join(self.root, "TEST*.npz"))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [
            d for d in self.datapath if "TRAIN_C_0140_left_0006-0" not in d
        ]
        ######

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):

        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, "rb") as fp:
                data = np.load(fp)
                pos1 = data["points1"]
                pos2 = data["points2"]
                color1 = data["color1"] / 255
                color2 = data["color2"] / 255
                flow = data["flow"]
                mask1 = data["valid_mask1"]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)

            pos1_ = np.copy(pos1[sample_idx1, :])
            pos2_ = np.copy(pos2[sample_idx2, :])
            color1_ = np.copy(color1[sample_idx1, :])
            color2_ = np.copy(color2[sample_idx2, :])
            flow_ = np.copy(flow[sample_idx1, :])
            mask1_ = np.copy(mask1[sample_idx1])
        else:
            pos1_ = np.copy(pos1[: self.num_points, :])
            pos2_ = np.copy(pos2[: self.num_points, :])
            color1_ = np.copy(color1[: self.num_points, :])
            color2_ = np.copy(color2[: self.num_points, :])
            flow_ = np.copy(flow[: self.num_points, :])
            mask1_ = np.copy(mask1[: self.num_points])

        return pos1_, pos2_, color1_, color2_, flow_, mask1_

 

