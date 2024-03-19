import glob
import os
import os.path as osp
import torch.utils.data as data

import numpy as np
__all__ = ['KITTI','KITTI_Occlusion']

class KITTI(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True):
        self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        print(self.root)
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
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
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

class KITTI_Occlusion(data.Dataset):
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True,npoints=16384):
        self.npoints = npoints
        
        
        self.root = osp.join(data_root, 'kitti_rm_ground')

        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground
        
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data["pos1"][:, (1, 2, 0)]
                pos2 = data["pos2"][:, (1, 2, 0)]
                flow = data["gt"][:, (1, 2, 0)]
                
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        loc1 = pos1[:,2] < 35
        pos1 = pos1[loc1]
        
        flow = flow[loc1]
        
        loc2 = pos2[:,2] < 35
        pos2 = pos2[loc2]
        
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

        pos1_ = np.copy(pos1)[sample_idx1, :]
        pos2_ = np.copy(pos2)[sample_idx2, :]
        flow_ = np.copy(flow)[sample_idx1, :]


        color1 = np.zeros([self.npoints, 3])
        color2 = np.zeros([self.npoints, 3])
        
        mask = np.ones([self.npoints])

        return pos1_, pos2_, color1, color2, flow_, mask

    def __len__(self):
        return len(self.datapath)
