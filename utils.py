import os
import numpy as np
import torch.utils.data as t_data
import matplotlib.pyplot as plt


from PIL import Image

IMG_EXTENSIONS = [
    #'.png',
    '.npy'
]

reverse_map = {}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def convert(rgb):
    if tuple(rgb) in reverse_map.keys():
        return np.array(reverse_map[tuple(rgb)])
    else:
        return np.array(0)

def default_loader(path):
    img = np.load(path)
    res = img.astype(np.float32)
    return res#[0][0][0][0]

class WeatherDatasetSimple(t_data.Dataset):
    def _make_dataset(self, dir_lr, dir_hr, max_dataset_size=float("inf")):
        images_lr, images_hr = [], []
        assert os.path.isdir(dir_lr), '%s is not a valid directory' % dir_lr
        assert os.path.isdir(dir_hr), '%s is not a valid directory' % dir_hr

        for root, _, fnames in sorted(os.walk(dir_lr)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    region_num = int(path.split('.')[-2].split('_')[-1])
                    images_lr.append((region_num, path)) # grouping images for same region. Default format groups them by timestamp first

        for root, _, fnames in sorted(os.walk(dir_hr)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    region_num = int(path.split('.')[-2].split('_')[-1])
                    images_hr.append((region_num, path))

                    assert (images_hr[len(images_hr) - 1][1][-10:] == images_lr[len(images_hr) - 1][1][-10:]), " file order mismatch : %s and %s " % (images_hr[len(images_hr) - 1][1], images_lr[len(images_hr) - 1][1])
                    
        images_lr = sorted(images_lr) 
        images_hr = sorted(images_hr)
        images_lr = [tulp[1] for tulp in  images_lr]
        images_hr = [tulp[1] for tulp in  images_hr]
        return images_lr[:min(max_dataset_size, len(images_lr))], images_hr[:min(max_dataset_size, len(images_hr))]
    
    def __init__(self, dir_lr, dir_hr, max_dataset_size=float("inf"), loader=default_loader):       
        imgs = self._make_dataset(dir_lr, dir_hr, max_dataset_size=max_dataset_size)
        if len(imgs[0]) == 0:
            raise(RuntimeError("Found 0 images in: " + dir_lr + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if len(imgs[1]) == 0:
            raise(RuntimeError("Found 0 images in: " + dir_hr + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.dir_lr, self.dir_hr = dir_lr, dir_hr
        self.imgs_lr, self.imgs_hr = imgs[0], imgs[1]
        self.loader = loader
        assert len(self.imgs_lr) == len(self.imgs_hr), " found an unpaired image "
        
    def __getitem__(self, index):
        path_lr = self.imgs_lr[index]
        
        img_lr = self.loader(path_lr)
        path_hr = self.imgs_hr[index]
        img_hr = self.loader(path_hr)
        return img_lr, img_hr

    def __len__(self):
        return len(self.imgs_lr)
