import os
import numpy as np
import torch.utils.data as t_data

from PIL import Image

IMG_EXTENSIONS = [
    '.npy', '.png'
]

MEAN = np.array([0.6636226, 0.5109665,  0.50087802])
STD = np.array([0.11283001, 0.06067654, 0.06711711])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path, normalization):
    if (path.endswith('.png')):
        res = np.moveaxis(np.array(Image.open(path).convert('RGB')),  [0,1,2], [1, 2, 0]).astype(np.float64) / 255.0
    else:
        res = np.moveaxis(np.load(path), [0,1,2], [1, 2, 0]).astype(np.float64) / 255.0
    if (normalization):
        res -= np.tile(MEAN, (res.shape[2], res.shape[1], 1)).T
        res /= np.tile(STD, (res.shape[2], res.shape[1], 1)).T
    return res

class WeatherDatasetSimple(t_data.Dataset):
    def _make_dataset(self, dir_lr, dir_hr, max_dataset_size=float("inf")):
        images_lr, images_hr = [], []
        assert os.path.isdir(dir_lr), '%s is not a valid directory' % dir_lr
        assert os.path.isdir(dir_hr), '%s is not a valid directory' % dir_hr

        for root, _, fnames in sorted(os.walk(dir_lr)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images_lr.append(path)

        for root, _, fnames in sorted(os.walk(dir_hr)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images_hr.append(path)
                    
              #  assert (images_hr[len(images_hr) - 1][:15] == images_lr[len(images_hr) - 1][:15] and 
               #         images_hr[len(images_hr) - 1][-6:] == images_lr[len(images_hr) - 1][-6:]), " file order mismatch : %s and %s " %images_hr[len(images_hr) - 1] % images_lr[len(images_hr) - 1]
                    
                    
        return images_lr[:min(max_dataset_size, len(images_lr))], images_hr[:min(max_dataset_size, len(images_hr))]
    
    def __init__(self, dir_lr, dir_hr, normalization=False, loader=default_loader):
        imgs = self._make_dataset(dir_lr, dir_hr)
        if len(imgs[0]) == 0:
            raise(RuntimeError("Found 0 images in: " + dir_lr + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if len(imgs[1]) == 0:
            raise(RuntimeError("Found 0 images in: " + dir_hr + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.normalization = normalization
        self.dir_lr, self.dir_hr = dir_lr, dir_hr
        self.imgs_lr, self.imgs_hr = imgs[0], imgs[1]
        self.loader = loader
        assert len(self.imgs_lr) == len(self.imgs_hr), " found an unpaired image "
        
    def __getitem__(self, index):
        path_lr = self.imgs_lr[index]
        img_lr = self.loader(path_lr, self.normalization)
        path_hr = self.imgs_hr[index]
        img_hr = self.loader(path_hr, self.normalization)
        
        return img_lr, img_hr

    def __len__(self):
        return len(self.imgs_lr)
