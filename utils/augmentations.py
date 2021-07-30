
import math
import random 
import numpy as np 
import tensorflow.image as tfi
import tensorflow_addons.image as tfi_a
import tensorflow.keras.layers.experimental.preprocessing as preproc


AUGMENTATIONS = {
    "cutout": Cutout,
    "gaussian_blur": GaussianBlur,
    "color_jitter": ColorJitter,
    "random_gray": RandomGrayscale,
    "random_resized_crop": RandomResizedCrop,
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_vertical_flip": RandomVerticalFlip,
    "center_crop": CenterCrop,
    "normalize": Normalize,
    "resize": Resize
}


class GaussianBlur:
    def __init__(self, kernel_size=(3, 3), sigma=[0.1, 2.0], p=0.0):
        self.p = p
        self.sigma = sigma 
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            out = tfi_a.gaussian_filter2d(img, filter_shape=self.kernel_size, sigma=sigma, padding="CONSTANT")
        else:
            out = img 
        return out

class Cutout:
    def __init__(self, n_cuts=0, maxlen=1, p=0.0):
        self.p = p
        self.n_cuts = n_cuts
        self.maxlen = maxlen

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            h, w = img.shape[0], img.shape[1]
            cutlen = random.randint(1, self.maxlen)
            mask = np.ones((h, w), np.float32)
            for _ in range(self.n_cuts):
                x, y = random.randint(0, w), random.randint(0, h)
                x1, x2 = np.clip(x - cutlen // 2, 0, w), np.clip(x + cutlen // 2, 0, w)
                y1, y2 = np.clip(y - cutlen // 2, 0, h), np.clip(y + cutlen // 2, 0, h)
                mask[y1:y2, x1:x2, :] = 0
            out = img * mask
        else:
            out = img
        return out

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.0):
        assert brightness >= 0, "Brightness should be >= 0"
        assert contrast >= 0, "Contrast should be >= 0"
        assert saturation >= 0, "Saturation should be >= 0"
        self.co_min, self.co_max = max(0, 1-contrast), 1+contrast
        self.sa_min, self.sa_max = max(0, 1-saturation), 1+saturation
        self.hu_delta = min(0.5, hue)
        self.br_delta = brightness
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = tfi.random_brightness(img, self.br_delta)
            img = tfi.random_saturation(img, self.sa_min, self.sa_max)
            img = tfi.random_contrast(img, self.co_min, self.co_max)
            out = tfi.random_hue(img, self.hu_delta)
        else:
            out = img 
        return out

class RandomGrayscale:
    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, img):
        channels = img.shape[-1]
        if random.uniform(0, 1) < self.p:
            img = img.mean(axis=-1, keepdims=True)
            out = np.repeat(img, repeats=channels, axis=-1)
        else:
            out = img 
        return out

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.33), p=0.0):
        self.scale_min, self.scale_max = scale
        self.ratio_min, self.ratio_max = ratio
        self.h, self.w = size
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            h, w, c = img.shape
            ratio = random.uniform(self.ratio_min, self.ratio_max)
            scale = random.uniform(self.scale_min, self.scale_max) * (h * w)
            cut_h, cut_w = math.sqrt(scale/ratio), math.sqrt(scale*ratio)
            cut_x, cut_y = random.randint(0, w), random.randint(0, h)
            x1, x2 = np.clip(cut_x - cut_w // 2, 0, w), np.clip(cut_x + cut_w // 2, 0, w)
            y1, y2 = np.clip(cut_y - cut_h // 2, 0, h), np.clip(cut_y + cut_h // 2, 0, h)
            out = img[y1:y2, x1:x2, :]
            out = tfi.resize(out, size=[self.h, self.w, c], method="bilinear", antialias=False)
        else:
            out = img 
        return out

class RandomHorizontalFlip:
    def __init__(self, p=0.0):
        self.p = p
    
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            out = tfi.random_flip_left_right(img)
        else:
            out = img 
        return out

class RandomVerticalFlip:
    def __init__(self, p=0.0):
        self.p = p
    
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            out = tfi.random_flip_up_down(img)
        else:
            out = img 
        return out

class CenterCrop:
    def __init__(self, size):
        self.size = size 

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        ctr_x, ctr_y = w // 2, h // 2
        x1, x2 = np.clip(ctr_x - self.size // 2, 0, w), np.clip(ctr_x + self.size // 2, 0, w)
        y1, y2 = np.clip(ctr_y - self.size // 2, 0, h), np.clip(ctr_y + self.size // 2, 0, h)
        return img[y1:y2, x1:x2, :]

class Resize:
    def __init__(self, size, method="bilinear", antialias=False):
        self.h, self.w = size 
        self.method = method
        self.antialias = antialias 

    def __call__(self, img):
        return tfi.resize(img, size=[self.h, self.w, img.shape[-1]], method=self.method, antialias=self.antialias)

class Normalize:
    def __init__(self, mean, std):
        var = [a**2 for a in std]
        self.norm_layer = preproc.Normalization(axis=-1, mean=mean, variance=var)

    def __call__(self, img):
        return self.norm_layer(img)

class Compose:
    def __init__(self, transform_list):
        self.transforms = transform_list

    def __call__(self, img):
        for func in transform_list:
            img = func(img)
        return img


def get_transform(config):
    transform = []
    for key, value in config.items():
        if value is not None:
            tr = AUGMENTATIONS[key](**value)
        else:
            tr = AUGMENTATIONS[key]()
        transform.append(tr)
    return Compose(transform)