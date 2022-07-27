
import torch
import random
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Cutout:
    def __init__(self, n_cuts=0, max_len=1):
        self.n_cuts = n_cuts
        self.max_len = max_len

    def __call__(self, img):
        h, w = img.shape[1:3]
        cut_len = random.randint(1, self.max_len)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_cuts):
            x, y = random.randint(0, w), random.randint(0, h)
            x1 = np.clip(x - cut_len // 2, 0, w)
            x2 = np.clip(x + cut_len // 2, 0, w)
            y1 = np.clip(y - cut_len // 2, 0, h)
            y2 = np.clip(y + cut_len // 2, 0, h)
            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class RandomAugment:
    def __init__(self, n_aug=4):
        self.n_aug = n_aug
        self.aug_list = [
            ("identity", 1, 1),
            ("autocontrast", 1, 1),
            ("equalize", 1, 1),
            ("rotate", -30, 30),
            ("solarize", 1, 1),
            ("color", 1, 1),
            ("contrast", 1, 1),
            ("brightness", 1, 1),
            ("sharpness", 1, 1),
            ("shear_x", -0.1, 0.1),
            ("shear_y", -0.1, 0.1),
            ("translate_x", -0.1, 0.1),
            ("translate_y", -0.1, 0.1),
            ("posterize", 1, 1),
        ]

    def __call__(self, img):
        aug_choices = random.choices(self.aug_list, k=self.n_aug)
        for aug, min_value, max_value in aug_choices:
            v = random.uniform(min_value, max_value)
            if aug == "identity":
                pass
            elif aug == "autocontrast":
                img = ImageOps.autocontrast(img)
            elif aug == "equalize":
                img = ImageOps.equalize(img)
            elif aug == "rotate":
                if random.random() > 0.5:
                    v = -v
                img = img.rotate(v)
            elif aug == "solarize":
                img = ImageOps.solarize(img, v)
            elif aug == "color":
                img = ImageEnhance.Color(img).enhance(v)
            elif aug == "contrast":
                img = ImageEnhance.Contrast(img).enhance(v)
            elif aug == "brightness":
                img = ImageEnhance.Brightness(img).enhance(v)
            elif aug == "sharpness":
                img = ImageEnhance.Sharpness(img).enhance(v)
            elif aug == "shear_x":
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
            elif aug == "shear_y":
                if random.random() > 0.5:
                    v = -v
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))
            elif aug == "translate_x":
                if random.random() > 0.5:
                    v = -v
                v = v * img.size[0]
                img = img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
            elif aug == "translate_y":
                if random.random() > 0.5:
                    v = -v
                v = v * img.size[1]
                img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
            elif aug == "posterize":
                img = ImageOps.posterize(img, int(v))
            else:
                raise NotImplementedError(f"{aug} not implemented")
        return img
    
    
class MultiCrop:
    def __init__(self, config):
        self.num_local = config.get("num_local_views", 6)
        self.num_global = config.get("num_global_views", 2)
        self.scale = config.get("scale_threshold", 0.3)
        
        self.global_crop = transforms.RandomResizedCrop(
            config["global_size"], scale=(self.scale, 1.0), 
            interpolation=InterpolationMode.BICUBIC
        )
        self.local_crop = transforms.RandomResizedCrop(
            config["local_size"], scale=(0.08, self.scale), 
            interpolation=InterpolationMode.BICUBIC
        )
        self.transforms = build_transform(config["train_transforms"])

    def __call__(self, img):
        global_1, global_2, local_1, local_2 = [], [], [], []
        aug_1, aug_2 = self.transforms(img), self.transforms(img)

        for _ in range(self.num_global):
            crop_1, crop_2 = self.global_crop(aug_1), self.global_crop(aug_2)
            global_1.append(crop_1), global_2.append(crop_2)
        
        for _ in range(self.num_local):
            crop_1, crop_2 = self.local_crop(aug_1), self.local_crop(aug_2)
            local_1.append(crop_1), local_2.append(crop_2)
        
        return {
            "global_1": torch.stack(global_1, 0), 
            "global_2": torch.stack(global_2, 0),
            "local_1": torch.stack(local_1, 0),
            "local_2": torch.stack(local_2, 0)
        }
        
        
class ToNormalizedNumpy:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean).reshape(1, 1, 3)
        self.std = np.asarray(std).reshape(1, 1, 3)
        
    def __call__(self, img):
        img = np.asarray(img, dtype=np.float32)
        img = (img / 255.0 - self.mean) / self.std
        return img


# Transformation helper
TRANSFORM_HELPER = {
    "gaussian_blur": GaussianBlur,
    "color_jitter": transforms.ColorJitter,
    "random_gray": transforms.RandomGrayscale,
    "random_crop": transforms.RandomCrop,
    "random_resized_crop": transforms.RandomResizedCrop,
    "center_crop": transforms.CenterCrop,
    "resize": transforms.Resize,
    "random_flip": transforms.RandomHorizontalFlip,
    "to_tensor": transforms.ToTensor,
    "normalize": transforms.Normalize,
    "to_normalized_numpy": ToNormalizedNumpy,
    "rand_aug": RandomAugment,
    "cutout": Cutout,
}

def build_transform(config):
    """
    Generates a torchvision.transforms.Compose pipeline
    based on given configurations.
    """
    transform = []
    # Obtain transforms from config in sequence
    for key, value in config.items():
        if value is not None:
            p = value.pop("apply_prob", None)
            tr = TRANSFORM_HELPER[key](**value)
            if p is not None:
                tr = transforms.RandomApply([tr], p=p)
        else:
            tr = TRANSFORM_HELPER[key]()
        transform.append(tr)
    return transforms.Compose(transform)