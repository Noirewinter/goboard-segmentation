import torchvision.transforms.functional as TF
import random
from PIL import Image

class LineAugment:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, is_augment=True):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if random.random() < self.p:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=(0,0,0))
            mask = TF.rotate(mask, angle, fill=(0))

        if random.random() < self.p:
            tx = random.randint(-32, 32)
            ty = random.randint(-32, 32)
            image = TF.affine(image, 0, (tx, ty), 1, 0)
            mask = TF.affine(mask, 0, (tx, ty), 1, 0)

        return image, mask