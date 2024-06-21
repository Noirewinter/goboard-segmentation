import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from utils import resize_and_pad, show_image
import torchvision.transforms as transforms

np.set_printoptions(threshold=np.inf)

class LineDataset(Dataset):
    def __init__(self, data_dir, imgsz, augment=None):
        self.data_dir = data_dir
        self.augment = augment
        self.items = []
        self.annotation_file = None
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.imgsz = imgsz

        # Check the xml file
        xml_files = [filename for filename in os.listdir(data_dir) if filename.endswith('.xml')]

        if xml_files:
            # Training / Validation
            self.annotation_file = os.path.join(data_dir, xml_files[0])
            self.parse_annotation_file()
        else:
            # Inference
            self.items = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG')]

    def parse_annotation_file(self):
        tree = ET.parse(self.annotation_file)
        root = tree.getroot()

        for image_elem in root.findall('image'):
            image_id = image_elem.get('id')
            image_name = image_elem.get('name')
            image_path = os.path.join(self.data_dir, image_name)
            self.items.append((image_id, image_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.annotation_file:
            # print(self.items)
            # Training / Validation
            image_id, image_path = self.items[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros_like(image[:, :, 0])
            tree = ET.parse(self.annotation_file)
            root = tree.getroot()

            for image_elem in root.findall(f'.//image[@id="{image_id}"]'):
                for polyline in image_elem.findall('./polyline'):
                    points = polyline.get('points')
                    points = points.split(';')
                    points = [point.split(',') for point in points]
                    points = np.array(points, dtype=np.float32)
                    points = points.astype(np.int32)
                    cv2.polylines(mask, [points], False, 255, 2)

            # Resize image
            image, mask, _ = resize_and_pad(image, self.imgsz, mask)

            if self.augment:
                image, mask = self.augment(image, mask)

            # show_image(image, "Image")
            # show_image(mask, "Mask")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            image = transform(np.array(image))
            mask = np.array(mask, dtype=np.float32) / 255.0

            return image, mask
        else:
            # Inference
            image_path = self.items[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image
            image, _, _ = resize_and_pad(image, self.imgsz)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            image = transform(np.array(image))

            return image, image_path