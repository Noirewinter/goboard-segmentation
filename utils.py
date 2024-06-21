import cv2
import numpy as np
from PIL import Image
import torch

def show_image(image, title="Image", wait_key=0):
    """
    Show the image
    """
    im = image
    if isinstance(im, Image.Image):
        # If PIL image, convert to openCV 
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    
    cv2.imshow(title, im)
    cv2.waitKey(wait_key)
    cv2.destroyWindow(title)

def resize_and_pad(image, target_size, mask=None):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full(target_size + (3,), 255, dtype=np.uint8)  # Padding white color

    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    if mask is not None:
        # Resize mask
        resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        padded_mask = np.zeros(target_size, dtype=np.uint8)  # Padding black color
        padded_mask[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_mask
        return padded_image, padded_mask, (x_offset, y_offset, new_width, new_height)
    else:
        return padded_image, None, (x_offset, y_offset, new_width, new_height)
    
def select_device():
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device