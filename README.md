# <img src="./board.png" width="24" /> GoBoard lines segmentation
![Python](https://img.shields.io/badge/-Python-blue?style=flat-square&logo=python&logoColor=white)
![Pytorch](https://img.shields.io/badge/-Pytorch-orange?style=flat-square&logo=Pytorch&logoColor=white)
![TFLite](https://img.shields.io/badge/-TFLite-yellow?style=flat-square&logo=tensorflow&logoColor=white)
![Linux](https://img.shields.io/badge/-Linux-orange?style=flat-square&logo=linux&logoColor=white)
![MacOS](https://img.shields.io/badge/-MacOS-black?style=flat-square&logo=apple&logoColor=white)

## Design goals

This project is an attempt to solve the recognition of Go board information through visual technology in the Go game scenario. Its main application is the segmentation task of lines on the Go board, accurately identifying the position of lines from the board. Subsequently, it can be combined with tasks such as detection to match the relationship between various objects to achieve accurate reconstruction of board information.

## Background and Implementation

When recognizing board information from freely captured Go board images, we typically employ traditional computer vision (CV) algorithms to perform noise reduction, localization, and reconstruction. However, we often encounter challenges such as tilted shooting angles, blurred target objects, and distorted target regions due to curling. For these adverse conditions that are almost always encountered in free-shooting scenarios, relying solely on CV approaches for processing cannot achieve ideal performance requirements.

By combining deep-learning-based detection tasks with CV methods, the robustness issues brought about by object interference and tilted angles can be improved. However, when facing distortion, especially excessive curling deformation in the context of capturing images of book pages, the results are still unsatisfactory. Therefore, an attempt is made to implement segmentation tasks：

- Leverage the high positional accuracy of segmentation tasks
- Combine with CV algorithms to accurately locate the intersection points of the board lines
- Reconstruct the board based on the relationship between the intersection points and the positions of the pieces (not a segmentation task)

The segmentation algorithm is implemented using [U-Net](https://arxiv.org/abs/1505.04597) ![Page](https://img.shields.io/badge/Page-arXiv:1505.04597-B3181B?style=flat-square&logo=arXiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597) to achieve better performance.

With a small amount of sample training, the algorithm has achieved excellent results. To attain even higher performance, more diverse data is needed for training. At the same time, this algorithm is an attempt to solve similar problems and can be extended to a range of applications such as chessboard detection, table detection, and more.

## Environment
**Ubuntu 20.04**  
**Python 3.8.x**  

## Usage
### Installation 
Quick install dependencies: 
```pip install -r requirements.txt```

### Training
To train a segmentor with pre-trained models, run:
```bash
./train.py [-h] [--train_data TRAIN_DATA] [--val_data VAL_DATA] [--save_path SAVE_PATH] [--pretrained_path PRETRAINED_PATH]
                [--num_epochs NUM_EPOCHS] [--lr LR] [--batch_size BATCH_SIZE] [--imgsz IMGSZ]
```

### Inference  
To inference based on PyTorch model, run:
```bash
./infer.py [-h] --model_path MODEL_PATH [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--imgsz IMGSZ]
```

To inference based on TFLite model, run:
```bash
./tflite_infer.py [-h] --model_path MODEL_PATH [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--imgsz IMGSZ]
```

**Here are some examples of visualization:** 
<p align="left">
    <img src="./images/outputs/intersections_IMG_0661.JPG" width="300" height="260" />
    <img src="./images/outputs/intersections_IMG_0789.JPG" width="300" height="260" />
    <img src="./images/outputs/intersections_IMG_0676.JPG" width="300" height="260" />
    <img src="./images/outputs/intersections_IMG_0684.JPG" width="300" height="260" />
</p>

### Export TFLite
```bash
./export.py [-h] --model_path MODEL_PATH --output_path OUTPUT_PATH [--imgsz IMGSZ]
```