#!python

import torch
from pathlib import Path
from unet import UNet
from utils import select_device
import shutil
import os
import argparse

class Exporter:
    def __init__(self, model_path, output_path, imgsz):
        self.model_path = model_path
        self.output_path = output_path
        self.imgsz = imgsz
        self.model = None
        self.im = None
        self.device = select_device()

    def load_model(self):
        # Input
        self.im = torch.zeros(1, 3, *self.imgsz).to(self.device)
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def export_onnx(self, prefix="ONNX:"):
        print(f"{prefix} starting ONNX export with torch.onnx {torch.__version__}...")
        
        f = os.path.join(self.output_path, "saved_model.onnx")
        torch.onnx.export(
            self.model,
            self.im,
            f,
            do_constant_folding=True,
            verbose=False,
            input_names=["images"],
            output_names=["output0"],
        )

        import onnx
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        try:
            import onnxsim
            print(f"{prefix} simplifying with onnxsim {onnxsim.__version__}...")
            # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
        except Exception as e:
            print(f"{prefix} simplifier failure: {e}")

        onnx.save(model_onnx, f)
        print(f"{prefix} export ONNX success: {str(f)}")
        return f

    def export_tflite(self, prefix="TensorFlow Lite:"):
        f = Path(self.output_path)
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder
        f.mkdir()

        # Export to ONNX
        f_onnx = self.export_onnx()

        # Export to TF
        import onnx2tf
        verbosity = "error"
        print(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
        onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity=verbosity,
            output_integer_quantized_tflite=False,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        )

        print(f"{prefix} export TFLite success: {str(f)}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a model')
    parser.add_argument('--model_path', help='model path', required=True)
    parser.add_argument('--output_path', help='output path', required=True)
    parser.add_argument('--imgsz', default=512, help='image size', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model_path = args.model_path
    output_path = args.output_path
    imgsz = args.imgsz
    exporter = Exporter(model_path, output_path, (imgsz, imgsz))
    exporter.load_model()
    exporter.export_tflite()

if __name__ == '__main__':
    main()