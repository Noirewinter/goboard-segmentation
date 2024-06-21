#!python

import os
import cv2
import numpy as np
import tensorflow as tf
from dataset import LineDataset
from utils import show_image, resize_and_pad
import platform
import argparse

def postprocess(pred_mask, orig_image, output_dir, output_filename):
    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    mask = (mask > 0).astype(np.uint8)

    # Empty mask for Drawing lines
    line_mask = np.zeros_like(mask)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=16, minLineLength=4, maxLineGap=8)

    # Store the endpoint coordinates
    line_endpoints = []

    # Draw lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            line_endpoints.append(((x1, y1), (x2, y2)))

    print("line_endpoints: ", len(line_endpoints))

    # Calculate intersections between lines
    intersection_points = []
    for i in range(len(line_endpoints)):
        for j in range(i+1, len(line_endpoints)):
            line1 = line_endpoints[i]
            line2 = line_endpoints[j]
            intersection_point = calculate_intersection(line1, line2)
            if intersection_point is not None:
                intersection_points.append(intersection_point)

    print("intersection_points: ", len(intersection_points))

    # Draw intersections onto mask
    for point in intersection_points:
        cv2.circle(line_mask, point, 4, 255, -1)

    # Find all regions with dense intersections
    if len(intersection_points) > 0:
        density_threshold = 1
        radius = 20

        dense_centers = []
        for center_point in intersection_points:
            overlap_area = 0
            for point in intersection_points:
                # Distance between two points
                distance = np.sqrt((point[0] - center_point[0])**2 + (point[1] - center_point[1])**2)
                # print("distance <= radius: ", distance)
                if distance <= radius:
                    overlap_area += 1

            # print("overlap_area >= density_threshold: ", overlap_area)
            if overlap_area >= density_threshold:
                dense_centers.append(center_point)

        print("dense_centers: ", len(dense_centers))

        # Non-maximum suppression based on the density of points
        nms_centers = []
        while len(dense_centers) > 0:
            # Select the first point
            current_center = dense_centers[0]
            nms_centers.append(current_center)

            # Remove other points within the radius range
            dense_centers = [p for p in dense_centers if np.sqrt((p[0] - current_center[0])**2 + (p[1] - current_center[1])**2) > radius]

        print("nms_centers_count: ", len(nms_centers))
        print("nms_centers: ", nms_centers)

        # Draw the filtered points onto original image
        for center in nms_centers:
            cv2.circle(orig_image, center, 4, (0, 255, 0), -1)
        

    # Merge image with mask
    line_image = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
    result_image = cv2.addWeighted(orig_image, 0.7, line_image, 0.3, 0)

    # Save image
    output_path = os.path.join(output_dir, f"intersections_{output_filename}")
    cv2.imwrite(output_path, result_image)

def calculate_intersection(line1, line2):
    # Extract the endpoint
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if denom == 0:
        return None  # Parallel, no intersection

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # Check if intersections is within the range of the line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return int(x), int(y)
    else:
        return None  # Outside the line segments

def tflite_inference(model_path, data_dir, output_dir, imgsz):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Create datasets
    infer_dataset = LineDataset(data_dir, imgsz)
    poi_conf = 0.8

    # Start inference
    for image, image_path in infer_dataset:
        image = image.unsqueeze(0).numpy().astype(np.float32)
        image = np.transpose(image, (0, 2, 3, 1))
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print("output: ", output.shape)

        pred_mask = output.squeeze()

        print("Prediction min:", output.min())
        print("Prediction max:", output.max())
        
        # Convert to 0/1 code
        pred_mask = np.where(pred_mask > poi_conf, pred_mask, 0.0)

        # Load original image
        orig_image = cv2.imread(image_path)
        original_size = orig_image.shape[:2]

        print("image_path: ", image_path)

        # Unpadding around mask
        _, _, (x_offset, y_offset, new_width, new_height) = resize_and_pad(orig_image, imgsz)
        unpadded_mask = pred_mask[y_offset:y_offset+new_height, x_offset:x_offset+new_width]

        # Restore mask to original size
        restored_mask = cv2.resize(unpadded_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Draw the predicted masks onto original image
        mask_image = np.zeros_like(orig_image)
        mask_image[restored_mask > 0] = (255, 0, 0)

        overlay = cv2.addWeighted(orig_image, 0.7, mask_image, 0.3, 0)

        output_filename = os.path.basename(image_path)
        # Save mask image
        mask_output_path = os.path.join(output_dir, f"mask_{output_filename}")
        cv2.imwrite(mask_output_path, restored_mask * 255)

        # Save result image
        output_path = os.path.join(output_dir, f"result_{output_filename}")
        cv2.imwrite(output_path, overlay)

        postprocess(restored_mask, orig_image, output_dir, output_filename)

def parse_args():
    parser = argparse.ArgumentParser(
        description='infer with a TFLite model')
    parser.add_argument('--model_path', help='model path', required=True)
    parser.add_argument('--data_dir', default='./images/inputs', help='input images path')
    parser.add_argument('--output_dir', default='./images/outputs', help='output images path')
    parser.add_argument('--imgsz', default=512, help='image size', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model_path = args.model_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    imgsz = args.imgsz
    tflite_inference(model_path, data_dir, output_dir, (imgsz, imgsz))

if __name__ == '__main__':
    main()