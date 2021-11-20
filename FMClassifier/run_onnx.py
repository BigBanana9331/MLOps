import onnxruntime as ort
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def main(image, label_file, model_file):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pre-process the image like mobilenet and resize it to 300x300
    img = pre_process_edgetpu(img, (224, 224, 3))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    img_batch = np.expand_dims(img, axis=0)
    labels = load_labels(label_file)

    sess = ort.InferenceSession(model_file)

    results = sess.run(["Identity"], {"input_1": img_batch})
    arr = results[0][0]
    if (arr[0] > arr[1]):
        print(labels[0])
    else: 
        print(labels[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()
    
    main(args.image, args.label_file, args.model_file)
    




























































# import os
# import sys
# import numpy as np
# import re
# import abc
# import subprocess
# import json
# import argparse
# import time
# from PIL import Image

# import onnx
# import onnxruntime
# from onnx import helper, TensorProto, numpy_helper
# from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType


# class ResNet50DataReader(CalibrationDataReader):
#     def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
#         self.image_folder = calibration_image_folder
#         self.augmented_model_path = augmented_model_path
#         self.preprocess_flag = True
#         self.enum_data_dicts = []
#         self.datasize = 0

#     def get_next(self):
#         if self.preprocess_flag:
#             self.preprocess_flag = False
#             session = onnxruntime.InferenceSession(self.augmented_model_path, None)
#             (_, _, height, width) = session.get_inputs()[0].shape
#             nhwc_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
#             input_name = session.get_inputs()[0].name
#             self.datasize = len(nhwc_data_list)
#             self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
#         return next(self.enum_data_dicts, None)


# def preprocess_func(images_folder, height, width, size_limit=0):
#     '''
#     Loads a batch of images and preprocess them
#     parameter images_folder: path to folder storing images
#     parameter height: image height in pixels
#     parameter width: image width in pixels
#     parameter size_limit: number of images to load. Default is 0 which means all images are picked.
#     return: list of matrices characterizing multiple images
#     '''
#     image_names = os.listdir(images_folder)
#     if size_limit > 0 and len(image_names) >= size_limit:
#         batch_filenames = [image_names[i] for i in range(size_limit)]
#     else:
#         batch_filenames = image_names
#     unconcatenated_batch_data = []

#     for image_name in batch_filenames:
#         image_filepath = images_folder + '/' + image_name
#         pillow_img = Image.new("RGB", (width, height))
#         pillow_img.paste(Image.open(image_filepath).resize((width, height)))
#         input_data = np.float32(pillow_img) - \
#         np.array([123.68, 116.78, 103.94], dtype=np.float32)
#         nhwc_data = np.expand_dims(input_data, axis=0)
#         nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
#         unconcatenated_batch_data.append(nchw_data)
#     batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
#     return batch_data


# def benchmark(model_path):
#     session = onnxruntime.InferenceSession(model_path)
#     input_name = session.get_inputs()[0].name

#     total = 0.0
#     runs = 10
#     input_data = np.zeros((1, 3, 224, 224), np.float32)
#     # Warming up
#     _ = session.run([], {input_name: input_data})
#     for i in range(runs):
#         start = time.perf_counter()
#         _ = session.run([], {input_name: input_data})
#         end = (time.perf_counter() - start) * 1000
#         total += end
#         print(f"{end:.2f}ms")
#     total /= runs
#     print(f"Avg: {total:.2f}ms")


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_model", required=True, help="input model")
#     parser.add_argument("--output_model", required=True, help="output model")
#     parser.add_argument("--calibrate_dataset", default="./test_images", help="calibration data set")
#     parser.add_argument("--quant_format",
#                         default=QuantFormat.QOperator,
#                         type=QuantFormat.from_string,
#                         choices=list(QuantFormat))
#     parser.add_argument("--per_channel", default=False, type=bool)
#     args = parser.parse_args()
#     return args


# def main():
#     args = get_args()
#     input_model_path = args.input_model
#     output_model_path = args.output_model
#     calibration_dataset_path = args.calibrate_dataset
#     dr = ResNet50DataReader(calibration_dataset_path)
#     quantize_static(input_model_path,
#                     output_model_path,
#                     dr,
#                     quant_format=args.quant_format,
#                     per_channel=args.per_channel,
#                     weight_type=QuantType.QInt8)
#     print('Calibrated and quantized model saved.')

#     print('benchmarking fp32 model...')
#     benchmark(input_model_path)

#     print('benchmarking int8 model...')
#     benchmark(output_model_path)


# if __name__ == '__main__':
#     main()