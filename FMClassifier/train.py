import os
import datetime

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import ImageClassifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
import hydra

@hydra.main(config_path="../conf", config_name="config")

def main(cfg):
    # prepare data
    print(cfg.dataset.path)
    data = DataLoader.from_folder(cfg.dataset.path)
    train_data, rest_data = data.split(cfg.dataset.train_ratio)
    validation_data, test_data = rest_data.split(cfg.dataset.test_ratio)
    # create model
    model = ImageClassifier(model_spec=cfg.model.name).create_model()
    model.train(train_data = train_data,validation_data=validation_data)

    # evaluate model
    model.summary()
    loss, accuracy = model.evaluate(test_data)


    # export model
    config = QuantizationConfig.for_float16()
    model.export(export_dir='../models',tflite_filename='model_fp16_{day}.tflite'.format(day = datetime.date.today()), quantization_config=config)
if __name__ == '__main__':
    main()
