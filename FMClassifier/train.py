import os
import datetime

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

import wandb

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import ImageClassifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
import hydra

wandb.init(project='FMClassifier', entity='nguyenhuy39',config={
    "model_spec":"resnet_50",
    "epochs":25
})

@hydra.main(config_path="../conf", config_name="config")

def main(cfg):
    # prepare data
    print(cfg.dataset.path)
    data = DataLoader.from_folder(cfg.dataset.path)
    train_data, rest_data = data.split(cfg.dataset.train_ratio)
    validation_data, test_data = rest_data.split(cfg.dataset.test_ratio)
    # create model
    # model = ImageClassifier(model_spec=cfg.model.name).create_model()
    # model.train(train_data = train_data,validation_data=validation_data)
    # with tf.compat.v1.Session() as sess:
    model = image_classifier.create(train_data, model_spec=model_spec.get(cfg.model.name), validation_data=validation_data, epochs = cfg.model.epochs)
    # model.summary()
    loss, accuracy = model.evaluate(test_data)
        # wandb.tensorflow.log(tf.summary.merge_all())
    # evaluate model
    # print(loss)
    # print(accuracy)
    wandb.log({"loss": loss})
    wandb.log({"accuracy": accuracy})
    # export model
    config = QuantizationConfig.for_float16()
    model.export(export_dir='D:\MLOps\models',tflite_filename='{model}_{day}.tflite'.format(day = datetime.date.today(),model=cfg.model.name), quantization_config=config)
    accuracy_tflite = model.evaluate_tflite('D:\MLOps\models\{model}_{day}.tflite'.format(day = datetime.date.today(),model=cfg.model.name), test_data)
    wandb.log({"accuracy_tflite": accuracy_tflite})

if __name__ == '__main__':
    main()
