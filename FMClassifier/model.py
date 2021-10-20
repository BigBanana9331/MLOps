import os
import datetime

import numpy as np

import tensorflow as tf
assert tf.__version__.startswithS('2')
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig

import matplotlib.pyplot as plt

class FMClassifierModel:

  def __init__(train_data,model,validation_data,epoch):
    self.train_data = train_data
    self.epochs = epoch
    self.model = model
    self.validation_data = validation_data

  def train():
    self.model = image_classifier.create(train_data, model_spec=model_spec.get(model), validation_data=validation_data,epochs=epoch)

  def evaluate(data):
    return loss, accuracy = model.evaluate(data)

  def summary():
    model.summary()

  def get_label_color(val1, val2):
    if val1 == val2:
      return 'black'
    else:
      return 'red'
  def predict():
    plt.figure(figsize=(20, 20))
    predicts = model.predict_top_k(test_data)
    for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
      ax = plt.subplot(10, 10, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(image.numpy(), cmap=plt.cm.gray)

      predict_label = predicts[i][0][0]
      color = get_label_color(predict_label,
                              test_data.index_to_label[label.numpy()])
      ax.xaxis.label.set_color(color)
      plt.xlabel('Predicted: %s' % predict_label)
    plt.show()
  def export(path):
    config = QuantizationConfig.for_float16()
    model.export(export_dir=path,tflite_filename='model_fp16{date}.tflite'.format(date=datetime.date.today()), quantization_config=config)