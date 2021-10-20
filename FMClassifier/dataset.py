from tflite_model_maker.image_classifier import DataLoader

class Data:
    def __init__(path):
        self.data = DataLoader.from_folder(path)
        self.train_data, rest_data = data.split(0.8)
        self.validation_data, self.test_data = rest_data.split(0.5)
    
    def summary_dataset():
        plt.figure(figsize=(10,10))
        for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image.numpy(), cmap=plt.cm.gray)
            plt.xlabel(data.index_to_label[label.numpy()])
        plt.show()
