import dataset
import model

dataset = Data('path')

model = FMClassifierModel.train(dataset.train_data,model,epoch)
model.summary()
model.evaluate()
model.predict()S
