from fastapi import FastAPI
from FMClassifier.run_onnx import predict
# from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")
import watchtower, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler())

# predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    # Logging metrics
    logger.info(dict(prediction=dict(label=True, prediction=0.9)))
    logger.info(dict(prediction=dict(label=False, prediction=0.2)))

    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction():
    label, acc = predict(image='D:\MLOps\FMClassifier\test\test1.jpg', label_file='D:\MLOps\FMClassifier\labels.names', model_file='D:\MLOps\models\efficientnet_lite0_2021-10-23.onnx')
    logger.info(dict(prediction=dict(label=label, prediction=acc)))
    return label, acc