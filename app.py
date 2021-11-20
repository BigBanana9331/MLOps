from fastapi import FastAPI
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
async def get_prediction(text: str):
    # result =  predictor.predict(text)
    return "oke"