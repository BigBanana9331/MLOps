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
    outputs = {
        "loss": 0.3,
        "acc": 0.3,
        "precision_macro": 0.3,
        "recall_macro": 0.3,
        "precision_micro": 0.3,
        "recall_micro": 0.3,
        "f1": 0.3,
    }
    
    # Logging metrics
    logger.info(dict(loss=outputs["loss"], details={"type": "prediction"}))
    logger.info(dict(acc=outputs["acc"], details={"type": "prediction"}))
    logger.info(dict(precision_macro=outputs["precision_macro"], details={"type": "prediction"}))
    logger.info(dict(recall_macro=outputs["recall_macro"], details={"type": "prediction"}))
    logger.info(dict(precision_micro=outputs["precision_micro"], details={"type": "prediction"}))
    logger.info(dict(recall_micro=outputs["recall_micro"], details={"type": "prediction"}))
    logger.info(dict(f1=outputs["f1"], details={"type": "prediction"}))

    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    # result =  predictor.predict(text)
    return "oke"