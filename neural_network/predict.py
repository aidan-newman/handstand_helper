from pathlib import Path

import numpy as np
from keras.models import model_from_json


def predict(vectors):

    p = Path("neural_network/model/structure.json").absolute()
    model_structure = p.read_text()

    model = model_from_json(model_structure)

    model.load_weights("neural_network/model/.weights.h5")

    x = np.array(vectors).astype("float32")
    x = np.expand_dims(vectors, axis=0)

    predictions = model.predict(x)

    print(predictions)
