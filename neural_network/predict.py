from pathlib import Path

import numpy as np
from keras.models import model_from_json
import paths


def predict(vectors):

    p = paths.MODEL / "structure.json"
    model_structure = p.read_text()

    model = model_from_json(model_structure)

    p = paths.MODEL / ".weights.h5"
    model.load_weights(p)

    x = np.array(vectors).astype("float32")
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)[0]

    i = 0
    for num in predictions:
        predictions[i] = '{:.2f}'.format(num)
        i += 1
    predictions = np.array(predictions, dtype=str)

    return predictions
