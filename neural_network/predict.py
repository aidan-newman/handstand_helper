from pathlib import Path

import numpy as np
from keras.models import model_from_json
import paths


def load_model():
    p = paths.MODEL / "structure.json"
    model_structure = p.read_text()

    model = model_from_json(model_structure)

    p = paths.MODEL / ".weights.h5"
    model.load_weights(p)
    return model


def predict(model, vectors, left_visible):

    # create list from vectors -- normalize to left-hand side
    vectors_list = []
    if left_visible:
        for vec in vectors:
            vectors_list.append(vec.to_list())
    else:
        for vec in vectors:
            vectors_list.append(vec.flip_x().to_list())

    # keras expects list of inputs, so add empty extra dimension since only one input
    x = np.array(vectors_list).astype("float32")
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)[0]

    i = 0
    for num in predictions:
        predictions[i] = '{:.2f}'.format(num)
        i += 1
    predictions = np.array(predictions, dtype=str)

    return predictions
