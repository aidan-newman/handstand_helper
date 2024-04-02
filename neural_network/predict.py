import numpy as np
from keras.models import model_from_json
import paths


def load_corrections_model():
    p = paths.MODELS / "corrections/structure.json"
    model_structure = p.read_text()

    model = model_from_json(model_structure)

    p = paths.MODELS / "corrections/.weights.h5"
    model.load_weights(p)
    return model


def load_identify_model():
    p = paths.MODELS / "identify/structure.json"
    model_structure = p.read_text()

    model = model_from_json(model_structure)

    p = paths.MODELS / "identify/.weights.h5"
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

    predictions = model.predict(x, verbose=0)[0]

    return predictions
