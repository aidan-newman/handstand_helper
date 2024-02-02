from pathlib import Path
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.applications import vgg16


f = Path("model/model_structure.json")
model_structure = f.read_text()

model = model_from_json(model_structure)
model.load_weights("model/model_weights.h5")

imgs = []

img = load_img("test.jpg", target_size=(256, 256))
imgs.append(img_to_array(img))

imgs = np.array(imgs)
imgs = vgg16.preprocess_input(imgs)

feature_extractor = vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3),
)
features = feature_extractor.predict(imgs)

results = model.predict(features)

single_result = results[0][0]

print(results)
