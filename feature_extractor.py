from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16


PERSON_PATH = Path("training_data") / "persons"
NOT_PERSON_PATH = Path("training_data") / "not_persons"

PERSON_PATH_V = Path("validation_data") / "persons"
NOT_PERSON_PATH_V = Path("validation_data") / "not_persons"


def fill_arrays(path, not_path):
    imgs = []
    lbls = []

    for img in path.glob("*.png"):
        img = image.load_img(img, target_size=(256, 256))
        img_ary = image.img_to_array(img)
        imgs.append(img_ary)
        lbls.append(1)

    for img in not_path.glob("*.png"):
        img = image.load_img(img, target_size=(256, 256))
        img_ary = image.img_to_array(img)
        imgs.append(img_ary)
        lbls.append(0)

    return [np.array(imgs), np.array(lbls)]


[x_train, y_train] = fill_arrays(PERSON_PATH, NOT_PERSON_PATH)
[x_test, y_test] = fill_arrays(PERSON_PATH_V, NOT_PERSON_PATH_V)

# normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)
x_test = vgg16.preprocess_input(x_test)

# load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# extract features for each image (all in one pass)
features_xtrain = pretrained_nn.predict(x_train)
features_xtest = pretrained_nn.predict(x_test)

# save the array of extracted features to a file
joblib.dump(features_xtrain, "extracted_features/x_train.dat")
joblib.dump(features_xtest, "extracted_features/x_test.dat")
# save expected values
joblib.dump(y_train, "extracted_features/y_train.dat")
joblib.dump(y_test, "extracted_features/y_test.dat")
