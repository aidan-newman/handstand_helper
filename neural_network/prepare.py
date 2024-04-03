import csv
import json
import shutil

import numpy as np

import paths
from tasks import analysis
from tasks import image
from tasks import file
from neural_network.predict import load_corrections_model

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input
from keras.callbacks import ModelCheckpoint


IMAGE_SAVE_HEIGHT = 800
INPUT_SHAPE = (5, 3)


def prepare_labels():

    # open csv file
    try:
        csv_file = open(paths.TRAINING_DATA / "corrections/labels.csv", "r")
    except OSError:
        raise OSError("Couldn't open csv_file in read mode.")

    lines = csv_file.readlines()
    index = int(lines[-1].split('.')[0]) + 1

    try:
        csv_file = open(paths.TRAINING_DATA / "corrections/labels.csv", 'a', newline='')
    except OSError:
        raise OSError("Couldn't open csv_file in append mode.")

    writer = csv.writer(csv_file, delimiter=',')

    cancel = False
    rows = []
    img_count = 0
    for img in (paths.TRAINING_DATA / "corrections/potential_images").iterdir():
        if cancel:
            break
        if img.name.endswith(".jpg") or img.name.endswith(".png"):
            img_count += 1
            image.compress_file(img, height=IMAGE_SAVE_HEIGHT)

            row = [str(index) + img.suffix]
            for c in analysis.CORRECTIONS:
                invalid = True
                while invalid:
                    print(c + "?:")
                    entry = input()

                    if str(entry).lower().strip() == "x":
                        file.safe_move(paths.TRAINING_DATA / "corrections/bad_images", img)
                        break
                    elif str(entry).lower() == "stop":
                        cancel = True
                        break
                    elif entry.isdigit():
                        entry = int(entry)
                        if entry == 0 or entry == 1:
                            invalid = False
                            row.append(entry)
                else:
                    continue
                break

            if len(row) == len(analysis.CORRECTIONS)+1:
                rows.append(row)
                shutil.move(img, paths.TRAINING_DATA / "corrections/labeled_images/" / (str(index) + img.suffix))
                index += 1

    if img_count == 0:
        print("No images to label.")
        return None
    writer.writerows(rows)
    print("Image labels appended to csv file.")
    return True


def compile_corrections_data(model):

    try:
        # read from csv file of labels
        readfile = open(paths.TRAINING_DATA / "corrections/labels.csv", newline='', mode='r')

        reader = csv.reader(readfile)
        data = {}

        for row in reader:
            img_data = []
            for img in (paths.TRAINING_DATA / "corrections/labeled_images").glob(str(row[0])):

                features = analysis.HandstandFeatures(image.load(img), True)

                vectors_list = []
                if features.left_visible:
                    for vec in features.form_vectors:
                        vectors_list.append(vec.to_list())
                else:
                    for vec in features.form_vectors:
                        vectors_list.append(vec.flip_x().to_list())

                for i in range(1, 9):
                    img_data.append(int(row[i]))

                for vec in vectors_list:
                    img_data.append(vec)
                data[row[0]] = img_data

        json_obj = json.dumps(data)

        (paths.TRAINING_DATA / "corrections/data.json").write_text(json_obj)
        print("Labels and vectors written to data.json.")
        return
    except OSError:
        raise OSError("Couldn't open csv_file in read mode.")


def build_corrections(shape):

    model = Sequential()

    model.add(Input(shape))  # input layer kernel_initializer='he_uniform',
    model.add(Flatten())
    model.add(Dense(24, activation="relu"))  # hidden layer
    model.add(Dense(224, activation="relu"))  # hidden layer
    model.add(Dropout(0.1))
    model.add(Dense(8, activation="sigmoid"))  # output layer

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print(model.summary())

    model_structure = model.to_json()
    p = paths.MODELS / "corrections/structure.json"
    p.write_text(model_structure)


def build_identify(shape):

    model = Sequential()

    model.add(Input(shape))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print(model.summary)

    model_structure = model.to_json()
    p = paths.MODELS / "identify/structure.json"
    p.write_text(model_structure)


def train_corrections():
    # Path to folders with training data
    with open(paths.TRAINING_DATA / "corrections/data.json", 'r') as json_in:
        datas = json.load(json_in)

        all_vectors = []
        all_labels = []

        # Load all the not-dog images
        for data in datas.values():
            vectors = data[8:]
            labels = data[0:8]

            all_vectors.append(vectors)
            all_labels.append(labels)

        x_train = np.array(all_vectors).astype("float32")
        y_train = np.array(all_labels).astype("float64")

        # continue..
        p = paths.MODELS / "corrections/structure.json"
        structure = p.read_text()

        model = model_from_json(structure)

        model.fit(
            x_train,
            y_train,
            batch_size=6,
            epochs=200,
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    paths.MODELS / "corrections/.weights.h5",
                    monitor="loss",
                    verbose=1,
                    save_weights_only=True,
                    save_best_only=True
                )
            ]
        )
        # model.save_weights(paths.MODELS / "corrections/.weights.h5")


def train_identify():

    x_train = []
    y_train = []
    for img in (paths.TRAINING_DATA / "identify/not_handstand").iterdir():
        features = analysis.HandstandFeatures(image.load(img), static=True)
        if features.form_vectors is not None:
            vectors_list = []
            if features.left_visible:
                for vec in features.form_vectors:
                    vectors_list.append(vec.to_list())
            else:
                for vec in features.form_vectors:
                    vectors_list.append(vec.flip_x().to_list())

            x_train.append(vectors_list)
            y_train.append(0)

    for img in (paths.TRAINING_DATA / "corrections/labeled_images").iterdir():
        features = analysis.HandstandFeatures(image.load(img), static=True)
        if features.form_vectors is not None:
            vectors_list = []
            if features.left_visible:
                for vec in features.form_vectors:
                    vectors_list.append(vec.to_list())
            else:
                for vec in features.form_vectors:
                    vectors_list.append(vec.flip_x().to_list())

            x_train.append(vectors_list)
            y_train.append(1)

    x_train = np.array(x_train).astype("float32")
    y_train = np.array(y_train).astype("int")

    p = paths.MODELS / "identify/structure.json"
    structure = p.read_text()

    model = model_from_json(structure)

    model.fit(
        x_train,
        y_train,
        batch_size=6,
        epochs=100,
        shuffle=True,
        callbacks=[
            ModelCheckpoint(
                paths.MODELS / "identify/.weights.h5",
                monitor="accuracy",
                verbose=1,
                save_weights_only=True,
                save_best_only=True
            )
        ]
    )


def ask_continue():

    while True:
        print("Would you like to continue [Y/N]?")
        entry = input()
        print("")
        if str(entry).lower() == 'y':
            return True
        elif str(entry).lower() == 'n':
            return False


# main
if __name__ == "__main__":

    model = None
    while True:
        print("Select Option:\n"
              "[1]: prepare correction labels\n"
              "[2]: build and retrain corrections model\n"
              "[3]: build and retrain identify model")
        entry = input()
        print("")

        if entry.isdigit():
            if int(entry) == 1:
                if prepare_labels():
                    if model is None:
                        model = load_corrections_model()
                    compile_corrections_data(model=model)

                if not ask_continue():
                    break

            if int(entry) == 2:
                build_corrections(INPUT_SHAPE)
                train_corrections()

                if not ask_continue():
                    break

            if int(entry) == 3:
                build_identify(INPUT_SHAPE)
                train_identify()

                if not ask_continue():
                    break
