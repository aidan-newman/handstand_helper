import csv
import json
import shutil

import numpy as np

import paths
from tasks import analysis
from tasks import image
from tasks import file

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint


IMAGE_SAVE_HEIGHT = 800
INPUT_SHAPE = (5, 3)


def label_prepper():

    # open csv file
    try:
        csv_file = open(paths.TRAINING_DATA / "labels.csv", "r")
    except OSError:
        raise OSError("Couldn't open csv_file in read mode.")

    lines = csv_file.readlines()
    index = int(lines[-1].split('.')[0]) + 1

    try:
        csv_file = open(paths.TRAINING_DATA / "labels.csv", 'a', newline='')
    except OSError:
        raise OSError("Couldn't open csv_file in append mode.")

    writer = csv.writer(csv_file, delimiter=',')

    cancel = False
    rows = []
    img_count = 0
    for img in (paths.TRAINING_DATA / "potential_images").iterdir():
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
                        file.safe_move(img, paths.TRAINING_DATA / "bad_images")
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
                shutil.move(img, paths.TRAINING_DATA / "labeled_images/" / (str(index) + img.suffix))
                index += 1

    if img_count == 0:
        print("No images to label.")
        return
    writer.writerows(rows)
    print("Image labels appended to csv file.")


def compile_data(model):

    try:
        # read from csv file of labels
        readfile = open(paths.TRAINING_DATA / "labels.csv", newline='', mode='r')

        reader = csv.reader(readfile)
        data = {}

        for row in reader:
            img_data = []
            for img in (paths.TRAINING_DATA / "labeled_images").glob(str(row[0])):
                # COME BACK

                _, vecs, _, _ = analysis.analyze_image(image.load(img), model, predict=False)
                for i in range(1, 9):
                    img_data.append(int(row[i]))

                for vec in vecs:
                    img_data.append(vec)
                data[row[0]] = img_data

        json_obj = json.dumps(data)

        (paths.TRAINING_DATA / "data.json").write_text(json_obj)
        print("Labels and vectors written to data.json.")
        return
    except OSError:
        raise OSError("Couldn't open csv_file in read mode.")


def build(shape):

    model = Sequential()

    model.add(Dense(24, input_shape=shape, activation="relu"))  # input layer kernel_initializer='he_uniform',
    model.add(Flatten())
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
    p = paths.MODEL / "structure.json"
    p.write_text(model_structure)


def train():
    # Path to folders with training data
    with open(paths.TRAINING_DATA / "data.json", 'r') as json_in:
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
        p = paths.MODEL / "structure.json"
        structure = p.read_text()

        model = model_from_json(structure)

        model.fit(
            x_train,
            y_train,
            batch_size=6,
            epochs=200,
            shuffle=True,
            callbacks=[
                ModelCheckpoint("model/.weights.h5",
                                monitor="loss",
                                verbose=1,
                                save_weights_only=True,
                                save_best_only=True),
            ]
        )

        # model.save_weights("model/.weights.h5")


# main
if __name__ == "__main__":
    print("Would you like to:\n"
          "[1]: prep labels,\n"
          "[2]: ")
