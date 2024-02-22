import csv
import json
from pathlib import Path
import shutil

import numpy as np
from tasks import analysis
from tasks import image

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten

INPUT_SHAPE = (5, 3)


def label_prepper():

    index = None
    with open("training_data/labels.csv", "r") as csv_file:
        lines = csv_file.readlines()
        index = int(lines[-1].split('.')[0]) + 1

    with open("training_data/labels.csv", 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        rows = []
        num = 0
        for img in Path("training_data/potential_images").iterdir():
            if img.name.endswith(".jpg") or img.name.endswith(".png"):
                num += 1
                analysis.analyze_image(image.load(str(img.absolute())), False, True, hold=True, destroy_windows=False)

                row = [str(index) + img.suffix]
                for c in analysis.CORRECTIONS:
                    invalid = True
                    while invalid:
                        print(c + "?:")
                        entry = input()
                        if str(entry).lower() == "x":
                            shutil.move(img, Path("training_data/bad_images/" + str(index) + img.suffix))
                            break
                        if entry.isdigit():
                            entry = int(entry)
                            if entry == 0 or entry == 1:
                                invalid = False
                                row.append(entry)
                    else:
                        continue
                    break

                if len(row) == len(analysis.CORRECTIONS)+1:
                    rows.append(row)
                    shutil.move(img, Path("training_data/labeled_images/" + str(index) + img.suffix))
                    index += 1

        if num == 0:
            print("No images to label.")
            return
        writer.writerows(rows)


def compile_data():

    with open("training_data/labels.csv", newline='', mode='r') as readfile:

        reader = csv.reader(readfile)
        data = {}

        next(reader)  # skip column titles
        for row in reader:
            img_data = []
            for file in Path("training_data/images").glob(str(row[0]) + ".*"):
                vecs = analysis.analyze_image(image.load(str(file.absolute())))
                img_data.append(int(row[1]))
                img_data.append(int(row[2]))
                for vec in vecs:
                    img_data.append(vec.to_list())
                data[row[0]] = img_data

        json_obj = json.dumps(data)

        Path("training_data/data.json").write_text(json_obj)
        print("Labels and vectors written to data.json.")


def build(shape):

    model = Sequential()

    model.add(Dense(24, input_shape=shape, activation="relu"))  # input layer kernel_initializer='he_uniform',
    model.add(Flatten())
    model.add(Dense(24, activation="relu"))  # hidden layer
    model.add(Dropout(0.25))  # dropout
    model.add(Dense(2, activation="sigmoid"))  # output layer

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print(model.summary())

    model_structure = model.to_json()
    p = Path("model/structure.json")
    p.write_text(model_structure)


def train():
    # Path to folders with training data
    with open("training_data/data.json", 'r') as json_in:
        datas = json.load(json_in)

        all_vectors = []
        all_labels = []

        # Load all the not-dog images
        for data in datas.values():
            vectors = data[2:]
            labels = data[0:2]

            all_vectors.append(vectors)
            all_labels.append(labels)

        x_train = np.array(all_vectors).astype("float32")
        y_train = np.array(all_labels).astype("float64")

        # continue..
        p = Path("model/structure.json")
        structure = p.read_text()

        model = model_from_json(structure)

        model.fit(
            x_train,
            y_train,
            batch_size=5,
            epochs=100,
            shuffle=True
        )

        model.save_weights(str("model/.weights.h5"))


# build(INPUT_SHAPE)
# train()
label_prepper()
