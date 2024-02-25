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


IMAGE_SAVE_HEIGHT = 800
INPUT_SHAPE = (5, 3)


def label_prepper():

    with open(paths.TRAINING_DATA / "labels.csv", "r") as csv_file:
        lines = csv_file.readlines()
        index = int(lines[-1].split('.')[0]) + 1

    with open(paths.TRAINING_DATA / "labels.csv", 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        cancel = False
        rows = []
        num = 0
        for img in (paths.TRAINING_DATA / "potential_images").iterdir():
            if cancel:
                break
            if img.name.endswith(".jpg") or img.name.endswith(".png"):
                num += 1
                image.compress_file(img, height=IMAGE_SAVE_HEIGHT)
                analysis.analyze_image(image.load(img),
                                       save_file=False,
                                       predict=False,
                                       window=True,
                                       hold=True,
                                       destroy_windows=False,
                                       annotate=1
                                       )

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

        if num == 0:
            print("No images to label.")
            return
        writer.writerows(rows)


def compile_data():

    with open(paths.TRAINING_DATA / "labels.csv", newline='', mode='r') as readfile:

        reader = csv.reader(readfile)
        data = {}

        for row in reader:
            img_data = []
            for img in (paths.TRAINING_DATA / "labeled_images").glob(str(row[0])):
                vecs = analysis.analyze_image(image.load(img), predict=False)
                img_data.append(int(row[1]))
                img_data.append(int(row[2]))
                img_data.append(int(row[3]))
                img_data.append(int(row[4]))
                img_data.append(int(row[5]))
                img_data.append(int(row[6]))
                img_data.append(int(row[7]))
                img_data.append(int(row[8]))
                for vec in vecs:
                    img_data.append(vec)
                data[row[0]] = img_data

        json_obj = json.dumps(data)

        (paths.TRAINING_DATA / "data.json").write_text(json_obj)
        print("Labels and vectors written to data.json.")


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
            epochs=1000,
            shuffle=True
        )

        model.save_weights(str("model/.weights.h5"))


# label_prepper()
compile_data()
build(INPUT_SHAPE)
train()
