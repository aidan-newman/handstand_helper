import csv
import json
from pathlib import Path
import numpy as np
from tasks import analysis
from tasks import image


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


def build():
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

        x_train = np.array(all_vectors)
        y_train = np.array(all_labels)

        # continue..


def train():
    a = 1
    # continue..


compile_data()
