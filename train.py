import joblib
from pathlib import Path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

x_train = joblib.load("extracted_features/x_train.dat")
y_train = joblib.load("extracted_features/y_train.dat")
x_test = joblib.load("extracted_features/x_test.dat")
y_test = joblib.load("extracted_features/y_test.dat")

# construct new model
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# train model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True,
    validation_data=[x_test, y_test],
)

# save model
model_structure = model.to_json()
f = Path("model/model_structure.json")
f.write_text(model_structure)

model.save_weights("model/model_weights.h5")
