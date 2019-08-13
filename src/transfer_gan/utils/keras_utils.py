from tensorflow.python.keras.models import model_from_json


def load_model_from_json(filename):
    with open(filename, "r") as json_file:
        loaded_json_file = json_file.read()

    return model_from_json(loaded_json_file)


def save_model_to_json(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as model_file:
        model_file.write(model_json)
