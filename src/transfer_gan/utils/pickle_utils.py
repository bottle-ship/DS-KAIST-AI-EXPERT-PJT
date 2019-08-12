import pickle


def save_to_pickle(instance, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(instance, file)


def load_from_pickle(file_name):
    with open(file_name, "rb") as file:
        instance = pickle.load(file)

    return instance
