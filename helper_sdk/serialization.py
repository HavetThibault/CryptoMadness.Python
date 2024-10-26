import os
import pickle


def save_binary(obj, filepath: str):
    if not os.path.exists(os.path.dirname(filepath)):
        os.mkdir(os.path.dirname(filepath))
    with open(filepath, 'wb') as file:
        return pickle.dump(obj, file)

def load_binary(filepath: str):
    with open(filepath, 'rb') as file:
        training_memory = pickle.load(file)
        return training_memory
