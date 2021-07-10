import pickle



def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)