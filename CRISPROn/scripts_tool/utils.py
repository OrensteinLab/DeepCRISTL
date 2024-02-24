import os
def get_all_models():
    path = 'tool data/models/'
    # get all folders names in the path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders
