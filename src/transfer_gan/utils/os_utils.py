import os


def make_directory(path):
    if os.path.exists(path):
        raise FileExistsError("'%s' is already exists." % path)
    else:
        path = path.split(os.sep)
        for i in range(0, len(path)):
            if not os.path.exists(os.sep.join(path[:i + 1])):
                os.mkdir(os.sep.join(path[:i + 1]))
