import os

import datetime


def make_directory(path, time_suffix=False):
    if time_suffix:
        path += datetime.datetime.now().strftime("_%Y%m%d-%H%M%S")

    if os.path.exists(path):
        raise FileExistsError("'%s' is already exists." % path)
    else:
        path = path.split(os.sep)
        for i in range(0, len(path)):
            if not os.path.exists(os.sep.join(path[:i + 1])):
                os.mkdir(os.sep.join(path[:i + 1]))
