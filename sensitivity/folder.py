import os


def create_folder(location):
    if not location:
        return
    else:
        try:
            if not os.path.exists(location):
                os.makedirs(location)
        except OSError as o:
            print ('Error creating directory ' + location + ":" + o.message)
