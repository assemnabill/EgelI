import os
import sys
from datetime import datetime


class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def extract_valid_models(models_folder=os.path.join("resources", "models")):
    files = os.listdir(models_folder)
    valid = []
    for model in files:
        if os.path.isdir(os.path.join(models_folder, model)):
            valid.append(model)

    return valid


def remove_all(lista, listb):
    return [obj for obj in lista if obj not in listb]


def get_record_datetime_string():
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")