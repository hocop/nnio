import urllib.request
import os
import pathlib
import getpass
import datetime
import numpy as np

from . import __version__

PACKAGE_NAME = 'nnio'

# Temperature logging flag
LOG_TEMPERATURE = False
temperature_files = {}

URL_MARKERS = ['http://', 'https://']

def is_url(s):
    '''
    Check if input string is url or not
    '''
    # Url is always one word
    if len(s.split()) != 1:
        return False
    # Url starts with some specific prefix
    for marker in URL_MARKERS:
        if s.startswith(marker):
            return True
    return False

def file_from_url(url, category='other', file_name=None, use_cached=True):
    '''
    Downloads file to "/home/$USER/.cache/nnio/" if it does not exist already.
    Returns path to the file
    '''
    # Remove prefix from url
    url_path = url
    for marker in URL_MARKERS:
        url_path = url_path.replace(marker, '')
    url_path = '/'.join(url_path.split('/')[:-1])
    # Get base path for file
    base_path = os.path.join(
        '/home',
        getpass.getuser(),
        '.cache',
        PACKAGE_NAME,
        '.'.join(__version__.split('.')[:2]),
        category,
        url_path,
    )
    # Create path if it does not exist
    if not os.path.exists(base_path):
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    # Get file path
    file_name = file_name or url.split('/')[-1]
    file_path = os.path.join(
        base_path,
        file_name
    )
    # Download file from the url
    if not os.path.exists(file_path) or not use_cached:
        print('Downloading file from: {}'.format(url))
        urllib.request.urlretrieve(url, file_path)
        print('Downloaded to: {}'.format(file_path))
    else:
        print('Using cached file: {}'.format(file_path))

    return file_path


# Flag setter
def enable_logging_temperature(enable=True):
    global LOG_TEMPERATURE
    LOG_TEMPERATURE = enable

def log_temperature(device, temperature):
    # Get path to file
    if device in temperature_files:
        file_path = temperature_files[device]
    else:
        # Get base path for file
        base_path = os.path.join(
            '/home',
            getpass.getuser(),
            '.telemetry',
        )
        # Create path if not exists
        if not os.path.exists(base_path):
            pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        # Make file name
        dev_id = device.replace('MYRIAD', '')
        time_format = "vpu{}_%Y-%m-%d_%H-%M-%S".format(dev_id)
        file_name = datetime.datetime.now().strftime(time_format)
        file_path = os.path.join(base_path, file_name)
        # Remember file path
        temperature_files[device] = file_path
        # Write first line to this file
        with open(file_path, 'w') as f:
            f.write('time,vpu_temp\n')
            f.flush()

    val_time = datetime.datetime.now().isoformat(timespec='milliseconds')
    val_vpu_temp = int(temperature)
    with open(file_path, 'a') as f:
        f.write('{},{}\n'.format(val_time, val_vpu_temp))
        f.flush()


class HumanDataBase:
    def __init__(
        self,
        new_entity_threshold=0.25,
        merging_threshold=0.2
    ):
        self.vectors = {}
        self.counts = {}
        self.new_entity_threshold = new_entity_threshold
        self.merging_threshold = merging_threshold

    def find_closest(self, vec):
        keys = list(self.vectors.keys())
        vec = self.normalize(vec)
        distances = [
            self.distance(vec, self.vectors[key])
            for key in keys
        ]
        if len(distances) == 0:
            id_min = None
        else:
            id_min = np.argmin(distances)
        if id_min is None or distances[id_min] > self.new_entity_threshold:
            new_key = str(len(self.vectors))
            print('adding', new_key)
            if id_min is not None:
                print(distances[id_min])
            self.vectors[new_key] = vec
            self.counts[new_key] = 1
            return new_key
        else:
            key = keys[id_min]
            self.vectors[key] = self.vectors[key] * self.counts[key] + vec
            self.vectors[key] = self.normalize(self.vectors[key])
            self.counts[key] = self.counts[key] + 1

        return keys[id_min]

    def optimize(self):
        keys = list(self.vectors.keys())
        if len(keys) < 2:
            return
        # Find vectors which are too close
        distances = []
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                dst = self.distance(
                    self.vectors[keys[i]],
                    self.vectors[keys[j]],
                )
                if dst < self.merging_threshold:
                    distances.append([i, j, dst])
        # Merge two clusters which are the closest
        if len(distances) > 0:
            idx = np.argmin(np.array(distances)[:, 2])
            i, j, dst = distances[idx]
            print('Merging {} with {}'.format(i, j))
            self.vectors[keys[i]] = self.vectors[keys[i]] + self.vectors[keys[j]]
            self.vectors[keys[i]] = self.normalize(self.vectors[keys[i]])
            self.counts[keys[i]] = 1
            del self.vectors[keys[j]]
            del self.counts[keys[j]]


    @staticmethod
    def normalize(vec):
        return vec / np.sqrt((vec**2).sum())
    
    @staticmethod
    def distance(vec1, vec2):
        return 1 - vec1 @ vec2
