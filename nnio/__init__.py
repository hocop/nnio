__version__ = '0.3.1.2'

import nnio.zoo

# Base model class
from .model import Model

# Models for specific backends
from .edgetpu import EdgeTPUModel
from .openvino import OpenVINOModel
from .onnx import ONNXModel
from .pytorch import TorchModel

# Preprocessing class
from .preprocessing import Preprocessing

# Output classes
from .output import DetectionBox
