from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import openvino as _openvino


class OSNet(_model.Model):
    '''
    Omni-Scale Feature Network for Person Re-ID taken from `here <https://github.com/KaiyangZhou/deep-person-reid>`_ and converted to openvino.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of this model (onnx version) working.
    '''

    URL_MODEL_BIN = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_fp16.bin'
    URL_MODEL_XML = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_fp16.xml'

    def __init__(
        self,
        device='CPU',
    ):
        '''
        :parameter device: str.
            Choose Intel device:
            ``CPU``, ``GPU``, ``MYRIAD``.
            If there are multiple devices in your system, you can use indeces:
            ``MYRIAD:0`` but it is not recommended since Intel automatically chooses a free device.
        '''
        super().__init__()

        # Load model
        self.model = _openvino.OpenVINOModel(self.URL_MODEL_BIN, self.URL_MODEL_XML, device)

    def forward(self, image, return_info=False):
        '''
        :parameter image: np array.
            Input image of a person.
        :parameter return_info: bool.
            If ``True``, return inference time.
        :return: np.array of shape ``[512]`` - person appearence vector. You can compare them by cosine or Euclidian distance.
        '''
        out = self.model(image, return_info=return_info)
        if return_info:
            vector, info = out
        else:
            vector = out
        vector = vector[0]
        if return_info:
            return vector, info
        else:
            return vector

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(128, 256),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            channels_first=True,
            batch_dimension=True,
            padding=True,
        )
