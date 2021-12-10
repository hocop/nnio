from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import onnx as _onnx


class OSNet(_model.Model):
    '''
    Omni-Scale Feature Network for Person Re-ID taken from `here <https://github.com/KaiyangZhou/deep-person-reid>`_ and converted to onnx.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of this model working.
    '''

    URL = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_op10.onnx'

    def __init__(
        self,
    ):
        '''
        '''
        super().__init__()

        # Load model
        self.model = _onnx.ONNXModel(self.URL)

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
