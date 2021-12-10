from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import edgetpu as _edgetpu


class OSNet(_model.Model):
    '''
    Omni-Scale Feature Network for Person Re-ID taken from `torchreid <https://github.com/KaiyangZhou/deep-person-reid>`_ and converted to tflite.

    This is the quantized version. It is not as accurate as its onnx and openvino versions.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of this model (onnx version) working.
    '''

    URL_CPU = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_quant.tflite'
    URL_TPU = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_quant_edgetpu.tflite'

    def __init__(
        self,
        device='CPU',
    ):
        '''
        :parameter device: str.
            ``CPU`` by default.
            Set ``TPU`` or ``TPU:0`` to use the first EdgeTPU device.
            Set ``TPU:1`` to use the second EdgeTPU device etc.
        '''
        super().__init__()

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU
        else:
            model_path = self.URL_TPU
        self.model = _edgetpu.EdgeTPUModel(model_path, device)

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
            channels_first=False,
            batch_dimension=True,
            padding=True,
        )
