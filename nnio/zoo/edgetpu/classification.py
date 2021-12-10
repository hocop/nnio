from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import edgetpu as _edgetpu


class MobileNet(_model.Model):
    '''
    MobileNet V2 (or V1) classifier trained on ImageNet

    Model is taken from the `google-coral repo <https://github.com/google-coral/edgetpu/tree/master/test_data>`_
    '''

    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_{}_1.0_224_quant.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_{}_1.0_224_quant_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/imagenet_labels.txt'

    def __init__(self, device='CPU', version='v2'):
        '''
        :parameter device: str.
            ``CPU`` by default.
            Set ``TPU`` or ``TPU:0`` to use the first EdgeTPU device.
            Set ``TPU:1`` to use the second EdgeTPU device etc.
        :parameter version: str.
            Either ``v1`` or ``v2``.
        '''
        super().__init__()

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU.format(version)
        else:
            model_path = self.URL_TPU.format(version)
        self.model = _edgetpu.EdgeTPUModel(model_path, device)

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels')
        self._labels = [
            ' '.join(line.strip().split()[1:])
            for line in open(labels_path)
        ]

    def forward(self, image, return_scores=False):
        '''
        :parameter image: np array.
            Input image
        :parameter return_scores: bool.
            If ``True``, return class scores.
        :return: ``str``: class label.
        '''
        scores = self.model(image)[0]
        label = self.labels[scores.argmax()]
        if return_scores:
            return label, scores
        else:
            return label

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(224, 224),
            dtype='uint8',
            padding=False,
            batch_dimension=True,
        )

    @property
    def labels(self):
        '''
        :return: list of ImageNet classification labels
        '''
        return self._labels
