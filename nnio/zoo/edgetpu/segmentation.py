from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import edgetpu as _edgetpu


class DeepLabV3(_model.Model):
    '''
    DeepLabV3 instance segmentation model trained in Pascal VOC dataset.

    Model is taken from the `google-coral repo <https://github.com/google-coral/edgetpu/tree/master/test_data>`_.
    '''

    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_dm05_pascal_quant.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/pascal_voc_segmentation_labels.txt'

    def __init__(self, device='CPU'):
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

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels')
        self._labels = []
        for line in open(labels_path):
            if line.strip() != '':
                self.labels.append(line.strip())
            else:
                break

    def forward(self, image):
        '''
        :parameter image: np array.
            Input image
        :return: numpy array.
            Segmentation map of the same size as the input image: ``shape=[batch, 513, 513]``.
            For each pixel gives an integer denoting class.
            Class labels are available through ``.labels`` attribute of this object.
        '''
        segmentation = self.model(image)[0]
        return segmentation

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(513, 513),
            dtype='uint8',
            padding=False,
            batch_dimension=True,
        )

    @property
    def labels(self):
        '''
        :return: list of Pascal VOC labels
        '''
        return self._labels
