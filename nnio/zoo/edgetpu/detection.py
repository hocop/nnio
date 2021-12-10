from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import edgetpu as _edgetpu
from ... import output as _output


class SSDMobileNet(_model.Model):
    '''
    MobileNet V2 (or V1) SSD object detector trained on COCO dataset.

    Model is taken from the `google-coral repo <https://github.com/google-coral/edgetpu/tree/master/test_data>`_.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of an analogous model (:class:`nnio.zoo.onnx.detection.SSDMobileNetV1`) working.
    '''

    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_{}_coco_quant_postprocess.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_{}_coco_quant_postprocess_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/coco_labels.txt'

    def __init__(
        self,
        device='CPU',
        version='v2',
        threshold=0.5
    ):
        '''
        :parameter device: str.
            ``CPU`` by default.
            Set ``TPU`` or ``TPU:0`` to use the first EdgeTPU device.
            Set ``TPU:1`` to use the second EdgeTPU device etc.
        :parameter version: str.
            Either "v1" or "v2"
        :parameter threshold: float.
            Detection threshold. Affects the detector's sensitivity.
        '''
        super().__init__()

        self.threshold = threshold

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU.format(version)
        else:
            model_path = self.URL_TPU.format(version)
        self.model = _edgetpu.EdgeTPUModel(model_path, device)

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels_google')
        self._labels = {
            int(line.split()[0]): line.strip().split()[1]
            for line in open(labels_path)
        }

    def forward(self, image, return_info=False):
        '''
        :parameter image: np array.
            Input image
        :parameter return_info: bool.
            If ``True``, return inference time.
        :return: list of :class:`nnio.DetectionBox`
        '''
        out = self.model(image, return_info=return_info)
        if return_info:
            (boxes, classes, scores, _num_detections), info = out
        else:
            boxes, classes, scores, _num_detections = out
        # Parse output
        out_boxes = []
        for i in range(len(boxes[0])):
            if scores[0, i] < self.threshold:
                continue
            y_min, x_min, y_max, x_max = boxes[0, i]
            label = self.labels[int(classes[0, i])]
            score = scores[0, i]
            out_boxes.append(
                _output.DetectionBox(x_min, y_min, x_max, y_max, label, score)
            )
        if return_info:
            return out_boxes, info
        else:
            return out_boxes

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(300, 300),
            dtype='uint8',
            batch_dimension=True,
        )

    @property
    def labels(self):
        '''
        :return: list of COCO labels
        '''
        return self._labels



class SSDMobileNetFace(_model.Model):
    '''
    MobileNet V2 SSD face detector.

    Model is taken from the `google-coral repo <https://github.com/google-coral/edgetpu/tree/master/test_data>`_.
    '''

    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_face_quant_postprocess.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'

    def __init__(
        self,
        device='CPU',
        threshold=0.5
    ):
        '''
        :parameter device: str.
            ``CPU`` by default.
            Set ``TPU`` or ``TPU:0`` to use the first EdgeTPU device.
            Set ``TPU:1`` to use the second EdgeTPU device etc.
        :parameter threshold: float.
            Detection threshold. Affects the detector's sensitivity.
        '''
        super().__init__()

        self.threshold = threshold

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU
        else:
            model_path = self.URL_TPU
        self.model = _edgetpu.EdgeTPUModel(model_path, device)

    def forward(self, image, return_info=False):
        '''
        :parameter image: np array.
            Input image
        :parameter return_info: bool.
            If ``True``, return inference time.
        :return: list of :class:`nnio.DetectionBox`
        '''
        out = self.model(image, return_info=return_info)
        if return_info:
            (boxes, _, scores, _num_detections), info = out
        else:
            boxes, _, scores, _num_detections = out
        # Parse output
        out_boxes = []
        for i in range(len(boxes[0])):
            if scores[0, i] < self.threshold:
                continue
            y_min, x_min, y_max, x_max = boxes[0, i]
            label = 'face'
            score = scores[0, i]
            out_boxes.append(
                _output.DetectionBox(x_min, y_min, x_max, y_max, label, score)
            )
        if return_info:
            return out_boxes, info
        else:
            return out_boxes

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(320, 320),
            dtype='uint8',
            batch_dimension=True,
        )
