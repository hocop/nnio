from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import onnx as _onnx
from ... import output as _output


class SSDMobileNetV1(_model.Model):
    '''
    SSDMobileNetV1 object detection model trained on COCO dataset.

    Model is taken from the `ONNX Model Zoo <https://github.com/onnx/models>`_.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of this model working.
    '''

    URL_MODEL = 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx'
    URL_LABELS = 'https://github.com/amikelive/coco-labels/raw/master/coco-labels-paper.txt'

    def __init__(self):
        '''
        '''
        super().__init__()

        # Load model
        self.model = _onnx.ONNXModel(self.URL_MODEL)

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels')
        self._labels = [
            line.strip()
            for line in open(labels_path)
        ]

    def forward(self, image, return_info=False):
        '''
        :parameter image: np array.
            Input image
        :parameter return_info: bool.
            If ``True``, return inference time.
        :return: list of :class:`nnio.DetectionBox`
        '''
        results = self.model(image, return_info=return_info)
        if return_info:
            results, info = results
        boxes, classes, scores, num_detections = results
        # Parse output
        classes = classes - 1
        out_boxes = []
        for i in range(int(num_detections[0])):
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
            resize=(224, 224),
            dtype='uint8',
            batch_dimension=True,
        )

    @property
    def labels(self):
        '''
        :return: list of COCO labels
        '''
        return self._labels
