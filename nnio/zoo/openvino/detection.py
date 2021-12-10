from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import openvino as _openvino
from ... import output as _output


class SSDMobileNetV2(_model.Model):
    '''
    SSDMobileNetV2 object detection model trained on COCO dataset.

    Model is taken `from openvino <https://docs.openvinotoolkit.org/latest/omz_models_public_ssd_mobilenet_v2_coco_ssd_mobilenet_v2_coco.html>`_ and converted to openvino.

    Here is the `webcam demo <https://github.com/FastSense/nnio/tree/master/demos>`_ of an analogous model (:class:`nnio.zoo.onnx.detection.SSDMobileNetV1`) working.
    '''

    URL_MODEL_BIN = 'https://github.com/FastSense/nnio/raw/development/models/openvino/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_fp16.bin'
    URL_MODEL_XML = 'https://github.com/FastSense/nnio/raw/development/models/openvino/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_fp16.xml'
    URL_LABELS = 'https://github.com/amikelive/coco-labels/raw/master/coco-labels-paper.txt'

    def __init__(
        self,
        device='CPU',
        lite=True,
        threshold=0.5
    ):
        '''
        :parameter device: str.
            Choose Intel device:
            ``CPU``, ``GPU``, ``MYRIAD``.
            If there are multiple devices in your system, you can use indeces:
            ``MYRIAD:0`` but it is not recommended since Intel automatically chooses a free device.
        :parameter threshold: float.
            Detection threshold. It affects sensitivity of the detector.
        :parameter lite: bool.
            If True, use SSDLite version (idk exactly how it is lighter).
        '''
        super().__init__()

        self.threshold = threshold

        path_bin = self.URL_MODEL_BIN
        path_xml = self.URL_MODEL_XML
        if lite:
            path_bin = path_bin.replace('ssd', 'ssdlite')
            path_xml = path_xml.replace('ssd', 'ssdlite')

        # Load model
        self.model = _openvino.OpenVINOModel(path_bin, path_xml, device)

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels')
        self._labels = [
            line.strip()
            for line in open(labels_path)
        ]

    def forward(self, image, return_info=False):
        '''
        :parameter image: np array.
            Input image of a person.
        :parameter return_info: bool.
            If ``True``, return inference time.
        :return: list of :class:`nnio.DetectionBox`
        '''
        results = self.model(image, return_info=return_info)
        if return_info:
            results, info = results
        # Parse output
        out_boxes = []
        for res in results[0, 0]:
            _, label, score, x_min, y_min, x_max, y_max = res
            if score < self.threshold:
                continue
            label = self.labels[int(label) - 1]
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
            dtype='float32',
            channels_first=True,
            batch_dimension=True,
            bgr=True,
        )

    @property
    def labels(self):
        '''
        :return: list of COCO labels
        '''
        return self._labels
