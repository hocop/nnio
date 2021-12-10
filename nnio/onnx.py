import time

from . import model as _model
from . import utils as _utils


class ONNXModel(_model.Model):
    '''
    This class is used with saved onnx models.

    Usage example::

        # Create model
        model = nnio.ONNXModel('path/to/model.onnx')
        # Create preprocessor
        preproc = nnio.Preprocessing(
            resize=(300, 300),
            dtype='uint8',
            batch_dimension=True,
            channels_first=True,
        )

        # Preprocess your numpy image
        image = preproc(image_rgb)

        # Make prediction
        class_scores = model(image)


    Using this class requires onnxruntime to be installed. See :ref:`installation`.
    '''
    def __init__(
        self,
        model_path: str,
    ):
        '''

        :parameter model_path: URL or path to the .onnx model
        '''
        super().__init__()
        # Download file from internet
        if _utils.is_url(model_path):
            model_path = _utils.file_from_url(model_path, 'models')
        # Load model and create inference session
        self.sess = self._make_interpreter(model_path)

    def forward(self, *inputs, return_info=False):
        assert len(inputs) == len(self.get_input_details())
        # List output names
        outputs = [
            info['name']
            for info in self.get_output_details()
        ]
        # Convert input to a dict
        inputs = {
            info['name']: inp
            for (info, inp) in zip(self.get_input_details(), inputs)
        }
        # Run network and measure time
        start = time.time()
        results = self.sess.run(outputs, inputs)
        end = time.time()
        # Process output a little
        if len(outputs) == 1:
            results = results[0]
        # Return results
        if return_info:
            info = {
                'invoke_time': end - start,
            }
            return results, info
        else:
            return results

    def get_input_details(self):
        return [
            {
                'name': info.name,
                'shape': info.shape,
                'dtype': info.type,
            }
            for info in self.sess.get_inputs()
        ]

    def get_output_details(self):
        return [
            {
                'name': info.name,
                'shape': info.shape,
                'dtype': info.type,
            }
            for info in self.sess.get_outputs()
        ]

    @staticmethod
    def _make_interpreter(model_path):
        'Load model and create inference session'
        import onnxruntime as rt
        sess = rt.InferenceSession(model_path)
        return sess
