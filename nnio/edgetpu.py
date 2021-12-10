import platform
import time

from . import model as _model
from . import utils as _utils

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


class EdgeTPUModel(_model.Model):
    '''
    This class works with tflite models on CPU and with quantized tflite models on Google Coral Edge TPU.

    Using this class requires some libraries to be installed. See :ref:`installation`.
    '''
    def __init__(
        self,
        model_path: str,
        device='CPU'
    ):
        '''
        :parameter model_path: URL or path to the tflite model
        :parameter device: str.
            ``CPU`` by default.
            Set ``TPU`` or ``TPU:0`` to use the first EdgeTPU device.
            Set ``TPU:1`` to use the second EdgeTPU device etc.
        '''
        super().__init__()
        # Download file from internet
        if _utils.is_url(model_path):
            model_path = _utils.file_from_url(model_path, 'models')
        # Create interpreter
        assert device == 'CPU' or device.split(':')[0] == 'TPU' or device[0] == ':'
        self.interpreter = self._make_interpreter(model_path, device)
        self.interpreter.allocate_tensors()

    def forward(self, *inputs, return_info=False):
        assert len(inputs) == self.n_inputs
        start = time.time()
        # Put input tensors into model
        for i in range(self.n_inputs):
            tensor = self._input_tensor(i)
            tensor[:, :, :, :] = inputs[i]
            del tensor
        before_invoke = time.time()
        # Call model
        self.interpreter.invoke()
        after_invoke = time.time()
        # Get results from the model
        results = [self._output_tensor(i) for i in range(self.n_outputs)]
        # Process output a little
        if self.n_outputs == 1:
            results = results[0]
        # Return results
        if return_info:
            info = {
                'assign_time': before_invoke - start,
                'invoke_time': after_invoke - before_invoke,
            }
            return results, info
        else:
            return results

    def get_input_details(self):
        return [
            {
                'name': inp['name'],
                'shape': inp['shape'],
                'dtype': str(inp['dtype']),
            }
            for inp in self.interpreter.get_input_details()
        ]

    def get_output_details(self):
        return [
            {
                'name': inp['name'],
                'shape': inp['shape'],
                'dtype': str(inp['dtype']),
            }
            for inp in self.interpreter.get_output_details()
        ]

    @staticmethod
    def _make_interpreter(model_file, device='CPU'):
        ' Load model and create tflite interpreter '
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            print('''
            Warning: tflite_runtime is not installed.
            
            Please follow these instructions to install driver:
            
            https://coral.ai/docs/m2/get-started/#2a-on-linux
            
            And these instructions to install tflite_runtime:
            
            https://www.tensorflow.org/lite/guide/python
            ''')
            if device == 'CPU':
                print('Trying to use tensorflow version on CPU')
                import tensorflow.lite as tflite
            else:
                raise ImportError
        if device != 'CPU':
            if device == 'TPU':
                device = 'TPU:0'
            return tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[
                    tflite.load_delegate(
                        EDGETPU_SHARED_LIB,
                        {
                            'device': device.replace('TPU', '')
                        }
                    )
                ])
        else:
            return tflite.Interpreter(
                model_path=model_file)

    def _input_tensor(self, i=0):
        '''
        Returns input tensor view as function returning numpy array
        '''
        tensor_index = self.interpreter.get_input_details()[i]['index']
        return self.interpreter.tensor(tensor_index)()

    def _output_tensor(self, i=0):
        """Returns output tensor view."""
        tensor_index = self.interpreter.get_output_details()[i]['index']
        tensor = self.interpreter.get_tensor(tensor_index)
        return tensor

    @property
    def n_inputs(self):
        ''' number of input tensors '''
        return len(self.interpreter.get_input_details())

    @property
    def n_outputs(self):
        ''' number of output tensors '''
        return len(self.interpreter.get_output_details())

