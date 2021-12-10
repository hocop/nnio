import time

from . import model as _model
from . import utils as _utils


class OpenVINOModel(_model.Model):
    '''
    This class works with OpenVINO models on CPU, Intel GPU and Intel Movidius Myriad.

    Using this class requires some libraries to be installed. See :ref:`installation`.
    '''
    def __init__(
        self,
        model_bin: str,
        model_xml: str,
        device='CPU',
    ):
        '''
        :parameter model_bin: URL or path to the openvino binary model file
        :parameter model_xml: URL or path to the openvino xml model file
        :parameter device: str.
            Choose Intel device:
            ``CPU``, ``GPU``, ``MYRIAD``
            If there are multiple devices in your system, you can use indeces:
            ``MYRIAD:0`` but it is not recommended since Intel automatically chooses a free device.
        '''
        super().__init__()

        # Download files from internet
        if _utils.is_url(model_bin):
            model_bin = _utils.file_from_url(model_bin, 'models')
        if _utils.is_url(model_xml):
            model_xml = _utils.file_from_url(model_xml, 'models')

        # Create interpreter
        self.ie, self.net, self.device = self._make_interpreter(model_xml, model_bin, device)

    def forward(self, inputs, return_info=False):
        r'''
        :parameter inputs: numpy array, input to the model
        :parameter return_info: bool, If True, will return inference time
        :return: numpy array or list of numpy arrays.
        '''
        # Find name of the input to the model
        input_name = list(self.net.input_info.keys())[0]
        # Call model
        start = time.time()
        out = self.net.infer({input_name: inputs})
        end = time.time()
        # Process output a little
        if len(out.keys()) == 1:
            out = out[list(out.keys())[0]]
        # Measure temperature
        temperature = None
        if self.device.startswith('MYRIAD.'):
            temperature = self.ie.get_metric(metric_name="DEVICE_THERMAL", device_name=self.device)
            if _utils.LOG_TEMPERATURE:
                _utils.log_temperature(self.device, temperature)
        # Return results
        if return_info:
            info = {
                'invoke_time': end - start,
            }
            if temperature is not None:
                info['temperature'] = temperature
            return out, info
        else:
            return out

    @staticmethod
    def _make_interpreter(model_xml, model_bin, device):
        'Load model and create openvino interpreter'
        try:
            from openvino.inference_engine import IECore
        except ImportError:
            print('''
            Warning: openvino is not installed.
            
            Please install openvino or use openvino docker container.
            ''')
            raise ImportError
        ie = IECore()
        # Get list of MYRIAD devices
        if device.startswith('MYRIAD:'):
            myriads = [
                dev for dev in ie.available_devices
                if 'MYRIAD' in dev
            ]
            _, idx = device.split(':')
            idx = int(idx)
            if len(myriads) <= idx:
                raise BaseException('Cannot find out which device is {}\nAvailable devices: {}'.format(device, ie.available_devices))
            device = myriads[idx]
        # Load model on device
        net = ie.read_network(model_xml, model_bin)
        print('Loading model to:', device)
        net = ie.load_network(net, device)
        return ie, net, device
