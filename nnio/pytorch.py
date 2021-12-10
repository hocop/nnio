import time

from . import model as _model
from . import utils as _utils


class TorchModel(_model.Model):
    '''
    This class is used with saved torchscript models.

    For saving model, use `torch.jit.trace` (easier) or `torch.jit.script` (harder).

    Usage example::

        # Create model
        model = nnio.TorchModel('path/to/model.pt')
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


    Using this class requires torch to be installed. See :ref:`installation`.
    '''
    def __init__(
        self,
        model_path: str,
        device: str='cpu',
    ):
        '''

        :parameter model_path: URL or path to the torchscript model
        :parameter device: Can be either ``cpu`` or ``cuda``.
        '''
        super().__init__()
        self.device = device

        # Download file from the internet
        if _utils.is_url(model_path):
            model_path = _utils.file_from_url(model_path, 'models')

        import torch
        self.torch = torch
        try:
            self.model = torch.jit.load(model_path)
        except:
            self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()


    def forward(self, *inputs, return_info=False):
        # Convert inputs to torch tensors
        # pylint: disable=no-member
        inp_torch = [self.torch.tensor(inp, device=self.device) for inp in inputs]

        # Run network and measure time
        start = time.time()
        with self.torch.no_grad():
            outp_torch = self.model(*inp_torch)
        end = time.time()
        results = outp_torch.cpu().numpy()

        # Return results
        if return_info:
            info = {
                'invoke_time': end - start,
            }
            return results, info
        else:
            return results
