import abc


class Model(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        r'''
        This method is called when the model is called.

        :parameter \*inputs: numpy arrays, Inputs to the model
        :parameter return_info: bool, If True, will return inference time
        :return: numpy array or list of numpy arrays.
        '''

    def get_preprocessing(self):
        """
        :return: :class:`nnio.Preprocessing` object.
        """

    def get_input_details(self):
        """
        :return: human-readable model input details.
        """

    def get_output_details(self):
        """
        :return: human-readable model output details.
        """
