import cv2
import numpy as np
import os

from . import model as _model
from . import utils as _utils

class Preprocessing(_model.Model):
    '''
    This class provides functionality of the image preprocessing.

    Example::

        preproc = nnio.Preprocessing(
            resize=(224, 224),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            batch_dimension=True,
            channels_first=True,
        )

        # Use with numpy image
        image_preprocessed = preproc(image_rgb)

        # Or use to read image from disk
        image_preprocessed = preproc('path/to/image.png')

        # Or use to read image from the web
        image_preprocessed = preproc('http://www.example.com/image.png')

    Object of this type is returned every time you call ``get_preprocessing()`` method of any model from :ref:`nnio.zoo`.
    '''
    def __init__(
        self,
        resize=None,
        dtype=None,
        divide_by_255=None,
        means=None,
        stds=None,
        scales=None,
        imagenet_scaling=False,
        to_gray=None,
        padding=False,
        channels_first=False,
        batch_dimension=False,
        bgr=False,
    ):
        '''
        :parameter resize: ``None`` or ``tuple``.
            (width, height) - the new size of image
        :parameter dtype: ``str`` or ``np.dtype``.
            Data type. By default will use ``uint8``.
        :parameter divide_by_255: ``bool``.
            Divide input image by 255. This is applied before ``means``, ``stds`` and ``scales``.
        :parameter means: ``float`` or iterable or ``None``.
            Substract these values from each channel
        :parameter stds: `float`` or iterable or ``None``.
            Divide each channel by these values
        :parameter scales: ``float`` or iterable or ``None``.
            Multipy each channel by these values
        :parameter imagenet_scaling: apply imagenet scaling.
            It is equivalent to ``divide_by_255=True, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]``.
            If this is specified, arguments ``divide_by_255``, ``means``, ``stds``, ``scales`` must be ``None``.
        :parameter to_gray: if ``int``, then convert rgb image to grayscale with specified number of channels (usually 1 or 3).
        :parameter padding: ``bool``.
            If ``True``, images will be resized with the same aspect ratio
        :parameter channels_first: ``bool``.
            If ``True``, image will be returned in ``[B]CHW`` format.
            If ``False``, ``[B]HWC``.
        :parameter batch_dimension: ``bool``.
            If ``True``, add first dimension of size 1.
        :parameter bgr: ``bool``.
            If ``True``, change channels to BRG order.
            If ``False``, keep the RGB order.
        '''
        self.resize = resize
        self.dtype = dtype or 'uint8'
        self.divide_by_255 = divide_by_255 or False
        if divide_by_255:
            MSG = 'If dividing image by 255, specify float data type'
            assert 'float' in dtype, MSG
        self.means = means
        if stds is not None and scales is not None:
            raise BaseException('Either scales or stds may be specified, not both')
        self.stds = stds
        self.scales = scales
        self.to_gray = to_gray
        assert to_gray is None or isinstance(to_gray, int)
        if to_gray is not None and to_gray <= 0 and channels_first:
            raise BaseException('to_gray <= 0 means that there will be no channel dimension, but found channels_first=True')
        self.padding = padding
        self.channels_first = channels_first
        self.batch_dimension = batch_dimension
        self.bgr = bgr

        if imagenet_scaling:
            if (
                divide_by_255 is not None
                or
                means is not None
                or
                stds is not None
                or
                scales is not None
            ):
                raise BaseException('If imagenet_scaling is True, then arguments divide_by_255, means, stds, scales must be None')
            MSG = 'If imagenet_scaling is True, specify float data type'
            assert dtype is None or 'float' in dtype, MSG
            if dtype is None:
                self.dtype = 'float32'
            self.means = [123.675, 116.28, 103.53]
            self.stds = [58.395, 57.12, 57.375]
        
        # Optimize
        self._scales = None if self.scales is None else np.array(self.scales)[None, None]
        self._means = None if self.means is None else np.array(self.means)[None, None]
        if self.stds is not None:
            self._scales = 1 / np.array(self.stds)[None, None]
        if self.divide_by_255:
            if self._scales is not None:
                self._scales = self._scales / 255
            else:
                self._scales = 1 / 255
            if self._means is not None:
                self._means = self._means * 255

    def forward(self, image, return_original=False):
        '''
        Preprocess the image.

        :parameter image: np.ndarray of type ``uint8`` or ``str``
            RGB image
            If ``str``, it will be concerned as image path.
        :parameter return_original: ``bool``.
            If ``True``, will return tuple of ``(preprocessed_image, original_image)``
        '''
        # Read image
        if isinstance(image, str):
            image = self._read_image(image)
        if return_original:
            orig_image = image.copy()
        if str(image.dtype) != 'uint8':
            raise BaseException('Input image data type for preprocessor must be uint8')

        # Convert colors
        if self.bgr:
            image = image[:, :, ::-1]

        # Resize image
        if self.resize is not None:
            image = self._resize_image(image, self.resize, self.padding)

        # Shift and scale
        if self._means is not None:
            image = image - self._means
        if self._scales is not None:
            image = image * self._scales

        # Convert to grayscale
        if self.to_gray is not None:
            image = image.mean(2, keepdims=self.to_gray > 0)
            if self.to_gray > 1:
                image = image.repeat(self.to_gray, axis=2)

        # Change shape
        if self.channels_first:
            image = image.transpose([2, 0, 1])
        if self.batch_dimension:
            image = image[None]

        # Change datatype
        image = image.astype(self.dtype)

        if return_original:
            return image.copy(), orig_image
        else:
            return image.copy()

    @staticmethod
    def _read_image(path):
        ''' Read image from file or url '''
        # Download image if path is url
        is_url = _utils.is_url(path)
        if is_url:
            path = _utils.file_from_url(path, 'temp', use_cached=False)
        # Read image
        # pylint: disable=no-member
        image = cv2.imread(path)
        # Delete temporary file
        if is_url:
            os.remove(path)
        # Throw exception
        if image is None:
            raise BaseException('Cannot read ' + path)
        # Convert from BGR to RGB
        image = image[:, :, ::-1]
        return image

    @staticmethod
    def _resize_image(image, resize, padding=False):
        ''' Resize image
        
        :parameter image: np.ndarray of type ``uint8`` or ``str``
            RGB image
            If ``str``, it will be concerned as image path.
        :parameter resize: ``None`` or ``tuple``.
            (width, height) - the new size of image
        :parameter padding: ``bool``.
            If ``True``, images will be resized with the same aspect ratio
        :return: np.array. Resized image.
        '''
        if not padding:
            # pylint: disable=no-member
            image = cv2.resize(image, resize)
        else:
            # Resize saving the aspect ratio
            ratio_0 = image.shape[1] / resize[0]
            ratio_1 = image.shape[0] / resize[1]
            ratio = max(ratio_0, ratio_1)
            new_size = (
                int(image.shape[1] / ratio),
                int(image.shape[0] / ratio)
            )
            # pylint: disable=no-member
            image = cv2.resize(image, new_size)
            # Pad with zeros
            padding = np.zeros([resize[1], resize[0], 3],
                               dtype=image.dtype)
            start_0 = (padding.shape[0] - image.shape[0]) // 2
            start_1 = (padding.shape[1] - image.shape[1]) // 2
            padding[
                start_0: start_0 + image.shape[0],
                start_1: start_1 + image.shape[1],
            ] = image
            image = padding.copy()
        return image

    def __str__(self):
        '''
        :return: full description of the ``Preprocessing`` object
        '''
        s = 'nnio.Preprocessing(resize={}, dtype={}, divide_by_255={}, means={}, stds={}, scales={}, to_gray={}, padding={}, channels_first={}, batch_dimension={}, bgr={})'
        s = s.format(
            self.resize,
            self.dtype,
            self.divide_by_255,
            self.means,
            self.stds,
            self.scales,
            self.to_gray,
            self.padding,
            self.channels_first,
            self.batch_dimension,
            self.bgr,
        )
        return s

    def __eq__(self, other):
        '''Compare two ``Preprocessing`` objects. Returns ``True`` only if all preprocessing parameters are the same.'''
        return str(self) == str(other)
