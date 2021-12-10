.. _basic_usage:

***********************
Basic Usage
***********************

Using your saved models
===========================

nnio provides four classes for loading models in different formats:

* :class:`nnio.ONNXModel`
* :class:`nnio.EdgeTPUModel`
* :class:`nnio.OpenVINOModel`
* :class:`nnio.TorchModel`

Loaded models can be simply called as functions on numpy arrays. Look at the example:

.. code-block:: python

    import nnio

    # Create model and put it on TPU device
    model = nnio.EdgeTPUModel(
        model_path='path/to/model_quant_edgetpu.tflite',
        device='TPU:0',
    )
    # Create preprocessor
    preproc = nnio.Preprocessing(
        resize=(224, 224),
        dtype='uint8',
        padding=True,
        batch_dimension=True,
    )

    # Preprocess your numpy image
    image = preproc(image_rgb)

    # Make prediction
    class_scores = model(image)

See also :class:`nnio.Preprocessing` documentation.

Description of the basic model classes
===============================================

.. autoclass:: nnio.ONNXModel
    :members:
    :special-members:


.. autoclass:: nnio.EdgeTPUModel
    :members:
    :special-members:


.. autoclass:: nnio.OpenVINOModel
    :members:
    :special-members:

.. autoclass:: nnio.TorchModel
    :members:
    :special-members:
