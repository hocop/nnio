.. _nnio.zoo:

***************
Model Zoo
***************

.. contents:: Table of Contents

Using pretrained models
=========================

Some popular models are already built in nnio. Example of using SSD MobileNet object detection model on CPU:

.. code-block:: python

    # Load model
    model = nnio.zoo.onnx.detection.SSDMobileNetV1()

    # Get preprocessing function
    preproc = model.get_preprocessing()

    # Preprocess your numpy image
    image = preproc(image_rgb)

    # Make prediction
    boxes = model(image)

Here :code:`boxes` is a list of :class:`nnio.DetectionBox` instances.

ONNX
==========

Classification
------------------

.. autoclass:: nnio.zoo.onnx.classification.MobileNetV2
    :members:
    :special-members:

Detection
------------------

.. autoclass:: nnio.zoo.onnx.detection.SSDMobileNetV1
    :members:
    :special-members:

Re-Identification
------------------

.. autoclass:: nnio.zoo.onnx.reid.OSNet
    :members:
    :special-members:

OpenVINO
==========

Detection
------------------

.. autoclass:: nnio.zoo.openvino.detection.SSDMobileNetV2
    :members:
    :special-members:

Re-Identification
------------------

.. autoclass:: nnio.zoo.openvino.reid.OSNet
    :members:
    :special-members:

EdgeTPU
============

Classification
------------------

.. autoclass:: nnio.zoo.edgetpu.classification.MobileNet
    :members:
    :special-members:


Detection
------------------

.. autoclass:: nnio.zoo.edgetpu.detection.SSDMobileNet
    :members:
    :special-members:

.. autoclass:: nnio.zoo.edgetpu.detection.SSDMobileNetFace
    :members:
    :special-members:

Re-Identification
------------------

.. autoclass:: nnio.zoo.edgetpu.reid.OSNet
    :members:
    :special-members:

Segmentation
------------------

.. autoclass:: nnio.zoo.edgetpu.segmentation.DeepLabV3
    :members:
    :special-members:
