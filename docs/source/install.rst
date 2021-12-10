.. _installation:

Installation
===================

Basic installation is simple:

.. code-block:: bash

    pip install nnio

To use one of backends, additional installs are needed:

ONNX
-------------

To work with onnx backend, install onnxruntime package:

.. code-block:: bash

    pip install onnxruntime


EdgeTPU
-----------

To work with EdgeTPU models, :code:`tflite_runtime` is required.  
See the installation guide: https://www.tensorflow.org/lite/guide/python.

If you intend to only use CPU inference, tensorflow installation will be enough.

OpenVINO
----------

To work with OpenVINO models user needs to install openvino package.  
The easiest way to do it is to use :code:`openvino/ubuntu18_runtime` docker.  
The following command allows to pass all Myriad and GPU devices into docker container:

.. code-block:: bash

    docker run -itu root:root --rm \
    -v /var/tmp:/var/tmp \
    --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v "$(pwd):/input" openvino/ubuntu18_runtime

Torch
-----

To work with saved torch models, :code:`torch` package needs to be installed. It weights around 0.8 GB, hense it is recommended to use other backends instead.

To install :code:`torch`:

.. code-block:: bash

    pip3 install torch