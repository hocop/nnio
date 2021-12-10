**************
What is it
**************

Very simple python API for inferencing neural networks. Ideal for sharing your models with colleagues who are not data scientists.

It supports running models on CPU as well as some of the edge devices:

* `Google USB Accelerator <https://coral.ai/products/accelerator/>`_
* `Intel Compute Stick <https://www.intel.ru/content/www/ru/ru/products/boards-kits/compute-stick.html>`_
* Intel integrated GPUs

For each device there exists an own library and a model format. We wrap all those in a single well-defined python package.

Look at this simple example:

.. code-block:: python

    import nnio

    # Create model and put it on a Google Coral Edge TPU device
    model = nnio.EdgeTPUModel(
        model_path='path/to/model_quant_edgetpu.tflite',
        device='TPU',
    )
    # Create preprocessor
    preproc = nnio.Preprocessing(
        resize=(224, 224),
        batch_dimension=True,
    )

    # Preprocess your numpy image
    image = preproc(image_rgb)

    # Make prediction
    class_scores = model(image)

**nnio** was developed for the `Fast Sense X <https://fastsense.readthedocs.io/en/latest/>`_ microcomputer.
It has **six neural accelerators**, which are all supported by nnio:

* 3 x `Google Coral Edge TPU <https://coral.ai/>`_
* 2 x `Intel Myriad VPU <https://www.intel.ru/content/www/ru/ru/products/processors/movidius-vpu/movidius-myriad-x.html>`_
* an Intel integrated GPU

Installation
=============
nnio is simply installed with pip, but it requires some additional libraries.
See :ref:`installation`.


Usage
=============

There are 3 ways one can use nnio:

1. Loading your saved models for inference - :ref:`basic_usage`
2. Using already prepared models from our model zoo: :ref:`nnio.zoo`
3. Using our API to wrap around your own custom models. :ref:`extending`


.. toctree::
    :caption: Table of Contents
    :maxdepth: 2

    Overview <self>
    install
    basic_usage
    zoo
    utils
    extending