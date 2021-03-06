# nnio

Please refer to the [project's documentation](https://nnio.readthedocs.io/).

## What is it

Very simple python wrapper for inferencing neural networks. Ideal for sharing your models with colleagues who are not data scientists.

![torch scheme](docs/images/torch_scheme.png)

**Features:**
1. Supports **different formats**: [ONNX](https://nnio.readthedocs.io/en/latest/basic_usage.html#nnio.ONNXModel), [TF Lite](https://nnio.readthedocs.io/en/latest/basic_usage.html#nnio.EdgeTPUModel), [OpenVINO](https://nnio.readthedocs.io/en/latest/basic_usage.html#nnio.OpenVINOModel), [Torch](https://nnio.readthedocs.io/en/latest/basic_usage.html#nnio.TorchModel)
2. Loading any model takes one line of code.
3. Acts on **numpy arrays** and outputs numpy arrays. Inference takes one line of code.
4. Has flexible [nnio.Preprocessing](https://nnio.readthedocs.io/en/latest/utils.html#nnio-preprocessing) class for images, which can also read image from file or URL.
5. **Can read** models and images from **URLs instead of paths**.
6. Has a bunch of **built-in models**. E.g. object detection, classification, person re-identification. See [nnio.zoo](https://nnio.readthedocs.io/en/latest/zoo.html)

It supports running models on CPU as well as some accelerators:

* GPU with cuda support (onnx and pytorch)
* [Google USB Accelerator](https://coral.ai/products/accelerator/) (tflite)
* [Intel Compute Stick](https://www.intel.ru/content/www/ru/ru/products/boards-kits/compute-stick.html) (openVino)
* Intel integrated GPUs (openVino)

For each device there exists an own library and a model format. We wrap all those in a single well-defined python package.

## Code examples

### Loading tflite model and running it on google coral device:

```python
import nnio

# Load model and put it on a Google Coral Edge TPU device
# model_path can be URL
model = nnio.EdgeTPUModel(
    model_path='path/to/model_quant_edgetpu.tflite',
    device='TPU',
)

# Create preprocessor
preproc = nnio.Preprocessing(
    resize=(224, 224),
    batch_dimension=True,
    imagenet_scaling=True,
)

# Load and preprocess the image
# argument can be path, URL or numpy array
image = preproc('path/to/image.png')

# Make prediction
class_scores = model(image)
```

### Loading a simple onnx model:

```python
# Load the model
model = nnio.ONNXModel(
    model_path='path/to/model.onnx',
)
# Inference code will be the same
```
For this example you will need `onnxruntime` or `onnxruntime-gpu` to be installed, depending on what device you want to use. Install them using pip. See [Installation docs](https://nnio.readthedocs.io/en/latest/install.html).

### Using pretrained model:

```python
# Load model
model = nnio.zoo.onnx.detection.SSDMobileNetV1()

# Get preprocessing function
preproc = model.get_preprocessing()

# Preprocess your numpy image
image = preproc(image_rgb)

# Make prediction
boxes = model(image)
```

`boxes` is a list of [nnio.DetectionBox](https://nnio.readthedocs.io/en/latest/utils.html#nnio-detectionbox) objects.

[More usage examples](https://nnio.readthedocs.io/en/latest/basic_usage.html) can be found in the [documentation](https://nnio.readthedocs.io/).
