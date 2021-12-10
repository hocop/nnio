from setuptools import setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1].strip()
    else:
        raise RuntimeError("Unable to find version string.")


LONG_DESCRIPTION = '''
Please refer to the [project's documentation](https://nnio.readthedocs.io/).

## What is it

**nnio** is a light-weight python package for easily running neural networks.

It supports running models on CPU as well as some of the edge devices:

* [Google USB Accelerator](https://coral.ai/products/accelerator/)
* [Intel Compute Stick](https://www.intel.ru/content/www/ru/ru/products/boards-kits/compute-stick.html)
* Intel integrated GPUs

For each device there exists an own library and a model format. We wrap all those in a single well-defined python package.

Look at this simple example:

```python
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
```

**nnio** was developed for the [Fast Sense X](https://fastsense.readthedocs.io/en/latest/) microcomputer.
It has **six neural accelerators**, which are all supported by nnio:

* 3 x [Google Coral Edge TPU](https://coral.ai/)
* 2 x [Intel VPU](https://www.intel.ru/content/www/ru/ru/products/processors/movidius-vpu/movidius-myriad-x.html)
* an integrated Intel GPU

[More usage examples](https://nnio.readthedocs.io/en/latest/basic_usage.html) can be found in the [documentation](https://nnio.readthedocs.io/).
'''

setup(
    name='nnio',
    version=get_version('nnio/__init__.py'),
    description='Neural network inference on accelerators simplified',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Ruslan Baynazarov',
    author_email='ruslan.baynazarov@fastsense.tech',
    url='https://github.com/FastSense/nnio',
    packages=['nnio', 'nnio.zoo', 'nnio.zoo.edgetpu', 'nnio.zoo.openvino', 'nnio.zoo.onnx'],
    license="MIT",
    install_requires=[
        'numpy',
        'opencv-python',
    ]
)
