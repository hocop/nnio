.. _extending:

Extending nnio
===================

Using our API to wrap around your own custom models
----------------------------------------------------

:class:`nnio.Model` is an abstract class from which all models in nnio are derived. It is easy to use by redefining :code:`forward` method:

.. code-block:: python

    class MyClassifier(nnio.Model):
        def __init__(self):
            super().__init__()
            self.model = SomeModel()

        def forward(self, image):
            # Do something with image
            result = self.model(image)
            # For example, classification
            if result == 0:
                return 'person'
            else:
                return 'cat'

        def get_preprocessing(self):
            return nnio.Preprocessing(
                resize=(224, 224),
                dtype='float',
                divide_by_255=True,
                means=[0.485, 0.456, 0.406],
                stds=[0.229, 0.224, 0.225],
                batch_dimension=True,
                channels_first=True,
            )


We also recommend to define :code:`get_preprocessing` method like in :ref:`nnio.zoo` models. See :class:`nnio.Preprocessing`.
We encourage users to wrap their loaded models in such classes. :class:`nnio.Model` abstract base class is described below:


nnio.Model
--------------

.. autoclass:: nnio.Model
    :members:
