************
Installation
************

Requirements
============

To use `idinn`, you will need `Python`_ and a set of required packages. For optimal compatibility and performance, we recommend using the following minimum versions, as specified in ``requirements.txt``:

* Python_      ``>= 3.9``
* matplotlib_  ``>= 3.7.1``
* numba_       ``>= 0.57``
* numpy_       ``>= 2.0``
* PyTorch_     ``>= 2.5``
* tensorboard_ ``>= 2.12``

The required packages will also be installed automatically during the installation process according to ``pyproject.toml``, ensuring all dependency requirements are met in your virtual environment.


Install `idinn`
===============

The package can be installed from `PyPI`_. To do that, run:

.. code-block:: bash

   pip install idinn

Alternatively, if you want to inspect and locally edit the source code, use the following commands:

.. code-block:: bash

   git clone https://gitlab.com/ComputationalScience/idinn.git
   cd idinn
   pip install -e .


.. _Python: https://www.python.org/downloads/
.. _matplotlib: https://matplotlib.org/stable/users/getting_started/
.. _numba: https://numba.pydata.org/
.. _numpy: https://numpy.org/install/
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _tensorboard: https://www.tensorflow.org/tensorboard/get_started
.. _PyPI: https://pypi.org/project/idinn/