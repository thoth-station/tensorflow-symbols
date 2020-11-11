Gather and manipulate with symbols provided by TensorFlow
---------------------------------------------------------

Gather and manipulate with symbols as provided by TensorFlow.

Gathering symbols provided by TensorFlow
========================================

To gather symbols provided by TensorFlow issue:

.. code-block:: console

  pipenv run python3 app.py gather --tensorflow-name tensorflow --tensorflow-versions "2.4.0,2.3.0,2.2.0" --index-url "https://pypi.org/simple"

Note the gathering is done on symbols that are available in the `_api/`
directory in a shipped wheel file. This directory is available only since
version ``2.2.0`` hence this tool is not usable for older TensorFlow releases.
See `the upstream issue for more info
<https://github.com/tensorflow/tensorflow/issues/44650>`_.

Merging symbols into one file
=============================

If you wish to use symbols gathered in a pipeline unit used in `adviser
<https://github.com/thoth-station/adviser>`_:

.. code-block:: console

  pipenv run python3 app.py merge --no-patch > api.json

After that, just move the resulting ``api.json`` file to adviser's data
directory.
