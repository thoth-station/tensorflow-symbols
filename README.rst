Gather and manipulate with symbols provided by TensorFlow
---------------------------------------------------------

Gather and manipulate with symbols as provided by TensorFlow.

The intention of this project is to aggregate TensorFlow symbols for Thoth's
adviser. The aggregation is done by Thoth developers or Thoth administrators to
have symbols available on the recommendation engine side.

Gathering symbols provided by TensorFlow
========================================

To gather symbols provided by TensorFlow issue:

.. code-block:: console

  ./gather.sh

Note the gathering is done on symbols that are available in the `_api/`
directory in a shipped wheel file. This directory is available only since
version ``2.2.0`` hence this tool is not usable for older TensorFlow releases.
See `the upstream issue for more info
<https://github.com/tensorflow/tensorflow/issues/44650>`_.

Once the gathering is done, results are available in ``data/`` directory. Their
merged form is in ``api.json`` file. This file is suitable to be used with
`thoth-adviser <https://github.com/thoth-station/adviser>`__ as a source of API
symbols for TensorFlow's API sieve pipeline unit.
