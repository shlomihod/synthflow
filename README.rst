=========
Synthflow
=========

|ci| |arxiv|

.. |ci| image:: https://github.com/shlomihod/synthflow/actions/workflows/ci.yaml/badge.svg?branch=main
   :target: https://github.com/shlomihod/synthflow/actions/workflows/ci.yaml

.. |arxiv| image:: https://img.shields.io/badge/arXiv-2405%2E00267-A42C25?logo=arxiv&logoColor=A42C25
   :target: https://arxiv.org/abs/2405.00267

**Synthflow** is a Python package for facilitating the end-to-end production of differentially private synthetic data. It has been utilized in the public release of Israel's National Birth Registry. For additional information about the release, please refer to `https://birth.dataset.pub`.

Setup
=====

Prerequisites
-------------

- Python 3.8
- `Poetry <https://python-poetry.org/docs/#installation>`_

Installation
------------

1. Clone this repository:

.. code-block:: bash

    git clone https://github.com/shlomihod/synthflow.git

2. Navigate to the directory:

.. code-block:: bash

    cd synthflow

3. Install dependencies using Poetry:

.. code-block:: bash

    poetry install

Usage
=====

Run the `synthflow` tool with the `--help` option to see available commands:

.. code-block:: bash

    poetry run python -m synthflow --help

Commands and Options
--------------------

- `execute`: Run the synthetic data generation and evaluation process
- `evaluate`: Evaluate a given synthetic data
- `span`: Span the space of generation configurations for a given privacy parameters (epsilon, delta)
- `parallel`: Run the synthetic data generation and evaluation process (`execute`) in parallel
- `report`: Generate an evaluation report of an execution

For a complete list of options and flags, refer to the initial command list above.

Testing
=======

Run tests using `pytest`:

.. code-block:: bash

    pytest


License
=======

This project is licensed under the MIT License. See `LICENSE <LICENSE>`_ for details.

Support
=======

For questions or issues, please open an `issue <https://github.com/shlomihod/synthflow/issues>`_.
