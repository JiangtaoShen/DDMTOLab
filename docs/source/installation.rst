.. _installation:

Installation
============

Requirements
------------

DDMTOLab requires:

* Python 3.7 or later
* PyTorch
* NumPy
* SciPy
* scikit-learn
* tqdm

Install from GitHub
-------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab
   pip install -r requirements.txt

Set Python Path
---------------

**Windows:**

.. code-block:: powershell

   $env:PYTHONPATH="D:\DDMTOLab;$env:PYTHONPATH"

**Linux/Mac:**

.. code-block:: bash

   export PYTHONPATH=/path/to/DDMTOLab:$PYTHONPATH

Verify Installation
-------------------

Test the installation:

.. code-block:: python

   from Algorithms.STSO.BO import BO
   print("DDMTOLab installed successfully!")

Troubleshooting
---------------

**Import Error: No module named 'Algorithms'**

Make sure PYTHONPATH is set correctly or run Python from the project root directory.

**Missing Dependencies**

Install all required packages:

.. code-block:: bash

   pip install numpy scipy torch scikit-learn tqdm matplotlib