.. _installation:

Installation
============

Requirements
------------

DDMTOLab requires:

* Python 3.10+
* PyTorch 2.5+ with CUDA 12.1 support (for GPU acceleration)
* BoTorch 0.16+
* GPyTorch 1.14+
* NumPy 2.0+
* SciPy 1.15+
* scikit-learn 1.7+
* Pandas 2.3+
* Matplotlib 3.10+
* tqdm

Optional (for documentation):

* Sphinx 7.4+
* sphinx-rtd-theme 3.0+
* myst-parser 3.0+

Installation Methods
--------------------

Method 1: Using Conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create Conda Environment**

.. code-block:: bash

   # Create a new conda environment
   conda create -n ddmtolab python=3.10

   # Activate the environment
   conda activate ddmtolab

**Step 2: Install PyTorch with CUDA Support**

.. code-block:: bash

   # For CUDA 12.1 (GPU acceleration)
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

   # For CPU only
   conda install pytorch cpuonly -c pytorch

**Step 3: Install BoTorch and GPyTorch**

.. code-block:: bash

   conda install botorch -c conda-forge
   conda install gpytorch -c gpytorch

**Step 4: Install Other Dependencies**

.. code-block:: bash

   conda install numpy scipy scikit-learn pandas matplotlib seaborn tqdm -c conda-forge

**Step 5: Clone and Install DDMTOLab**

.. code-block:: bash

   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab

Method 2: Using pip
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab

   # Install dependencies
   pip install -r requirements.txt

Method 3: Install from requirements.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``requirements.txt`` file with the following content:

.. code-block:: text

   torch>=2.5.1
   botorch>=0.16.0
   gpytorch>=1.14.2
   numpy>=2.0.1
   scipy>=1.15.3
   scikit-learn>=1.7.2
   pandas>=2.3.3
   matplotlib>=3.10.6
   seaborn>=0.13.2
   tqdm>=4.67.1
   pyro-ppl>=1.9.1
   jaxtyping>=0.3.2

Then install:

.. code-block:: bash

   pip install -r requirements.txt

Set Python Path
---------------

To use DDMTOLab modules, add the project directory to your Python path.

**Windows (PowerShell):**

.. code-block:: powershell

   # Temporary (current session only)
   $env:PYTHONPATH="D:\DDMTOLab;$env:PYTHONPATH"

   # Permanent (add to PowerShell profile)
   echo '$env:PYTHONPATH="D:\DDMTOLab;$env:PYTHONPATH"' >> $PROFILE

**Windows (Command Prompt):**

.. code-block:: bat

   set PYTHONPATH=D:\DDMTOLab;%PYTHONPATH%

**Linux/Mac:**

.. code-block:: bash

   # Temporary (current session only)
   export PYTHONPATH=/path/to/DDMTOLab:$PYTHONPATH

   # Permanent (add to ~/.bashrc or ~/.zshrc)
   echo 'export PYTHONPATH=/path/to/DDMTOLab:$PYTHONPATH' >> ~/.bashrc
   source ~/.bashrc

**Alternative: Run from Project Root**

You can also run Python from the project root directory without setting PYTHONPATH:

.. code-block:: bash

   cd D:\DDMTOLab
   python your_script.py

Development Installation
------------------------

For development, install in editable mode:

.. code-block:: bash

   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab
   pip install -e .

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/JiangtaoShen/DDMTOLab/issues>`_
2. Review the documentation at `https://jiangtaoshen.github.io/DDMTOLab/ <https://jiangtaoshen.github.io/DDMTOLab/>`_
3. Contact: j.shen5@exeter.ac.uk