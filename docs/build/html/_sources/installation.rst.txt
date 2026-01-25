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
* Seaborn 0.13+
* tqdm 4.67+
* OpenPyXL 3.1+
* PyYAML 6.0+
* Bottleneck 1.4+
* Pyro-PPL 1.9+

Optional (for documentation):

* Sphinx 7.4+
* sphinx-rtd-theme 3.0+
* myst-parser 3.0+

Installation Methods
--------------------

Method 1: Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quick Installation**

The easiest way to install DDMTOLab is directly from PyPI:

.. code-block:: bash

   pip install ddmtolab

This will automatically install all required dependencies except PyTorch. You need to install PyTorch separately based on your system configuration.

**Complete Installation with PyTorch**

.. code-block:: bash

   # For CUDA 12.1 (GPU acceleration)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install ddmtolab

   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install ddmtolab

**Installation in Virtual Environment (Recommended)**

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv ddmtolab_env

   # Activate on Windows
   ddmtolab_env\Scripts\activate

   # Activate on Linux/Mac
   source ddmtolab_env/bin/activate

   # Install PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install DDMTOLab
   pip install ddmtolab

Method 2: Using Conda
~~~~~~~~~~~~~~~~~~~~~

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

**Step 3: Install DDMTOLab**

.. code-block:: bash

   pip install ddmtolab

**Alternative: Install All Dependencies via Conda**

.. code-block:: bash

   # Install PyTorch
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

   # Install BoTorch and GPyTorch
   conda install botorch -c conda-forge
   conda install gpytorch -c gpytorch

   # Install other dependencies
   conda install numpy scipy scikit-learn pandas matplotlib seaborn tqdm openpyxl pyyaml bottleneck -c conda-forge
   conda install pyro-ppl -c conda-forge

   # Install DDMTOLab
   pip install ddmtolab

Method 3: Install from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the latest development version or if you want to contribute:

**Step 1: Clone the Repository**

.. code-block:: bash

   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab

**Step 2: Create Virtual Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv ddmtolab_env

   # Activate on Windows
   ddmtolab_env\Scripts\activate

   # Activate on Linux/Mac
   source ddmtolab_env/bin/activate

**Step 3: Install PyTorch**

.. code-block:: bash

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

**Step 4: Install DDMTOLab and Dependencies**

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Development Installation
------------------------

For development work, install DDMTOLab in editable mode. This allows you to modify the code and see changes immediately without reinstalling.

**Step 1: Clone the Repository**

.. code-block:: bash

   git clone https://github.com/JiangtaoShen/DDMTOLab.git
   cd DDMTOLab

**Step 2: Create Virtual Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv ddmtolab_dev

   # Activate on Windows
   ddmtolab_dev\Scripts\activate

   # Activate on Linux/Mac
   source ddmtolab_dev/bin/activate

**Step 3: Install PyTorch**

.. code-block:: bash

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

**Step 4: Install Dependencies**

.. code-block:: bash

   pip install -r requirements.txt

**Step 5: Install in Editable Mode**

.. code-block:: bash

   pip install -e .

This installs DDMTOLab in development mode, allowing you to:

* Modify source code and see changes immediately
* Run tests and contribute to development
* Use the package from any directory

**Step 6: Install Development Tools (Optional)**

.. code-block:: bash

   # Install testing tools
   pip install pytest pytest-cov

   # Install documentation tools
   pip install sphinx sphinx-rtd-theme myst-parser

   # Install code quality tools
   pip install black flake8 mypy

Verify Installation
-------------------

To verify that DDMTOLab and all dependencies are correctly installed:

.. code-block:: python

   # Test basic imports
   import numpy as np
   import pandas as pd
   import torch
   import botorch
   import gpytorch
   import ddmtolab

   # Check versions
   print(f"DDMTOLab version: {ddmtolab.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"BoTorch version: {botorch.__version__}")
   print(f"GPyTorch version: {gpytorch.__version__}")

   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"Number of GPUs: {torch.cuda.device_count()}")
       print(f"GPU name: {torch.cuda.get_device_name(0)}")

**Quick Test**

Run a quick test to ensure everything works:

.. code-block:: python

   from ddmtolab.bayesian_optimization import BayesianOptimizer
   from ddmtolab.gaussian_process import GaussianProcess

   print("DDMTOLab installation successful!")

Upgrading DDMTOLab
------------------

To upgrade to the latest version:

.. code-block:: bash

   # Upgrade from PyPI
   pip install --upgrade ddmtolab

   # Upgrade from source
   cd DDMTOLab
   git pull
   pip install --upgrade -e .

Uninstalling DDMTOLab
---------------------

To uninstall DDMTOLab:

.. code-block:: bash

   pip uninstall ddmtolab

Troubleshooting
---------------

**Common Issues:**

1. **PyTorch not found:**

   Install PyTorch first before installing DDMTOLab:

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      pip install ddmtolab

2. **CUDA compatibility issues:**

   Ensure your CUDA version matches PyTorch requirements. Check compatibility at:
   https://pytorch.org/get-started/locally/

3. **Import errors:**

   If you installed from source, make sure you ran ``pip install -e .`` in the project directory.

4. **Dependency conflicts:**

   Use a fresh virtual environment or conda environment to avoid conflicts:

   .. code-block:: bash

      python -m venv fresh_env
      source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
      pip install torch --index-url https://download.pytorch.org/whl/cu121
      pip install ddmtolab

5. **BoTorch/GPyTorch installation fails:**

   Install PyTorch first, then install DDMTOLab which will handle BoTorch and GPyTorch dependencies.

6. **Version conflicts:**

   Check installed versions:

   .. code-block:: bash

      pip list | grep -E "torch|botorch|gpytorch|ddmtolab"

   Reinstall with specific versions if needed:

   .. code-block:: bash

      pip install torch==2.5.1 botorch==0.16.0 gpytorch==1.14.2
      pip install ddmtolab

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/JiangtaoShen/DDMTOLab/issues>`_
2. Review the documentation at `https://jiangtaoshen.github.io/DDMTOLab/ <https://jiangtaoshen.github.io/DDMTOLab/>`_
3. Submit a bug report on GitHub with:

   * Your operating system and Python version
   * Complete error message
   * Installation method used
   * Output of ``pip list``

4. Contact: j.shen5@exeter.ac.uk