# Impact Workshop 2025

## Introduction

This repository contains source of training material for the `Impact Workshop 2025`. The topics covered are:
- Recap of Nonlinear Programming, Optimal Control and Model Predictive Control
- Tutorial 1: Using Impact to rapidly specify, prototype and deploy model predictive controllers
- Tutorial 2: Neural Network-MPC

## Development environment

Should work for Windows, Linux and Mac.

### Clone/download this repository

### Install Conda

If you have not installed Conda, you should first install Miniconda. Miniconda is a minimal installer for Conda. You can install Miniconda by following these steps:

#### For Linux

1. Download the Miniconda installer for Linux from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
2. Open a terminal.
3. Navigate to the directory where you downloaded the installer.
4. Run the installer with the following command, replacing `Miniconda3-latest-Linux-x86_64.sh` with the name of the file you downloaded:
   ```sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
5. Follow the on-screen instructions to complete the installation.
6. Close and reopen your terminal to apply the changes.

#### For Windows

For Windows users, follow these steps to install Miniconda:

1. Download the Miniconda installer for Windows from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
2. Double-click the downloaded file to start the installation.
3. Follow the on-screen instructions, making sure to check the option to "Add Miniconda to my PATH environment variable" for an easier use of Conda from the command prompt.
4. After installation, open the Command Prompt or Anaconda Prompt from the Start menu to start using Conda.

### Creating Conda environment

### Figure out a desired PYTHON_VERSION
 * If you don't plan on using Matlab, pick PYTHON_VERSION=3.12
 * If you do plan on using Matlab, choose the highest version that is listed for your Matlab version at https://nl.mathworks.com/support/requirements/python-compatibility.html

### Open a Conda shell
  ⚠️ On Windows, pick "Anaconda Prompt", not "Anacoda PowerShell Prompt"

### Create the Conda environment for the workshop

```
conda create --name workshop_dirac python=<PYTHON_VERSION> -y --channel=defaults --override-channels
conda activate workshop_dirac
(Only Windows) $ conda install -y --channel conda-forge cmake clang=15 lld=15 llvmdev=15 ninja 
(Only linux and Mac) $ conda install -y --channel conda-forge cmake clang lld llvmdev ninja

pip install -r requirements.txt
```

## Verify the environment

4. Run `python test.py`
 
  You'll be queried "Do you wish to set up Tera renderer automatically?" Answer 'y'.

  This file should run succesfully.

  Getting the error "CMake was unable to find a build program corresponding to Ninja" or "No CMAKE_RC_COMPILER could be found"?
  That means that you opened the wrong shell on Windows. Please see 2.




<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

