# Impact Workshop 2025

## Introduction

This repository contains source of training material for the `Impact Workshop 2025`. The topics covered are:
- Recap of Nonlinear Programming, Optimal Control and Model Predictive Control
- Tutorial 1: Using Impact to rapidly specify, prototype and deploy model predictive controllers
- Tutorial 2: Neural Network-MPC

## Development environment

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

Create Conda environment with *CasADi 3.6.5* + *Rockit* + *Impact* using the following command:

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

