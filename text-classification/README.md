# Text classification example app


## Python version

Due to its use of Hugging Face transformers (with PyTorch), this app requires
Python 3.7-3.10.  If necessary to explicitly control your Python version, we
recommend using an environment manager such as conda.  For example:
```bash
conda create --name my_env python=3.10   # or any Python version >= 3.7
conda activate my_env
```


## Setup

1. To install the required Python dependencies, run the following at the command
line:
```bash
source pip_install.sh
```
(As with any Python application, we recommend running within a `venv` or `conda`
environment to manage your dependencies.)

2. To populate a local standin for a production database of product reviews, run
the following at the command line:
```bash
python setup_external_data.py
```


## Running the app

To run the app, execute the following at the command line:
```bash
inductor up app
```

Then, open the URL printed in the console, in your browser.
