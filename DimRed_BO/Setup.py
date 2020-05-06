"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='DimRed_BO',
    version='1.0.0',
    description='A dimension reduction integrated bayesian optimization algorithm', 
    author='Laura Schultz and Vadim Sokolov',  
    keywords='dimension reduction bayesian optimization',  
    packages=find_packages(),
    python_requires='>=3.6',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy',
    'scipy>=0.15.0',
    'matplotlib',
    'scikit-learn,
    'torch>=1.1',
    'torchvision',
    'gpytorch>=0.3.4',
    'PyTorch>= 1.1',
    'botorch'],  # Optional
    dependency_links=[
        "torch===1.3.0 torchvision===0.4.1 -f https://download.pytorch.org/whl/torch_stable.html"
        
    ]
)
""" 
pip install numpy scipy matplotlib scikit-learn
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/paulcon/active_subspaces.git
cd active_subspaces
git fetch https://github.com/paulcon/active_subspaces.git refs/pull/52/head
py setup.py install
conda install botorch -c pytorch -c gpytorch

 """