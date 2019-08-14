from setuptools import setup, find_packages

setup(
   name='SensitivityAnalyser',
   version='1.0',
   description='Functions for performing sensitivity analysis of simulation models',
   author='Michael Pitcher',
   author_email='mjp22@st-andrews.ac.uk',
   packages=find_packages(),
   install_requires=['epyc', 'numpy', 'scipy', 'sklearn', 'pandas', 'matplotlib'],
)