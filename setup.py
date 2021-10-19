from setuptools import setup, find_packages
from os.path import join, dirname
import neftpy

setup(
    name='neftpy',
    version=neftpy.__version__,
    description='Petroluem production engineering calculations package. Inspired by Unifloc VBA.',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    url='https://github.com/unifloc/neftpy',
    author='Rinat Khabibullin',
    author_email='khabibullinra@gmail.com',
    license='BSD 3-Clause License',
    install_requires=[
          'numpy', 'scipy',
      ],
)