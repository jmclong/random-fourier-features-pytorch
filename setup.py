from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='random-fourier-features-pytorch',
      version='1.0.0',
      description='Random Fourier Features for PyTorch',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jmclong/random-fourier-features-pytorch',
      author='Joshua Long',
      author_email='joshualong@live.com',
      license='MIT',
      packages=['rff'],
      install_requires=["numpy>=1.17",
                        "torch>=1.7",
                        "torchvision>=0.9"])
