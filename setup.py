from setuptools import setup
setup(name='random-fourier-features-pytorch',
      version='0.1',
      description='Random Fourier Features for PyTorch',
      url='https://github.com/jmclong/random-fourier-features-pytorch',
      author='Your Name',
      author_email='joshualong@live.com',
      license='BSD-3',
      packages=['rff'],
      install_requires=["numpy>=1.17",
                        "torch>=1.7",
                        "torchvision>=0.9"])
