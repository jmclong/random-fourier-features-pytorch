from setuptools import setup
setup(name='random-fourier-features-pytorch',
      version='0.2',
      description='Random Fourier Features for PyTorch',
      url='https://github.com/jmclong/random-fourier-features-pytorch',
      author='Joshua Long',
      author_email='joshualong@live.com',
      license='MIT',
      packages=['rff'],
      install_requires=["numpy>=1.17",
                        "torch>=1.7",
                        "torchvision>=0.9"])
