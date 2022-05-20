# Random Fourier Features Pytorch
[![Python package](https://github.com/jmclong/random-fourier-features-pytorch/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmclong/random-fourier-features-pytorch/actions/workflows/python-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/jmclong/random-fourier-features-pytorch/badge.svg)](https://coveralls.io/github/jmclong/random-fourier-features-pytorch)
[![Documentation Status](https://readthedocs.org/projects/random-fourier-features-pytorch/badge/?version=latest)](https://random-fourier-features-pytorch.readthedocs.io/en/latest/?badge=latest)

[![PyPI](https://img.shields.io/pypi/v/random-fourier-features-pytorch.svg?style=plastic&PyPI)](https://pypi.org/project/random-fourier-features-pytorch/)
[![Downloads](https://img.shields.io/pypi/dm/random-fourier-features-pytorch.svg?style=plastic&label=Downloads)](https://pypi.org/project/random-fourier-features-pytorch/)

Random Fourier Features Pytorch is an implementation of "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" by Tancik et al. designed to fit seamlessly into any PyTorch project.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package.

```bash
pip install random-fourier-features-pytorch
```

## Usage
See the [documentation](https://random-fourier-features-pytorch.readthedocs.io/en/latest/) for more details, but here are a few simple usage examples:
### Gaussian Encoding
```python
import torch
import rff

X = torch.randn((256, 256, 2))
encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=256)
Xp = encoding(X)
```
### Basic Encoding
```python
import torch
import rff

X = torch.randn((256, 256, 2))
encoding = rff.layers.BasicEncoding()
Xp = encoding(X)
```
### Positional Encoding
```python
import torch
import rff

X = torch.randn((256, 256, 2))
encoding = rff.layers.PositionalEncoding(sigma=1.0, m=10)
Xp = encoding(X)
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Citation
If you end up using this repository, please cite it as:
```bibtex
@article{long2021rffpytorch,
  title={Random Fourier Features Pytorch},
  author={Joshua M. Long},
  journal={GitHub. Note: https://github.com/jmclong/random-fourier-features-pytorch},
  year={2021}
}
```
also cite the original work
```bibtex
@misc{tancik2020fourier,
      title={Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains}, 
      author={Matthew Tancik and Pratul P. Srinivasan and Ben Mildenhall and Sara Fridovich-Keil and Nithin Raghavan and Utkarsh Singhal and Ravi Ramamoorthi and Jonathan T. Barron and Ren Ng},
      year={2020},
      eprint={2006.10739},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
This is released under the MIT license found in the [LICENSE](LICENSE) file.
