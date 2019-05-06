Learned Cardinalities in PyTorch
====

PyTorch implementation of multi-set convolutional networks (MSCNs) to estimate the result sizes of SQL queries [1, 2].

## Requirements

  * PyTorch 1.0
  * Python 3.7

## Usage

```python3 train.py --help```

Example usage:

```python3 train.py synthetic```

To reproduce the results in [1] use:

```python3 train.py --queries 100000 --epochs 100 synthetic```

```python3 train.py --queries 100000 --epochs 100 scale```

```python3 train.py --queries 100000 --epochs 100 job-light```

## References

[1] [Kipf et al., Learned Cardinalities: Estimating Correlated Joins with Deep Learning, 2018](https://arxiv.org/abs/1809.00677)

[2] [Kipf et al., Estimating Cardinalities with Deep Sketches, 2019](https://arxiv.org/abs/1904.08223)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2018learned,
  title={Learned cardinalities: Estimating correlated joins with deep learning},
  author={Kipf, Andreas and Kipf, Thomas and Radke, Bernhard and Leis, Viktor and Boncz, Peter and Kemper, Alfons},
  journal={arXiv preprint arXiv:1809.00677},
  year={2018}
}
```
