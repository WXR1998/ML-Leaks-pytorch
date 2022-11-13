# A Naive Implementation of ML-Leaks

A simple implementation of the Adversary 1 on CIFAR-10 and MNIST.

- The backbone model on the two datasets are different. See `nn.py` for details.

## Requirements

Python 3.9

```shell
pip install -r requirements.txt
```

## Functions

See the body of `main.py`.

`experiment()`:
- First, divide the original dataset into 4 parts (`target_train, target_out, shadow_train, shadow_out`), as introduced in the paper.
- Train a target model and a shadow model based on `target_train`, `shadow_train`.
- Inference them on the four datasets and then extract the top-3 posteriors of the outputs.
- Train the attack model on the extracted posteriors and evaluate the results.

## Results

### Quantitative Results

| Dataset | Acc.(%) |
|:-------:|---------|
|  MNIST  | 51.0    |
|  CIFAR  | 75.8    |
