# Vivacious Sharpness Aware Minimization

Sharpness-aware minimization fails to converge in the simple convex case. Can we do better?

# Requirements
```
conda create -n py310MSAM python=3.10 -y
conda activate py310MSAM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge tqdm wandb
```

## SAM convergence test on a simple convex function

Make sure you have `python`, `pytorch`, and `matplotlib` installed.

The implementation of SAM is from [Sharpness-Aware Minimization for Efficiently Improving Generalization in Pytorch](https://github.com/davda54/sam).

The [notebook](https://drive.google.com/file/d/1OEDlYUmlv7a53a_V7fqUIrpy0SXvSKBd/view?usp=sharing) compares the convergence of SAM and SGD on a simple convex function `f(x) = x^2`.
