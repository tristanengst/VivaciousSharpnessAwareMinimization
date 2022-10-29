# Vivacious Sharpness Aware Minimization

Sharpness-aware minimization fails to converge in the simple convex case. Can we do better?

## SAM convergence test on a simple convex function

Make sure you have `python`, `pytorch`, and `matplotlib` installed.

The implementation of SAM is from [Sharpness-Aware Minimization for Efficiently Improving Generalization in Pytorch](https://github.com/davda54/sam).

The notebook compares the convergence of SAM and SGD on a simple convex function `f(x) = x^2`.
