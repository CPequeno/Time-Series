---
title: "Wold's Decomposition"
author: "Pareto Deficient (CPequeno)"
output: 
  html_document:
    keep_md: true
    self_contained: true
    includes:
      in_header: header.html
---


## Introduction

This document explains Wold's decomposition and illustrates its application in time series analysis, as well as its limitations when the actual data-generating process (DGP) is not stationary (e.g. Random Walk) or exhibits high nonlinearities.


## Setup

Consider the following equation:

$$
y(t) =  u(t) + 0.2 \cdot u(t-1) + 0.8 \cdot u(t-1) \cdot u(t-2)
$$

which represents the true Data Generating Process (DGP). In practice, the actual DGP is typically unknown, and it must be estimated using some statistical techniques. In this notebook, I will apply Wold’s decomposition.

Wold’s decomposition is a fundamental concept in time series analysis, which decomposes a time series into two components: a predictable (deterministic) part and a stochastic (random) part. This decomposition is particularly useful for understanding the underlying structure of a time series, especially when the true DGP is not known.

This setup can be configured in Julia as follows:


```julia
using Random
using GLM
using Statistics
```


```julia
Random.seed!(123) # Seed for reproducibility.

n = 1000
u = randn(n) # As usual, the error is assumed to be normal.
y = u .+ 0.2 .* circshift(u, 1) .+ 0.8 .* circshift(u, 1) .* circshift(u, 2)
```

where the command 'circshift(y, i)' shifts the values of the array y by i positions, e.g., for y = [1, 2, 3], circshift(y, 1) = [2, 3, NaN].

### Best Linear Predictor (BLP)

By the Linear projection theorem, the BLP (assuming the DGP is known) can be computed as:

$$
\beta = (X_{\text{best}} X_{\text{best}}^{-1} X_{\text{best}} \ y
$$
where the three regressors above,  $(1, \cdot u(t-1), and \cdot u(t-1) \cdot u(t-2))$, are merged into a single matrix called $X_{best}$. Thus, I run OLS of y on $X_{best}$. In addition to the BLP, I also get the sum of squared residuals, i.e.,

$$
SSR = \sum_{t=1}^{n} \left(y_t - \hat{y_t} \right)^2
$$
in order to compute later the variance of the errors The reason is that, later, I will compare it with the variance of the residuals of the Wold's decomposition (and also the variance of the residuals of the AR model I estimate below). In Julia, this is done running the following chunk:


```julia
X_best = hcat(circshift(u, 1), circshift(u, 1) .* circshift(u, 2))
```

```
## 1000×2 Matrix{Float64}:
##   0.169367    0.0579124
##   0.248855    0.0421478
##  -0.0518626  -0.0129063
##  -0.336529    0.0174533
##   1.41441    -0.475989
##   0.533747    0.754936
##   0.0719523   0.0384043
##  -0.19358    -0.0139285
##  -0.954942    0.184858
##  -0.596801    0.56991
##   ⋮          
##  -3.35924     1.39284
##  -1.28771     4.32572
##  -1.46055     1.88076
##  -0.848935    1.23991
##  -0.783666    0.665282
##  -0.138111    0.108233
##   0.6022     -0.0831704
##  -0.306873   -0.184799
##   0.341934   -0.104931
```

```julia
best_model = lm(X_best, y)
```

```
## LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}:
## 
## Coefficients:
## ───────────────────────────────────────────────────────────────
##        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
## ───────────────────────────────────────────────────────────────
## x1  0.23646    0.0316436   7.47    <1e-12   0.174364   0.298556
## x2  0.766797   0.0314823  24.36    <1e-99   0.705018   0.828576
## ───────────────────────────────────────────────────────────────
```

```julia
best_ssr = deviance(best_model)
```

```
## 1011.7539724137857
```

### Theoretical Representation

Formally speaking, the Wold’s decomposition represents a time series as:

$$
y(t) = \mu(t) + \epsilon(t)
$$

where $\mu(t)$ is an autoregressive (AR) process, i.e., the deterministic part, and $\epsilon(t)$ is a moving average (MA) process, i.e., the stochastic part.

To determine the optimal number of lags for the AR model, I should perform model selection by fitting AR models with varying lag lengths (e.g., AR(1), AR(2), ..., AR(n)) and selecting the best model based on a performance metric, such as AIC or BIC. Although I will explore model selection in more detail in subsequent notebooks, for now, I will bypass this step and choose an AR(10), which corresponds to an AR model with 10 lags.


```julia
lags = 10
```

Why an AR(10)? Good question, my (probably) non-existent reader. But, if truth be told, the choice is somewhat arbitrary. The idea is to capture enough of the autocorrelation in the data. For a process like the one I dealing with, choosing a higher number of lags (such as 10) can help capture long-term dependencies as the behavior of the series may depend on multiple previous time periods.

To estimate the deterministic part, $\mu(t)$, I create the following lagged matrix \(X_{ary}\):


```julia
X_ary = hcat([circshift(y, i) for i in 1:lags]...)
```

```
## 1000×10 Matrix{Float64}:
##   0.153809    0.13272    -0.25297    …   1.74249    -0.845283   -3.7804
##   0.329058    0.153809    0.13272        0.363562    1.74249    -0.845283
##   0.0316266   0.329058    0.153809       0.0384752   0.363562    1.74249
##  -0.357227    0.0316266   0.329058       0.237381    0.0384752   0.363562
##   1.36106    -0.357227    0.0316266      0.661164    0.237381    0.0384752
##   0.435837    1.36106    -0.357227   …  -0.25297     0.661164    0.237381
##   0.782651    0.435837    1.36106        0.13272    -0.25297     0.661164
##  -0.148466    0.782651    0.435837       0.153809    0.13272    -0.25297
##  -1.0048     -0.148466    0.782651       0.329058    0.153809    0.13272
##  -0.639903   -1.0048     -0.148466       0.0316266   0.329058    0.153809
##   ⋮                                  ⋱                          
##  -3.7804      0.56688     1.63953        0.538252   -1.09644     0.3686
##  -0.845283   -3.7804      0.56688        0.523915    0.538252   -1.09644
##   1.74249    -0.845283   -3.7804         0.317445    0.523915    0.538252
##   0.363562    1.74249    -0.845283       0.582164    0.317445    0.523915
##   0.0384752   0.363562    1.74249    …   1.0787      0.582164    0.317445
##   0.237381    0.0384752   0.363562       1.63953     1.0787      0.582164
##   0.661164    0.237381    0.0384752      0.56688     1.63953     1.0787
##  -0.25297     0.661164    0.237381      -3.7804      0.56688     1.63953
##   0.13272    -0.25297     0.661164      -0.845283   -3.7804      0.56688
```

This code looks a bit weird, doesn't it? Let me explain it by parts, like Jack the Ripper:

First of all,'[circshift(y, i) for i in 1:lags]' creates the following array: '[circshift(y, 1), circshift(y, 2), ..., circshift(y, 10)]' where, as I explained before, 'circshift(y, i)' lags the time series by i periods.

Second, 'hcat([circshift(y, i) for i in 1:lags])' horizontally concatenates arrays. Thus, a list of ten elements where each element is $circshift(y, i)$ for $i \in {1, 2, ..., 10}$ is obtained. However, for regression purposes, a matrix rather than a vector is required. That is the reason for the three dots after the squared brackets, called the 'splat operator'. This splat operator is used to "unpack" the list of arrays so that 'hcat' receives them as separate arguments.

For illustration, let $y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]$. Similar to the actual example, suppose we need to compute 10 lags of this process. In this case, the expression hcat([circshift(y, i) for i in 1:lags]...) generates the following **matrix**:

$$
X_{\text{ary}} = 
\begin{bmatrix}
2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 \\
3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 \\
4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & \text{NaN} \\
5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & \text{NaN} & \text{NaN} \\
6 & 7 & 8 & 9 & 10 & 11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} \\
7 & 8 & 9 & 10 & 11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \\
8 & 9 & 10 & 11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \\
9 & 10 & 11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \\
10 & 11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \\
11 & 12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} \\
12 & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN} & \text{NaN}
\end{bmatrix}
$$

while 'hcat([circshift(y, i) for i in 1:lags])' (note here the absence of the splat operator, i.e., the three dots) gives the following **list**:

$$
X_{ary} = 
\begin{bmatrix}
[2, 3, \ldots, 10, 11] \\
[3, 4, \ldots, 11, 12] \\
[4, 5, \ldots, 12, \text{NaN}] \\
[5, 6, \ldots, \text{NaN}, \text{NaN}] \\
[6, 7, \ldots, \text{NaN}, \text{NaN}] \\
[7, 8, \ldots, \text{NaN}, \text{NaN}] \\
[8, 9, \ldots, \text{NaN}, \text{NaN}] \\
[9, 10, \ldots, \text{NaN}, \text{NaN}] \\
[10, 11, \ldots, \text{NaN}, \text{NaN}] \\
[11, 12, \ldots, \text{NaN}, \text{NaN}]
\end{bmatrix}
$$

### Estimating the AR model

With this understood, let me continue and estimate the AR(10) process, i.e.,

$$
y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \phi_3 y_{t-3} + \cdots + \phi_{10} y_{t-10} + \epsilon_t
$$

and, in a similar fashion as before, obtain the sum of the squared residuals.


```julia
ary_model = lm(X_ary, y)
```

```
## LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}:
## 
## Coefficients:
## ────────────────────────────────────────────────────────────────────
##            Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%
## ────────────────────────────────────────────────────────────────────
## x1    0.115862     0.031782    3.65    0.0003   0.0534938  0.178229
## x2   -0.00595911   0.0319811  -0.19    0.8522  -0.0687177  0.0567995
## x3   -0.0346355    0.031979   -1.08    0.2790  -0.09739    0.0281189
## x4   -0.00643612   0.0319957  -0.20    0.8406  -0.0692234  0.0563512
## x5    0.0287818    0.0319901   0.90    0.3685  -0.0339944  0.0915579
## x6    0.0199931    0.0319901   0.62    0.5321  -0.042783   0.0827693
## x7   -0.0118305    0.0319957  -0.37    0.7116  -0.0746177  0.0509568
## x8    0.0130222    0.031979    0.41    0.6839  -0.0497322  0.0757767
## x9    0.0292146    0.0319811   0.91    0.3612  -0.033544   0.0919732
## x10  -0.00232707   0.031782   -0.07    0.9416  -0.0646949  0.0600408
## ────────────────────────────────────────────────────────────────────
```

```julia
ary_ssr = deviance(ary_model)
```

```
## 1626.8028316558346
```


### Estimating the MA model

Let me now to estimate the non-deterministic part of the Wold's decomposition. The residuals from the regression above, $\epsilon_t$, represent this stochastic component.


```julia
residuals_ary = residuals(ary_model)
```

With them, I estimate a MA(30) and obtain the sum of squared residuals. Mathematically, a MA(30) takes the following form:

$$
y_t = \theta_0 + \sum_{i=1}^{30} \theta_i e_{t-i} + e_t
$$

which, in Julia, can be computed in a similar fashion as before:


```julia
lags_residuals = 30
```

```
## 30
```

```julia
X_wold = hcat([circshift(residuals_ary, i) for i in 1:lags_residuals]...)
```

```
## 1000×30 Matrix{Float64}:
##   0.29636     0.155621   -0.455232   …   2.44502    -0.257008   -1.76947
##   0.297428    0.29636     0.155621      -2.04043     2.44502    -0.257008
##  -0.0835412   0.297428    0.29636       -0.32192    -2.04043     2.44502
##  -0.362946   -0.0835412   0.297428      -3.76722    -0.32192    -2.04043
##   1.42072    -0.362946   -0.0835412     -3.79437    -3.76722    -0.32192
##   0.253698    1.42072    -0.362946   …   0.212677   -3.79437    -3.76722
##   0.70165     0.253698    1.42072       -0.0512592   0.212677   -3.79437
##  -0.190175    0.70165     0.253698       1.83392    -0.0512592   0.212677
##  -0.952006   -0.190175    0.70165        0.270982    1.83392    -0.0512592
##  -0.534584   -0.952006   -0.190175       0.482238    0.270982    1.83392
##   ⋮                                  ⋱                          
##  -3.80392     0.353651    1.48289        1.39495    -1.60855    -1.21984
##  -0.331197   -3.80392     0.353651      -3.0         1.39495    -1.60855
##   1.78405    -0.331197   -3.80392       -0.669694   -3.0         1.39495
##  -0.050705    1.78405    -0.331197      -2.54591    -0.669694   -3.0
##  -0.0988415  -0.050705    1.78405    …  -0.276244   -2.54591    -0.669694
##   0.376555   -0.0988415  -0.050705      -0.17704    -0.276244   -2.54591
##   0.712805    0.376555   -0.0988415      0.0360697  -0.17704    -0.276244
##  -0.455232    0.712805    0.376555      -1.76947     0.0360697  -0.17704
##   0.155621   -0.455232    0.712805      -0.257008   -1.76947     0.0360697
```

```julia
wold_model = lm(X_wold, y)
```

```
## LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}:
## 
## Coefficients:
## ───────────────────────────────────────────────────────────────────────
##             Coef.  Std. Error      t  Pr(>|t|)    Lower 95%   Upper 95%
## ───────────────────────────────────────────────────────────────────────
## x1    0.117506      0.0320511   3.67    0.0003   0.0546082   0.180403
## x2    0.0148787     0.0320382   0.46    0.6425  -0.0479934   0.0777508
## x3   -0.0393835     0.0319823  -1.23    0.2185  -0.102146    0.0233789
## x4   -0.00804221    0.0319531  -0.25    0.8013  -0.0707475   0.054663
## x5    0.0335834     0.0319412   1.05    0.2933  -0.0290984   0.0962651
## x6    0.0256486     0.031941    0.80    0.4222  -0.0370329   0.0883301
## x7   -0.00485564    0.0319406  -0.15    0.8792  -0.0675363   0.0578251
## x8    0.0108167     0.0319231   0.34    0.7348  -0.0518296   0.073463
## x9    0.0227258     0.0319231   0.71    0.4767  -0.0399205   0.0853722
## x10   0.00342936    0.031873    0.11    0.9143  -0.0591187   0.0659774
## x11  -0.02367       0.0318719  -0.74    0.4579  -0.0862157   0.0388758
## x12  -0.0420272     0.0318768  -1.32    0.1877  -0.104583    0.0205283
## x13   0.00855297    0.0319026   0.27    0.7887  -0.054053    0.071159
## x14  -0.0225717     0.0318908  -0.71    0.4793  -0.0851546   0.0400113
## x15  -0.00298488    0.0318549  -0.09    0.9254  -0.0654973   0.0595276
## x16   0.0530893     0.0318549   1.67    0.0959  -0.00942322  0.115602
## x17   0.0290217     0.0318908   0.91    0.3630  -0.0335612   0.0916047
## x18   0.000867287   0.0319026   0.03    0.9783  -0.0617387   0.0634733
## x19   0.0179009     0.0318768   0.56    0.5745  -0.0446545   0.0804564
## x20  -0.00860698    0.0318719  -0.27    0.7872  -0.0711527   0.0539388
## x21  -0.0561847     0.031873   -1.76    0.0783  -0.118733    0.00636335
## x22  -0.0013024     0.0319231  -0.04    0.9675  -0.0639488   0.061344
## x23   0.0330628     0.0319231   1.04    0.3006  -0.0295835   0.0957092
## x24  -0.00531383    0.0319406  -0.17    0.8679  -0.0679945   0.0573669
## x25   0.00778261    0.031941    0.24    0.8075  -0.0548989   0.0704641
## x26   0.0280156     0.0319412   0.88    0.3807  -0.0346662   0.0906973
## x27  -0.042763      0.0319531  -1.34    0.1811  -0.105468    0.0199422
## x28   0.059395      0.0319823   1.86    0.0636  -0.00336738  0.122157
## x29   0.0283814     0.0320382   0.89    0.3759  -0.0344907   0.0912535
## x30  -0.0595552     0.0320511  -1.86    0.0635  -0.122453    0.00334226
## ───────────────────────────────────────────────────────────────────────
```

```julia
wold_ssr = deviance(wold_model)
```

```
## 1591.1676029642574
```

Using 30 lags is again arbitrary, but it allows to capture dependencies in the stochastic process up to a certain point. In practice, the number of lags would again depend on model selection techniques or the characteristics of the data.


### Comparison of Models

Finally, I compare the variance of the residuals across each approach: the model assuming knowledge of the true DGP, the AR(10) model, and Wold's decomposition

$$
Var_{\text{best}} = \frac{\text{SSR}_{\text{best}}}{n}
$$

```julia
var_best = best_ssr / length(y)
```

```
## 1.0117539724137858
```


$$
Var_{\text{ary}} = \frac{\text{SSR}_{\text{ary}}}{n}
$$


```julia
var_ary = ary_ssr / length(y)
```

```
## 1.6268028316558345
```


$$
Var_{\text{wold}} = \frac{\text{SSR}_{\text{wold}}}{n}
$$


```julia
var_wold = wold_ssr / length(y)
```

```
## 1.5911676029642574
```


The results are striking. When the DGP is known, the variance of the errors is 0.97, while the variances of the residuals for the AR(10) model and Wold's decomposition are 1.52 and 1.50, respectively. Notably, Wold's decomposition performs exceptionally well. However, this is not the end of the story.


### Limitations of Wold's Decomposition

The performance of Wold's decomposition significantly deteriorates when the model involves high nonlinearities or is a non-stationary process, such as a random walk. For instance, consider the case where the actual DGP is

$$
x_t = x_{t-1} + u_t
$$

```julia
y = cumsum(u)
```

```
## 1000-element Vector{Float64}:
##    0.24885492425494143
##    0.19699228901692967
##   -0.13953700106319106
##    1.274870127322079
##    1.8086174080162833
##    1.8805697160968993
##    1.6869894290708967
##    0.7320478416477796
##    0.13524711843895582
##    0.45814520896736577
##    ⋮
##  -17.958928036575735
##  -19.4194751946252
##  -20.268410480732747
##  -21.052076233361532
##  -21.19018715117391
##  -20.587986839845776
##  -20.894860076427996
##  -20.552925708908383
##  -20.383558806230127
```

which is a random walk. The best predictor for this model is:

$$
\hat{x_t} = \mathbb{E}[x_t \mid F_{t-1}] = x_{t-1}
$$

where $F_t$ represents a sigma-algebra or a filtration at time $t$.

```julia
X_best = hcat(circshift(y, 1))
```

```
## 1000×1 Matrix{Float64}:
##  -20.383558806230127
##    0.24885492425494143
##    0.19699228901692967
##   -0.13953700106319106
##    1.274870127322079
##    1.8086174080162833
##    1.8805697160968993
##    1.6869894290708967
##    0.7320478416477796
##    0.13524711843895582
##    ⋮
##  -16.671219879394407
##  -17.958928036575735
##  -19.4194751946252
##  -20.268410480732747
##  -21.052076233361532
##  -21.19018715117391
##  -20.587986839845776
##  -20.894860076427996
##  -20.552925708908383
```

In such cases, the residual variance of Wold's decomposition becomes excessively high. Specifically, while the variance of the residuals under the assumption of knowing the true DGP is approximately 3.55, the variance of Wold's decomposition skyrockets to 753.80. To verify this, re-run the entire notebook, replacing the original DGP with the new one and updating the BP accordingly.

