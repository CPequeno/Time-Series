# Gauss-Newton and Hannan-Rissanen Methods for MA Processes
### Pareto Deficient (CPequeno)

***

<br>

## Introduction

Consider the following MA(1) process:

$$
Y_t = \mu + \epsilon_t + \theta \epsilon_{t-1}
$$

where $|\theta| < 1$ and $\{\epsilon_t\} \sim \text{iid}(0, \sigma^2)$.

As discussed in previous notebooks, it is not possible to directly estimate a MA(q) process because the actual error terms are unobserved. In contrast, for an AR(p), the observed data, X provides the necessary inputs for estimation. In this notebook, I will present two alternative methods for estimating the coefficients of an MA(q) process: Maximum Likelihood Estimation (MLE) and Nonlinear Least Squares (NLLS). Additionally, I will introduce two practical algorithms to implement these estimation techniques.

## Theoretical derivation

### Autocovariance and Autocorrelation functions

In order to derive the theta, we need to state in function of some parameters we can estimate. In particular, we need to derive the autocorrelation of order 1 (because the model is of order 1, otherwise we need to compute as correlation as parameters). For this task, we need first to compute the autocovariance of order 0, i.e., $\text{Cov}(Y_t, Y_t) = \text{Var}(Y_t)$.

Substituting for $Y_t$, we get the following expression:

$$
\text{Var}(Y_t) = \text{Var}(\epsilon_t) + \theta^2 \text{Var}(\epsilon_{t-1}) = \sigma^2 + \theta^2 \sigma^2 = (1 + \theta^2) \sigma^2.
$$

Similarly, the expression for the autocovariance of order 1 is defined as:

$$
\text{Cov}(Y_t, Y_{t-1}).
$$

Substituting for $Y_t$ and $Y_{t-1}$, we get the following expression:

$$
\text{Cov}(Y_t, Y_{t-1}) = \text{Cov}(\epsilon_t + \theta \epsilon_{t-1}, \epsilon_{t-1} + \theta \epsilon_{t-2}).
$$

and since $\epsilon_t$ is independent and identically distributed:

$$
\text{Cov}(Y_t, Y_{t-1}) = \theta \text{Var}(\epsilon_{t-1}) = \theta \sigma^2.
$$

The general expression for the autocorrelation parameter of order 1 is

$$
\rho_1 = \frac{\text{Cov}(Y_t, Y_{t-1})}{\text{Var}(Y_t)}.
$$

but, in particular for a MA(1) model, the expression is the following:

$$
\rho_1 = \frac{\theta \sigma^2}{(1 + \theta^2) \sigma^2} = \frac{\theta}{1 + \theta^2}.
$$

As can be seen, this expression gives us the correlation in terms of $\theta$ The problem is that we do not know the actual autocorrelation of order 1. However, as usual, we can estimate it using the sample autocorrelation:

$$
\hat{\rho}_1 = \frac{\sum_{t=2}^n (Y_t - \bar{Y})(Y_{t-1} - \bar{Y})}{\sum_{t=1}^n (Y_t - \bar{Y})^2}
$$


where $\bar{Y}$ is the sample mean of the $Y_t$ series.

Finally, solving for $\hat{\theta}$ we obtain two possible solutions (because this is a quadratic function):

$$
\hat{\theta}_+ = \frac{1+ \sqrt{1-4\hat{\rho_1}^2}}{2\hat{\rho_1}} \quad \text{and} \quad \hat{\theta}_- = \frac{1- \sqrt{1-4\hat{\rho_1}^2}}{2\hat{\rho_1}}
$$


In addition, if we impose the stationarity assumption, i.e., $|\theta| < 1$, then $-\frac{1}{2} < \rho_1 < \frac{1}{2}$ or $0 < |\rho_1| < \frac{1}{2}$. Given these restrictions, only \hat{\theta}_- is a solution. However, we need to deal with extreme cases. The quadratic equation above assumes $1-4\rho_1^2 \geq 0$, otherwise the solution is imaginary, and in economics we only deal with real things, like invisible hands and so on. Thus, it is required to impose $\theta = 1$ if $\rho_1 > .5$ and $\theta = -1$ if $\rho_1 < .5$ to avoid this problem. Finally, if $\rho_1 = 0$, then $\theta = 0$.

Now, let's simulate a particular MA(1). As always, these are the required packages: 


```julia
using Random, LinearAlgebra, Statistics, Plots, DataFrames, GLM
```

I define a function to simulate an MA(q) process. Above I show the derivations for an MA(1) process because they are easy, but we can do it for any MA(q) if we are patient enough. Anyway, the function shown below generates such a process and returns the vector of endogenous variables and the regressors, which are the errors in an MA(q) process.


```julia
# Generate synthetic data for MA(q)
function generate_ma(q, θ, n, μ)
Random.seed!(123)
    ε = randn(n + q)
    y = zeros(n)
    for t in 1:n
        y[t] = μ + ε[q + t] + sum(θ[i] * ε[q + t - i] for i in 1:q)
    end
    return y, ε[q+1:end]
end
```

```
## generate_ma (generic function with 1 method)
```

Furthermore, I define a function to compute the OLS residuals as the difference between the actual variable, y, and the estimated one, $\hat{y}$.


```julia
# Define the model residuals for MA(q)
function ma_residuals(θ, y, q)
    n = length(y)
    res_ols = zeros(n - q)
    for t in q+1:n
        res_ols[t - q] = y[t] - sum(θ[i] * y[t - i] for i in 1:q)
    end
    return res_ols
end
```

```
## ma_residuals (generic function with 1 method)
```

### Estimation methods

To estimate $\theta$, there are mainly two methods available: maximum likelihood (ML hereafter, but do not confuse it with machine learning or Marxism-Leninism) and nonlinear least squares (NLS hereafter).


#### Maximum-Likelihood

The likelihood function for the MA(1) process is based on the joint density of $Y_t$. Assuming Gaussian innovations $\{\epsilon_t\} \sim N(0, \sigma^2)$, the log-likelihood function is:

$$
L(\mu, \theta, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{t=1}^n \epsilon_t^2,
$$

where $\epsilon_t = Y_t - \mu - \theta \epsilon_{t-1}$. Maximizing $L$ with respect to $\mu$, $\theta$, and $\sigma^2$ yields the ML estimates. 


#### Non Linear Least Squares

In NLS, we minimize the sum of squared residuals:

$$
Q(\theta) = \sum_{t=2}^n \big(Y_t - \mu - \theta (Y_{t-1} - \mu)\big)^2.
$$

The presence of nonlinearities in this equation makes it more difficult or even impossible to obtain a closed-form solution. Therefore, estimating $\theta$ requires an iterative optimization technique such as the Gauss-Newton algorithm or using other methods such as the Hannan-Rissanen algorithms. In the next section, both algorithms are defined and compared in terms of efficiency and generability.


##### Gauss-Newton Algorithm

The Gauss-Newton method approximates the Hessian matrix in Newton's method by ignoring second-order derivatives. The algorithm iteratively updates the parameter estimates:

$$
\theta^{(k+1)} = \theta^{(k)} - \big(J^T J\big)^{-1} J^T r,
$$

where $J$ is the Jacobian matrix of partial derivatives, and $r$ is the residual vector.

The take aways of Gauss-Newton algorithm is that it requires a good starting value for convergence and that converges quadratically near the true parameter if the residuals are small and $J^T J$ is well-conditioned. However, its strength is the lower variance and its versatility, i.e., it can be applied in a wild range of models.

The code for the Gauss-Newton method is the following:


```julia
# Gauss-Newton algorithm for MA(q)
function gauss_newton_ma(y, q, max_iter=5000, tol=1e-8)
    n = length(y)
    θ_hat = zeros(q)
    iter = 0

    while iter < max_iter
        res_ols = ma_residuals(θ_hat, y, q)
        n_res = length(res_ols)
        
        # Jacobian matrix
        J = zeros(n_res, q)
            for t in 1:n_res
                for j in 1:q
                    if t > j  # Ensure valid indexing
                        J[t, j] = -res_ols[t - j]
                    end
                end
            end

        # Update rule
        Δθ = - inv(J' * J) * J' * res_ols
        
        # Update the parameters
        θ_hat += Δθ
        iter += 1

        # Convergence check
        if norm(Δθ) < tol
            break
        end
    end

    return θ_hat, iter
end
```

```
## gauss_newton_ma (generic function with 3 methods)
```


##### Hannan-Rissanen Algorithm

The algorithm can be summarized as follows: First, an overfitted AR(p) model is fitted to the time series $Y_t$ using ordinary least squares (OLS). The residuals from this model, $\hat{\epsilon}_t$, are then computed. Next, these residuals are used in a regression where $Y_t$ is expressed as a function of lagged residuals, such as $\hat{\epsilon}_{t-1}, \hat{\epsilon}_{t-2}$, and so forth. This step provides the estimates for the MA parameters.

The key advantages of the Hannan-Rissanen algorithm is its simplicity and the fact that it avoids iterative optimization, making it computationally efficient. The take aways are its higher variance with respect to Gauss-Newton method. That is, although more computationally efficient, Hannan-Rissanen is less efficient in terms of standard errors.

The code for Hanann-Rissanen algorithm is the following:



```julia
function hannan_rissanen(y, q, p)
    n = length(y)
    
    # Estimate AR(p) model (overfitting)
    x_lags = hcat([circshift(y, i) for i in 1:p]...)
    reg_ar = x_lags[p+1:end, :]
    y = y[p+1:end]
    
    df_lags = DataFrame(reg_ar, :auto)
    df_lags[!, :y] = y
    var_names_ar = join(names(df_lags[:, 1:p]), "+")
    formula_ar = @eval @formula(y ~ $(Meta.parse(var_names_ar)))
    
    ols_model_ar = lm(formula_ar, df_lags)
    res_ols = residuals(ols_model_ar)
    
    # Fit MA(q) using AR(p) residuals
    ma_lags = hcat([circshift(res_ols, i) for i in 1:q]...)
    reg_ma = ma_lags[q+1:end, :]
    y = y[q+1:end]
    
    df_ma = DataFrame(reg_ma, :auto)
    df_ma[!, :y] = y
    var_names_ma = join(names(df_ma[:, 1:q]), "+")
    formula_ma = @eval @formula(y ~ $(Meta.parse(var_names_ma)))
    
    ols_model_ma = lm(formula_ma, df_ma)
    
    theta_est = coef(ols_model_ma)[2:end]  # MA coefficients
    return theta_est, residuals
end
```

```
## hannan_rissanen (generic function with 1 method)
```

## Simulation

Let $\theta$ = 0.8 and $\mu = 0$, so the true model is

$$
Y_t = \epsilon_t + 0.8 \epsilon_{t-1}
$$

In Julia:


```julia
θ_true = [0.8]
n = 1000
μ = 0
```


```julia
# Generate synthetic data
y, ε = generate_ma(1, θ_true, n, μ)
```

And, finally, we are estimating $\theta$ using Gauss-Newton and Hannan-Rissanen algorithms:


```julia
# Gauss-Newton Estimation
@time begin
    θ_gn, iter_gn = gauss_newton_ma(y, 1)
    println("Gauss-Newton estimated MA(1) theta: $θ_gn")
    println("Gauss-Newton iterations: $iter_gn")
end
```

```
## Gauss-Newton estimated MA(1) theta: [0.6803123867770745]
## Gauss-Newton iterations: 35
##   2.155574 seconds (1.65 M allocations: 114.122 MiB, 4.35% gc time, 99.93% compilation time)
```


```julia
# Hannan-Rissanen Estimation
@time begin
    θ_hr = hannan_rissanen(y, 1, 10)
    println("Hannan-Rissanen estimated MA(1) theta: $θ_hr")
end
```

```
## Hannan-Rissanen estimated MA(1) theta: ([0.7062939201493024], StatsAPI.residuals)
##  16.039448 seconds (17.34 M allocations: 1.085 GiB, 4.12% gc time, 99.88% compilation time: 2% of which was recompilation)
```

Next, I repeat the procedure for a MA(5) and a MA(10):

$$
Y_t = \epsilon_t + 0.6 \epsilon_{t-1} - 0.4 \epsilon_{t-2} + 0.3 \epsilon_{t-3} - 0.2 \epsilon_{t-4} + 0.1 \epsilon_{t-5}
$$


```julia
# Generate synthetic data
θ_true_2 = [0.6,−0.4,0.3,−0.2,0.1]
μ_2 = 0  # Scalar mean
```


```julia
y_2, ε_2 = generate_ma(5, θ_true_2, n, μ_2)
```


```julia
# Gauss-Newton Estimation
@time begin
    θ_gn2, iter_gn2 = gauss_newton_ma(y_2, 5)
    println("Gauss-Newton estimated MA(1) theta: $θ_gn2")
    println("Gauss-Newton iterations: $iter_gn2")
end
```

```
## Gauss-Newton estimated MA(1) theta: [0.055678787154190806, -0.16669250201494218, 0.12387123629752347, -0.13525402051629123, 0.10570048648183784]
## Gauss-Newton iterations: 14
##   0.009452 seconds (635 allocations: 741.068 KiB, 86.20% compilation time)
```


```julia
# Hannan-Rissanen Estimation
@time begin
    θ_hr2 = hannan_rissanen(y_2, 5, 10)
    println("Hannan-Rissanen estimated MA(1) theta: $θ_hr2")
end
```

```
## Hannan-Rissanen estimated MA(1) theta: ([0.050546375533578075, -0.17642527460111634, 0.09782944218220316, -0.09966194738663924, 0.053004838292738665], StatsAPI.residuals)
##   1.809492 seconds (935.90 k allocations: 59.844 MiB, 3.23% gc time, 99.42% compilation time)
```


```julia
θ_true_3 = [0.2, -0.15, 0.1, -0.05, 0.03, -0.02, 0.01, -0.008, 0.005, -0.003]
μ_3 = 0

y_3, ε_3 = generate_ma(10, θ_true_3, n, μ_3)
```


```julia
# Gauss-Newton Estimation
@time begin
    θ_gn3, iter_gn3 = gauss_newton_ma(y_3, 10)
    println("Gauss-Newton estimated MA(1) theta: $θ_gn3")
    println("Gauss-Newton iterations: $iter_gn3")
end
```

```
## Gauss-Newton estimated MA(1) theta: [0.10986389359225178, -0.21573162608616192, 0.13818006615653544, -0.1135201380915189, 0.09208018865957344, -0.15560657889666857, 0.05281409369900133, -0.09489915689601333, 0.0335075140459026, -0.03441854031247698]
## Gauss-Newton iterations: 15
##   0.010514 seconds (663 allocations: 1.406 MiB, 64.85% compilation time)
```


```julia
# Hannan-Rissanen Estimation
@time begin
    θ_hr3 = hannan_rissanen(y_3, 10, 10)
    println("Hannan-Rissanen estimated MA(1) theta: $θ_hr3")
end
```

```
## Hannan-Rissanen estimated MA(1) theta: ([0.10471398563185386, -0.21439991279942913, 0.08502404424361293, -0.049079608132977424, 0.027218150050031118, -0.09804103531000737, -0.01756058093225093, -0.02222802662554225, -0.018730886093954516, 0.004965436257681065], StatsAPI.residuals)
##   0.010917 seconds (3.48 k allocations: 1.937 MiB, 61.50% compilation time)
```

As observed, the computational efficiency of the Hannan-Rissanen method improves relative to the Gauss-Newton algorithm as the order of the MA process increases. This is because Hannan-Rissanen avoids the iterative optimization steps required by Gauss-Newton, such as calculating the Hessian matrix, which becomes computationally intensive for higher-dimensional systems.

It is also important to note that as the order q of the MA process grows, the variability in parameter estimation increases. For an MA(1) process, there is only one unique parameter configuration that can exactly replicate the data-generating process (DGP). In contrast, for an MA(q) process where q > 1, multiple combinations of parameters can produce identical time series, leading to greater uncertainty in parameter estimates.

Finally, something to take into account is that, also, as q increases, Gauss-Newton algorithm might not converge. For example, consider the following MA(10) process:


```julia
θ_true_4 = [-0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.05, 0.3, -0.1, 0.25]
μ_4 = 0

y_4, ε_4 = generate_ma(10, θ_true_4, n, μ_4)
```


```julia
# Gauss-Newton Estimation
@time begin
    θ_gn4, iter_gn4 = gauss_newton_ma(y_4, 10)
    println("Gauss-Newton estimated MA(1) theta: $θ_gn4")
    println("Gauss-Newton iterations: $iter_gn4")
end
```

```
## Gauss-Newton estimated MA(1) theta: [-2788.6266638885327, 438.9982014985375, -421.84647124956103, 453.11272975094676, 334.97641291899316, -88.19932829922455, -187.6406972523508, 123.52662572452202, 155.8560851922902, 157.0827338455622]
## Gauss-Newton iterations: 5000
##   1.348244 seconds (60.48 k allocations: 457.113 MiB, 7.83% gc time, 0.80% compilation time)
```


```julia
# Hannan-Rissanen Estimation
@time begin
    θ_hr4 = hannan_rissanen(y_4, 10, 10)
    println("Hannan-Rissanen estimated MA(1) theta: $θ_hr3")
end
```

```
## Hannan-Rissanen estimated MA(1) theta: ([0.10471398563185386, -0.21439991279942913, 0.08502404424361293, -0.049079608132977424, 0.027218150050031118, -0.09804103531000737, -0.01756058093225093, -0.02222802662554225, -0.018730886093954516, 0.004965436257681065], StatsAPI.residuals)
##   0.013567 seconds (3.48 k allocations: 1.940 MiB, 69.87% compilation time)
```

As opposed to earlier, the coefficients in this case are larger, increasing the risk of large gradients or Hessian values during optimization. This makes the derivatives highly sensitive, potentially leading to instability. Additionally, alternating signs and having the sum of coefficients close to zero are factors that enhance stability and help avoid biased estimates. However, both MA(10) processes satisfy these conditions. Finally, as mentioned earlier, initial guesses are critical: a well-chosen initial guess is essential for ensuring convergence, while poor guesses can result in divergence or slow convergence.
