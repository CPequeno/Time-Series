
## Introduction

This document provides an intuitive explanation of the ergodic theorem. The theorem asserts that if a process meets two key conditions —weak stationarity and the mixing property— the time series mean will converge (in probability or almost surely, depending of the ergodic theorem) to the ensemble mean. This theorem is a kind of (weak) law of large numbers (WLLN) where (1) identical distributed property is replaced by weak stationarity, which ensures the process has a constant mean over time and an autocovariance function that depends solely on the number of lags, and (2) the independent assumption is replaced by the mixing property, which guarantees that as the time intervals between observations grow, the dependence between variables diminishes to the point of being negligible.

## Setup

Consider a time series process, for example Spain GDP recordings from 1930 to 2020. Although following Karl Marx, history repeats itself, once as tragedy and later as farce, in general we cannot repeat exactly the same process again and again to check different counterfactuals in history. That is, we cannot repeat the history in order to check the GDP recordings if e.g. Franco did not stage a coup d'état in 1936. Thus, we only have one and only one -finite- realization of an stochastic process. Let f(t) denotes the actual time series. We can compute the mean and the autocovariance function and an infinite number of statistical properties.

Strict stationarity states that the statistical properties do not change over time. Formally: A stochastic process ${X_t}$ is said to be strictly stationary if the joint probability distribution of $(X_{t_1}, X_{t_2}, \dots, X_{t_n})$ is the same as the joint probability distribution of:

$$
(X_{t_1+h}, \ X_{t_2+h}, \dots, \ X_{t_n+h}),
$$

for all $t_1, t_2, \dots, t_n \in \mathbb{Z}$, $h \in \mathbb{Z}$, and $n \in \mathbb{N}$.
 
Weakly stationarity states, basically, that the first and second moments of the process are time-invariant. Formally: a stochastic process ${X_t^{i}}$ for $i \in {1, 2, ..., N}$ is said to be weakly stationary if the following conditions hold:

(1) the mean of the process i for $i \in {1, 2, ..., N}$ is constant over time;

$$
E[X_t] = \mu, \quad \forall t \in \mathbb{Z}.
$$


(2) the variance is constant over time; and

$$
Var(X_t) = \sigma^2, \quad \forall t \in \mathbb{Z}.
$$
    
    
(3) the autocovariance between two time points depends only on the time lag $(h = t_2 - t_1$, and not on the actual times $t_1$ and $t_2$
    
$$
Cov(X_t, \ X_{t+h}) = \gamma(h), \quad \forall t, h \in \mathbb{Z}.
$$

The transition to a probability model occurs when we consider all possible realizations with the same statistical properties (or, at least, the first two moments) as the recorded one. Thus, we create an artificially infinite number of realizations. This is the so-called ensemble. Due to the fact that we are assuming stationarity, we can assume that e.g. the time average of each realization (including, obviously, the actual one) ${X_t^{(1)}}, {X_t^{(2)}}, ..., {X_t^{(n)}}$ is constant over time. That is,

$$
E[X_t^{(i)}] = E[X_t] = \mu
$$


$\forall i \in {1, 2, ..., N}$. Note that the time average is not only constant across t but it is the same for all realizations. This is due to the fact that all realizations comes from the same probability (ensemble) model. Now, let mixing property holds. That is,

$$
\lim_{t \to \infty} Cov(X_t, \ X_{t+h}) = 0.
$$

Therefore, by the Ergodic theorem, we can conclude that

$$
\lim_{t \to \infty} \frac{1}{T} \sum_{t=1}^{T} X_t^{(i)} = E[X_t] = \mu,
$$

where the first equality is the ergodicity property, i.e., the equality of the time average with the ensemble average and the second equality comes from stationarity.


### Define the AR and random walk processes

Now, consider two stochastic processes: an AR(1)

$$
X_t = \phi_1 X_{t-1} + \epsilon_t
$$

where we assume that $\phi = 0.5$, so the process is not only stationary but causal, that is, the roots of the process are outside of the unitary circle, so the process can be expressed as a function of only past observations. It ensures the process does not rely on future inputs, making it feasible for real-world prediction. (I know, a somehow weird property to call it 'causality', but basically causality is just invertibility for AR processes. That is, a causal AR can be expressed as a MA($\infty$) and an invertible MA can be rewritten as a -causal- AR($\infty$)).


```julia
using Random
using Plots
using Statistics
using Base.Threads
using Distributions
```


```julia
ϕ = 0.5
```

```
## 0.5
```

```julia
function ar1_process(t, ϕ)
    ar1 = zeros(t)
    for t in 2:t
        ar1[t] = ϕ * ar1[t-1] + randn()
    end
    return ar1
end
```

```
## ar1_process (generic function with 1 method)
```

and a random walk

$$
X_t = X_{t-1} + u_t
$$


```julia
function random_walk(t)
    walk = cumsum(randn(t))
    return walk
end
```

```
## random_walk (generic function with 1 method)
```

Note that, for the random walk process, $E[X_t] = X_{t-1}$, so the mean is far from being constant. Also can be checked that the variance is not constant and that autocovariance depends not only on the lags, but on time. Hence, this is not a stationary process.

### Simulation and parallelization

The following chunk basically generates, for each of N simulations, a random walk and an AR(1) process, both of size n (the number of time steps or data points). These are stored in two vectors, random_walks and ar1_processes.


```julia
t = 100 # Number of observations of the process.
N = 1000 # Number of simulations (different realizations of the process).
```


```julia
random_walks = Vector{Vector{Float64}}(undef, N)
ar1_processes = Vector{Vector{Float64}}(undef, N) # The undef keyword indicates that the vectors are being allocated without initializing the values right away. They will be filled during the parallelized loop.

Threads.@threads for i in 1:N
    random_walks[i] = random_walk(t)
    ar1_processes[i] = ar1_process(t, ϕ)
end
```

Nonetheless, there is something else in the chunk: the 'threads' command. By using Threads.@threads, I am instructing Julia to divide the task of generating ARs and random walks into multiple smaller tasks, each of which can be run concurrently (that is why I did allocate but not initialized the values of both processes). This can significantly reduce the overall time it takes to generate the data, especially if the machine has multiple CPU cores. This process is called parallelization. Without parallelization, I would generate each of the N random walks and AR(1) processes sequentially. However, this can be time-consuming, especially when N or n is large.

Parellelization works in the following way:

When the program executes the loop, the system might look like this (assuming N = 1000 and n = 100):

Thread 1 might handle the first 250 random walks and AR(1) processes (from indices 1 to 250).
Thread 2 handles the next 250 (from indices 251 to 500).
Thread 3 handles another 250 (from indices 501 to 750).
Thread 4 handles the final 250 (from indices 751 to 1000).

Threads.@threads is used to take advantage of multi-core processing, where each core can handle different parts of the task simultaneously. By distributing the iterations across threads, you can scale the process efficiently, especially on machines with multiple cores, resulting in a significant speedup compared to running the loop sequentially.

Each thread computes its assigned subset of random walks and AR(1) processes independently and in parallel, allowing for concurrent execution and thus reducing the overall computation time.

### Plot the results

The code below creates an animation where, as the sample size increases from 1 to N, random walks and AR(1) processes are generated along with histograms of their time averages. The animated plot is included in the folder.

Basically, the code is generating an animated visualization where, as the sample size (i.e., the number of simulations, N) increases, several things are calculated. For each simulation, the code calculates the (ensemble) means of random walks and AR(1) processes in parallel. It then visualizes the random walks, AR(1) processes, and the distributions of their means, showing how the data and the mean distributions evolve as N grows.


```julia
animation = @animate for i in 1:50
    
    sample_size = Int(N * i / 100)
    sampled_random_walks = random_walks[1:sample_size]
    sampled_ar1_processes = ar1_processes[1:sample_size]

    random_walk_averages = Vector{Float64}(undef, sample_size)
    ar1_averages = Vector{Float64}(undef, sample_size)

    Threads.@threads for j in 1:sample_size
        random_walk_averages[j] = mean(sampled_random_walks[j])
        ar1_averages[j] = mean(sampled_ar1_processes[j])
    end

    
    p1 = plot(title="Random Walks", legend=false)
    for rw in sampled_random_walks
        plot!(p1, rw, lw=0.5, alpha=0.3, color=RGB(rand(), rand(), rand()))  # Random colors
    end

    p2 = plot(title="AR(1) Processes", legend=false)
    for ar in sampled_ar1_processes
        plot!(p2, ar, lw=0.5, alpha=0.3, color=RGB(rand(), rand(), rand()))  # Random colors
    end

    p3 = histogram(random_walk_averages, bins=30, normalize=true, alpha=0.5,
                   label="Random Walk Means", color=:blue, title="Random Walk Means Density")
    p4 = histogram(ar1_averages, bins=30, normalize=true, alpha=0.5,
                   label="AR(1) Means", color=:red, title="AR(1) Means Density")

    plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 600))
end
```


```julia
gif(animation, "animated_processes_and_histograms_parallel.gif", fps=4)
```


![Ergodic Theorem GIF](ergodic_animation.gif)


On the one hand, for the AR(1) process, the sample means become increasingly concentrated around zero as the sample size grows, reflecting the process's ergodic behavior. On the other hand, for the random walk, the sample means show wider and wider tails as N increases. That is, while the ensemble mean is zero (i.e., the average across the simulations) as S goes to infinity, the time average (that is, the average across t for each simulation) is not zero, as t goes to infinity. That is why the tails of the random walk processes keep increasing as the number of simulations increases, contrasting with the remarkably narrow tails of the AR processes. This behavior reflects the non-ergodic nature of random walk process. A random walk does not converge to a fixed mean because the process tends to "drift" over time, and the time average does not converge to a well-defined constant value, thus violating the assumptions of the ergodic theorem. That is, while AR processes mean reversion, that is not the case for random walks, which impairs stationarity and, therefore, ergodicity.

To make it even clearer, imagine that {$X_t$} are random walks which take only two values: 1 and -1. As t increases, a particular random walk process, e.g. $X_t^{i}$, for $i \in {1, 2, ..., N}$ is highly implausible that satisfies $\frac{1}{T} \sum_{t=1}^{T} X_t^{(i)} = 0$, because for a given value of t, the process can be consistently above or below zero. And as I said above, the trend precludes the process to exhibit mean reversion. Without loss of generality, let $X_t^{1}$ to be a realization that more or less consistently went upwards zero. However, as the number or realizations, N, goes to infinity, there will be a realization, e.g., $X_t^{24}$, which is, also more or less consistently, going downwards, compensating the trend of ${X_t^{1}}$. Hence, $E[X_t] = 0$ as N, i.e., the sample size, goes to infinity. This is basically what we see in the top-left figure of the graph.

### T-test, R2 and Durbin-Watson (DW) statistics.

Finally, for each simulation of each stochastic process, let me calculate the $R^2$:

$$
R^2 = 1 - \frac{\sum_{t=1}^{n} (X_t - \hat{X}_t)^2}{\sum_{t=1}^{n} (X_t - \bar{X})^2}
$$


```julia
function r_squared(x, ϕ)
    residuals = x[2:end] .- ϕ .* x[1:end-1]
    ss_total = sum((x .- mean(x)) .^ 2)
    ss_residual = sum(residuals .^ 2)
    return 1 - ss_residual / ss_total
end
```

```
## r_squared (generic function with 1 method)
```

the t-statistic:

$$
t = \frac{\bar{X}}{\text{SE}(\bar{X})}
$$


```julia
function t_statistic(x)
    return mean(x) / (std(x) / sqrt(length(x)))
end
```

```
## t_statistic (generic function with 1 method)
```

and the DW statistic:

$$
DW = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
$$


```julia
function durbin_watson(x, ϕ)
    residuals = x[2:end] .- ϕ .* x[1:end-1]
    numerator = sum(diff(residuals) .^ 2)
    denominator = sum(x[2:end] .^ 2)
    return numerator / denominator
end
```

```
## durbin_watson (generic function with 1 method)
```

which is useful for detecting first-order autocorrelation in the residuals of regression models. By comparing the residuals' behavior over time, it tells us whether the model has captured all relevant temporal patterns or if it needs adjustments (e.g., by including lagged variables, trends, or seasonality). Values close to 2 suggest that the model residuals are independent, while values much smaller or larger than 2 indicate potential autocorrelation issues.

Next, I create a matrix of two columns and N rows to store the statistics for each realization of both processes too.


```julia
# Matrices to store statistics
ts = zeros(N, 2)    # t-statistics for random walk and AR(1)
DW = zeros(N, 2)   # Durbin-Watson for random walk and AR(1)
R2 = zeros(N, 2)   # R-squared for random walk and AR(1)
```

Also, let 'reject_RW' and 'reject_AR' be the number of times that we reject the null hypothesis in $H_0: E[X_t] = 0$ vs $H_0: E[X_t] != 0$. (They are in levels, but once the algorithm computes everything, they are normalized to be interpreted as probabilities.)


```julia
reject_RW = 0
reject_AR = 0
```

Set a significance level for the test:


```julia
α = 0.05
```

and compute all the statistics for each simulation (rows) and process (columns) using a for loop.


```julia
for i in 1:N

    # Random walk t-statistic
    ts[i, 1] = t_statistic(random_walks[i])
    # AR(1) process t-statistic
    ts[i, 2] = t_statistic(ar1_processes[i])
    
end
   
ts 
```

```
## 1000×2 Matrix{Float64}:
##   -7.6902    1.45266
##   24.9184    2.59349
##  -16.4098    0.603435
##   18.5249   -0.0478928
##    3.58169  -1.80247
##   11.3312   -2.02835
##    2.17107  -2.51517
##   25.6629   -2.11658
##  -11.8221   -2.74491
##   22.2943   -3.99699
##    ⋮        
##   19.7538   -0.67553
##    1.5086   -1.69493
##   19.6111    1.02441
##   21.3076   -2.38796
##    6.30843   2.48825
##   -3.37393   2.36008
##  -10.2983    4.33156
##   15.8516    0.25064
##    6.96762   2.41402
```
    

```julia
for i in 1:N
    
    # R-squared for random walk
    R2[i, 1] = r_squared(random_walks[i], 1)
    # R-squared for AR(1) process
    R2[i, 2] = r_squared(ar1_processes[i], 0.5)
    
end

R2
```

```
## 1000×2 Matrix{Float64}:
##  0.831357  0.198068
##  0.909851  0.277835
##  0.907014  0.205129
##  0.921362  0.240254
##  0.944089  0.250543
##  0.926229  0.190937
##  0.828086  0.0539583
##  0.831484  0.117249
##  0.984224  0.30487
##  0.893729  0.106481
##  ⋮         
##  0.940681  0.311854
##  0.924967  0.132724
##  0.959227  0.263485
##  0.877799  0.138383
##  0.852484  0.143204
##  0.921223  0.313455
##  0.947059  0.0793354
##  0.976172  0.266704
##  0.859036  0.198259
```


```julia
for i in 1:N

    # Durbin-Watson statistic for random walk
    DW[i, 1] = durbin_watson(random_walks[i], 1)
    # Durbin-Watson statistic for AR(1) process
    DW[i, 2] = durbin_watson(ar1_processes[i], 0.5)
    
end
```

Finally, given that I already computed the t-statistic, I can check the (relative) number of times the null is rejected:


```julia
reject_RW = 0
reject_AR = 0

for i in 1:N
    if abs(ts[i, 1]) > quantile(TDist(t-1), 1 - α / 2)
        reject_RW += 1
    end
    if abs(ts[i, 2]) > quantile(TDist(t-1), 1 - α / 2)
        reject_AR += 1
    end
end
```

As it can be checked, the $DW \approx 1 < 2$, which implies positive autocorrelation (by construction, the existence of autocorrelation in an AR or a random walk was something to expect) and the null is rejected with high probability for the random walk process than for the AR(1), as expected.
