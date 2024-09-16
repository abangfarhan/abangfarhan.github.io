---
layout: post
title: Visualizing BG/NBD Model Distributions
category: blog
math: true
tags: [math]
---

Recently, I learned about the BG/NBD (beta-geometric/negative binomial distribution) model for forecasting customer transactions (as part of a data science course by [pacmann.io](https://pacmann.io/)). It involves several probability distributions that I wasn't familiar with, and so I created several graphs using [Desmos graphing calculator](https://www.desmos.com/calculator) to visualize them. I was particularly interested in seeing how changes in the probability density functions' (PDF) parameters affect the shape of the PDF. I hope these visualizations also help you too.

I wouldn't go into detail about the BG/NBD model or about each probability distribution. You can read the original paper [here](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf).

# Poisson Distribution for Transaction Rates

For a particular customer, the probability that they will make $x$ transactions in a period assuming that they make $\lambda$ transactions per period on average is described by the Poisson probability distribution:

$$
f(x|\lambda) = \frac{e^{-\lambda}\lambda^x}{x!}, x \geq 0
$$

<iframe src="https://www.desmos.com/calculator/1vjj32dg12" frameBorder="0" width="100%" style="min-height:400px"></iframe>

Note: the chart should actually be bar charts, since we're plotting a PMF for discrete values.

# Gamma Distribution for Heterogeneity of Transaction Rate

The heterogeneity of $\lambda$ among all customers is described the Gamma distribution, parameterized by the shape parameter $r$ and the scale parameter $\alpha$.

$$
f(\lambda | r, \alpha) = \frac{\alpha^r \lambda^{r-1} e^{-\lambda \alpha}}{\Gamma(r)}, \lambda > 0
$$

In Desmos, the Gamma function is accessed via the factorial, where $\Gamma(r) = (r - 1)!$.

<iframe src="https://www.desmos.com/calculator/jdzwigjmyc" frameBorder="0" width="100%" style="min-height:400px"></iframe>

# Beta Distribution for Heterogeneity of Drop-off Probability

Every time a customer makes a transaction, there's a chance that they will stop making any transactions afterwards with probability $p$, which is called the drop-off probability. This probability is assumed to be independent across time and across customers, and the heterogeneity of $p$ among all customers is assumed to follow the Beta distribution, which is parameterized by the shape parameters $a$ and $b$:

$$
f(p | a, b) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)}, 0 \leq p \leq 1
$$

Where $B(a,b)$ is the beta function, which can be expressed as $B(a, b) = \Gamma(a) \Gamma(b) / \Gamma(a + b)$.

<iframe src="https://www.desmos.com/calculator/prh0yk8za4" frameBorder="0" width="100%" style="min-height:400px"></iframe>

Notice the following properties of the Beta distribution:

- If $a = b = 1$, the distribution is equal to the uniform distribution, where all drop-off probabilities are equally likely to occur.
- If $a = b > 1$, the distribution is symmetric around 0.5.
- If $a > b$, the distribution is skewed to the left, meaning customers with high drop-off probability are more likely to occur.
- If $a < b$, the distribution is skewed to the right, meaning customers with low drop-off probability are more likely to occur.
- As $a$ and $b$ increases, the distribution's spread becomes narrower.
