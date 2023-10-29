---
layout    : post
title     : Basic Portfolio Theory
subtitle  : Just some notes about portfolio theory.
date      : 2018-04-15 15:30
published : false
author    : abangfarhan
category  : blog
math: true
tags      : [math, finance]
---

A few months ago I learned about portfolio theory, and in this post I will write
some notes about it because I found the mathematics to be quite easy to
understand. However, since this is just a blog post and I am quite lazy, for now
I won't include any references.

# What is a Portfolio?

In this post, when I talked about portfolio, I am referring to portfolio of
stocks. Furthermore, the portfolio would only consists of two stocks (i.e. stock
from two different corporations) because it is simple.

# The Basics

Suppose that we are interested in creating a portfolio of that consisted of two
stocks: Stock A and Stock B. We need to know some information about those
stocks: the expected return and the standard deviation. The notation for the
expected return is $E(R_A)$ and $E(R_B)$ for Stock A and Stock B, respectively.
While the notation for the standard deviation is $\sigma_A$ and $\sigma_B$,
respectively. There are two approaches in calculating each of them: using past
historical data, or using forecasts about future events. However, I would not
explain about them further for now.

The expected return for a particular stock expresses the average return an
investor could earn if she invested in that stock. It is expressed as a
percentage (or in decimal, e.g. you can express it as 10% or 0.1). Meanwhile,
the standard deviation measures the risk of investing in that stock. It is
expressed in percentage (or in decimal), just like the expected return. The
higher the standard deviation, the higher the risk (i.e. you can expect to earn
much more than the average, but you also should expect to earn much less than
the average too).

<div hidden>
<!-- useful macros for MathJax -->
$\newcommand{\cov}[1]{\text{COV} _ {#1}}$
$\newcommand{\corr}[1]{\rho _ {#1}}$
</div>

In addition to the expected return and standard deviation, we also need to
calculate the covariance (denoted as $\cov{AB}$) or the coefficient of
correlation (denoted as $\corr{AB}$). The covariance tells us the nature of
relationship between Stock A and Stock B. If the covariance is positive, then
the relationship is also positive (i.e. when Stock A's return increased, the
Stock B's return will also increase), and vice versa. The covariance doesn't
tells us how strong is the relationship, but the correlation coefficient does.
Correlation coefficient lies between -1 and +1, where a correlation coefficient
of +1 tells us that the relationship between Stock A and Stock B is perfectly
positive (i.e. when Stock A's return increased then Stock B's return will always
increase too).

In summary:

$$
\begin{align}
E(R_i) &= \text{expected return of Stock } i \\
\sigma_i &= \text{standard deviation of Stock } i \\
\cov{ij} &= \text{covariance between Stock } i \text{ and Stock } j \\
\corr{ij} &= \text{correlation coefficient between Stock } i \text{ and Stock } j
\end{align}
$$

# Why Invest in a Portfolio?

Why do we want to invest in a portfolio of stocks, not just a stock? After all,
we can just invest in the stock with the highest expected return, right? Wrong.
The keyword here is **diversification**. By investing in two different stocks,
we can minimize the risk. That's why we have to consider the correlation. The
more negatively correlated the stocks are, the better. For example, if the
$\corr{AB} = -1$, then we can be sure that when Stock A's performed badly (i.e.
the price decrease), the loss would be countered by return from the Stock B
(i.e. the Stock B's price will increase).

# What do We Want to Do?

The main objective in portfolio management is determining the **weight** of each
stock in the portfolio, here denoted $W_A$ and $W_B$ for Stock A and Stock B,
respectively. This weight tells us the proportion of our money to invest in each
stock. For example, if $W_A = 0.4$ and $W_B = 0.6$, and we have decided to
invest $\\$1000$ to the portfolio, then we will use $0.4 \times \\$1000 = \\$400$
to invest in Stock A, and $0.6 \times \\$1000 = \\$600$ to invest in Stock B.

# Properties of a Portfolio

Like stocks, portfolio also has expected return and standard deviation, which is
determined by the expected return, standard deviation, and weight of the stocks
that are included in the portfolio. The formula is given below.

$$
\begin{align}
E(R_p) &= \text{expected return of the portfolio} \\
       &= W_A E(R_A) + W_B E(R_B) \\
\sigma_p^2 &= \text{variance of the portfolio} \\
           &= W_A^2 \sigma_A^2 + W_B^2 \sigma_B^2 + 2 W_A W_B \cov{AB} \\
\sigma_p &= \text{standard deviation of the portfolio} \\
         &= \sqrt{\sigma_p^2} \\
\end{align}
$$

As you can see, both the portfolio's standard deviation and the expected return
is determined by the weight. It means that for every possible values of the
weight we can calculate the standard deviation, expected return, and create a
scatter plot that shows the relationship between standard deviation and expected
return. Below is an example ($E(R_A)=0.15$, $E(R_B)=0.08$, $\sigma_A=0.19$,
$\sigma_B=0.10$, $\corr{AB}=-0.45$).

![portfolio curve]({{site.baseurl}}/img/2018-04-15-basic-portfolio-math/00.png)

As you can see, the plot is convex to the y-axis. I think this can be
mathematically proven, but I have not done that yet. Maybe another time.

# Building the Portfolio

There are various ways to calculate the weight for the portfolio. We will talk
about them in this section.

## Minimum Variance Portfolio

If we want to minimize the risk of our portfolio, then we might want to create a
portfolio that have the possible minimum variance. To do that, we determine the
weight based on this formula:

$$
\begin{align}
W_A = \frac{\sigma_B^2 - \cov{AB}}{\sigma_A^2 + \sigma_B^2 - 2\cov{AB}}
\end{align}
$$

In the future I will show the derivation of that formula. But for now we will
skip it. Meanwhile, on the graph below you can see where the minimum portfolio
is on the portfolio curve.

![portfolio curve with MVP]({{site.baseurl}}/img/2018-04-15-basic-portfolio-math/01.png)

## Optimal Portfolio

Now I will introduce you to the concept of risk aversion.
