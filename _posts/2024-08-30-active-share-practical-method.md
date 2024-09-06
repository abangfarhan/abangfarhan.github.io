---
layout: post
title: A Practical Way to Calculate Fund Active Share
category: blog
math: true
tags: [finance, math]
---

Active Share is a metric that measures how different a fund is from its
benchmark. Its value ranges from 0% (meaning the fund's underlying stocks and
their weights are completely the same as the benchmark) to 100% (the fund's
underlying stocks and their weights are completely different from the
benchmark). It is calculated with this formula:

$$
AS = 0.5 \times \sum_{i=1}^n | w_i - b_i |
$$

where $w_i$ is the weight of stock $i$ in the portfolio, and $b_i$ is the
weight of stock $i$ in the benchmark; $w_i - b_i$ is usually called "active
weight". In other words, Active Share is equal to half of the sum of the fund's
absolute active weights. For more on this you can read [this Investopedia
article](https://www.investopedia.com/articles/mutualfund/07/active-share.asp),
or [this Alpha Architect
article](https://alphaarchitect.com/2019/10/active-share-predictor-of-future-performance-or-urban-legend/)
covering several researches on the relationship between Active Share and fund
performance.

In this post I will show you a practical way to calculate Active Share. Note that
I haven't found this method discussed anywhere else, but if you have please
contact me so I can credit them.

# The Long Way: Example

Consider a portfolio consisting of two stocks, and its benchmark consisting of three stocks.
Here are the weights for the portfolio:

|Stock|Portfolio Weight|
|-|-|
|A|60%|
|B|40%|

And here are the weights for the benchmark:

|Stock|Benchmark Weight|
|-|-|
|A|40%|
|C|20%|
|D|40%|

To calculate the Active Share, we can join the two tables above as follows:

|Stock|Portfolio Weight|Benchmark Weight|
|-|-|-|
|A|60%|40%|
|B|40%|-|
|C|-|20%|
|D|-|40%|

Then we can calculate the absolute active weight for each stock:

|Stock|Portfolio Weight|Benchmark Weight|Absolute Active Weight|
|-|-|-|-|
|A|60%|40%|20%|
|B|40%|-|40%|
|C|-|20%|20%|
|D|-|40%|40%|

The Active Share in this case is equal to (20% + 40% + 20% + 40%) / 2 = 60%.

# The Practical Way: Example

The previous calculation is complicated because it requires us to combine the
two tables. The combined table need to include all stocks that are in the
portfolio or in the benchmark, which might not be too easy to do in Excel.

Fortunately, there is a trick to calculate Active Share without having to merge
the two tables. I will show you an example.

Let's start with the original table for the portfolio weights:

|Stock|Portfolio Weight|
|-|-|
|A|60%|
|B|40%|

Then we add a column indicating the benchmark weight, but only for the stocks
that are in the portfolio:

|Stock|Portfolio Weight|Benchmark Weight|
|-|-|-|
|A|60%|40%|
|B|40%|-|

In Excel, this can be easily done using INDEX-MATCH or VLOOKUP by referencing
the benchmark weights table.

The next step is to calculate the absolute active weights:

|Stock|Portfolio Weight|Benchmark Weight|Absolute Active Weight|
|-|-|-|-|
|A|60%|40%|20%|
|B|40%|-|40%|

Now we need to calculate these two values:

- $a$ = the sum of absolute active weights in the table above: 20% + 40% = 60%.
- $b$ = the sum of benchmark weights in the table above: 40% + 0% = 40%.

The Active Share is equal to $(a + (1 - b)) / 2$ = (60% + (1 - 40%)) / 2 = 60%, which is the same as before.

Why does this formula works? Consider that:

- $a$ is the sum of absolute active weights of all stocks that are in the fund
- $b$ is the sum of benchmark weights of all stocks that are in the fund. This
  implies that $1 - b$ is equal to the weights of all stocks in the benchmark
  that are not in the fund, which (by definition) is equal to the sum of
  absolute active weights of all stocks that are not in the fund.

Let me repeat it once more for clarity:

- $a$ is the sum of absolute active weights of all stocks that are in the fund
- $1 - b$ is the sum of absolute active weights of all stocks that are **not** in the fund

Therefore, $a + (1 - b)$ is equal to the sum of absolute active weights of all
stocks (both in the fund and not in the fund). If we divide this number by 2 we
will immediately get the Active Share.
