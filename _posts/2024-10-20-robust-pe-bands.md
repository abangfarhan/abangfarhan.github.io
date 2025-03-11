---
layout: post
title: Creating Robust PE Bands
category: blog
math: true
tags: [finance,math]
---

Equity analysts often use a tool called "PE (price to earnings ratio) bands", which plots a stock's time series PE and compare it with the historical mean and standard deviation. Here's an example PE band for BBCA stock:

![BBCA PE Band](/img/robust-pe-bands/bbca-pe-band.png)

The idea is to assess the stock's valuation relative to its historical valuations. For example, if the PE is "below -1 standard deviation" (which is a short-hand for "below the mean minus 1 standard deviation"), then the analyst might conclude that the stock is currently cheap by historical standard, and might have potentially high upside.

This style of analysis is not without problems. For example, if the PE is below -1 standard deviation, without any other information it is impossible to know if the stock is really "cheap" or it is actually being de-rated by investors.

One problem that I want to address in this post is the fact that mean and standard deviation are known to be easily distorted by extreme values in the data. Here's an example PE band for INCO stock:

![INCO PE Band](/img/robust-pe-bands/inco-pe-band.png)

In the chart above, because of extremely high PE ratios in the 2020 period, the PE band became very high and very wide. The extremely high PE ratios usually happened during periods of very low earnings, which caused the PE ratio to explode. Since these periods rarely happen, using the conventional method of creating the PE band might not be accurate.

What I usually do to alleviate this problem are as follows:

1. Use the median instead of the mean
2. Use "IQR standard deviation" or "MAD standard deviation" instead of the conventional standard deviation

IQR standard deviation is calculated as follows[^1]:

$$
\sigma_{\text{IQR}}(x) =  \text{IQR}(x) / 1.349
$$

where IQR is the interquartile range, which is calculated as the difference between the data's 75th and 25th percentile. Here's how to retrieve $\sigma_{\text{IQR}}$ using BQL:

```
=BQL(<ticker>,"(quantile(#pe,0.75) - quantile(#pe,0.25))/1.349","#pe=pe_ratio(dates=range(-5Y,0D))")
```

Another measure that you can use is "MAD standard deviation"[^2] ("MAD" stands for median absolute deviation):

$$
\sigma_{\text{MAD}}(x) = \text{median}(|x - \text{median}(x)|) / 1.4826 
$$

I found that stocks with periods of extremely high PE tend to have lower $\sigma_{\text{MAD}}$ compared to $\sigma_{\text{IQR}}$, which in my opinion is better.

Using BQL in Excel, it is easy to retrieve $\sigma_{\text{MAD}}$:

```
=BQL(<ticker>,"median(abs(#pe-median(#pe)))/1.4826 ","#pe=pe_ratio(dates=range(-5Y,0D))")
```

Below I compare the PE bands created using conventional method vs the robust methods for BRIS stock:

![BRIS PE Bands 1](/img/robust-pe-bands/bris-pe-bands-1.png)

Let's zoom in a little:

![BRIS PE Bands 2](/img/robust-pe-bands/bris-pe-bands-2.png)

As you can see, the robust method results in much more reasonable PE bands compared to the conventional method. Moreover, as I have stated, the MAD method results in narrower band than the IQR method.

Below are the PE bands for INCO stock:

![INCO PE Bands 2](/img/robust-pe-bands/inco-pe-bands-2.png)

Let's zoom in a little:

![INCO PE Bands 3](/img/robust-pe-bands/inco-pe-bands-3.png)

Once again, the robust method results in much more reasonable PE bands. Using this method, I become more confident that the PE bands are less affected by extremely high values in the data.

For further illustration, I have created the PE bands for various Indonesia's stocks (using MAD method):

![Various PE Bands (MAD Method)](/img/robust-pe-bands/pe-bands-mad.png)

That's it for this post. I hope you learn something new.

Notes:

- All PE data shown in this post are retrieved from Bloomberg. I used the trailing 12-month earnings data as the denominator.

[^1]: <https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/iqrange.htm>
[^2]: [Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median](https://doi.org/10.1016%2Fj.jesp.2013.03.013)
