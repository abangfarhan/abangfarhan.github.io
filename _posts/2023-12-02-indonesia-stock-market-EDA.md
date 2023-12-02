---
layout: post
title: "Exploratory Data Analysis: Indonesia’s Stock Market Characteristics and Trends"
category: blog
math: true
tags: [finance, data visualizations]
---

<script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>

*Please use a PC or use landscape mode to see this post for optimal experience. Thank you.*

***Disclaimer**: the information contained on this post and the resources available for download through this post is not intended as, and shall not be understood or construed as, financial advice. All views and ideas expressed on this post are my own and do not reflect the views of my employer.*

## Background

Gaining a better understanding about a stock market is important for anyone looking to invest in it. While there are so many aspects to explore in this field, one particularly interesting topic is the behavior of the stock market's investors, especially regarding their investment size and sectoral allocations. Here in Indonesia, the stock market is a place where [trillions of rupiah are traded every day](https://money.kompas.com/read/2021/09/14/202000326/bei-targetkan-rata-rata-transaksi-harian-mencapai-rp-13-5-triliun-pada-2022) (on average) by various parties--including retail investors, asset management companies, and so on--and facilitated by [Indonesia Stock Exchange](https://www.idx.co.id), stock brokers, and other [supporting institutions](https://www.ojk.go.id/id/kanal/pasar-modal/Pages/Lembaga-dan-Profesi-Penunjang.aspx).

## Objectives

In this post, we'll explore some interesting characteristics and trends of Indonesia's stock market. In particular, I will try to answer the following questions:

1. **What's the trends and characteristics of Indonesia's stock market in terms
   of number of stocks and total market cap?** Answering this question may help
   you get a general view of how Indonesia's stock market is evolving over
   time.
2. **What's the trends of foreign vs local investors' investment size in
   Indonesia's stock market?** Answering this question may reveal the level of
   confidence that international and domestic investors have in the Indonesian
   market and how it has changed over time.
3. **What's the trends of each investor category's investment in Indonesia's
   stock market?** Answering this question may provide valuable insights into
   the dynamics of the stock market. It may also help you decide how much or
   whether you should invest in Indonesia's stock market at all, especially if
   you're representing one of the investor category discussed here.
4. **How do foreign and local mutual funds allocate their portfolio into each
   sector?** If you already have a portfolio, you may compare your own sectoral
   allocation with that of the local and foreign mutual funds. You may also use
   it as a starting point if you're just beginning to invest.
5. **How concentrated is the aggregate holding of each investor category?** By
   answering this question, you may use it as a reference point regarding how
   much you should diversify your portfolio.
6. **How concentrated is Indonesia's stock market?** Understanding the stock
   market concentration is important to assess the stability of the stock market, since a
   highly concentrated market may be more volatile than a diversified one.

## Dataset

I used two data sources for this analysis: holding composition data from [KSEI](https://www.ksei.co.id/) (Kustodian Sentral Efek Indonesia) and stock sector classification from Bloomberg, which will be explained below.

### KSEI Equity Ownership Data

The ownership data are downloaded from <https://www.ksei.co.id/archive_download/holding_composition>. On that web page, you can download monthly capital market ownership data (including for stocks, bonds, ETFs, and so on). The data are released at every beginning of the month, based on the last trading day of the previous month's position. Here is an example of the data, based on October 2023's position:

|Date|Code|Type|Sec. Num|Price|Local IS|Local CP|Local PF|...|
|---|---|---|---|---|---|---|---|---|
|10/31/2023|AALI|EQUITY|1,924,688,333|7,050|65,013,727|27,880,124|7,443,655|...|
|10/31/2023|ABBA|EQUITY|3,935,892,857|73|107,406,600|2,341,806,805|0|...|
|10/31/2023|ABDA|EQUITY|620,806,680|6,000|0|40,248|0|...|
|…|…|…|…|…|…|…|…|…|

The data above contain the following columns:

- Date: reference date for the data
- Code: asset ticker
- Type: asset type
- Sec. Num: number of total shares outstanding (including public and non-public shares)
- Price: the security's last closing price in the month (in rupiah)
- Shares owned by each type of investor is placed in its own column (the definition of each abbreviation below is taken from [this KSEI documentation](https://www.ksei.co.id/Download/Panduan_Acuan_Informasi_Berdasarkan_Tipe_Investor_guna_Pembentukan_SID.PDF)):
	- IS: Insurance
	- CP: Corporate
	- PF: Pension Fund
	- IB: Financial Institution
	- ID: Individual
	- MF: Mutual Fund
	- SC: Securities Company
	- FD: Foundation
	- OT: Others

In addition, each investor category is also separated between "local" and "foreign" parties.

As an example, based on the last table, local insurance ("Local IS") owns 65,013,727 shares of AALI stock.

For this analysis, I have scraped the data going back as far as possible, which starts from March 2009, until the latest data as of the time of writing, which is October 2023. I have also transformed the data (with the help of Python) to a format that can be easily processed by Tableau. Among other things, I did the following operations on the data:

- Filter with the following criteria
    - Asset type equals to "EQUITY"
    - "Code" column must be a four-letter string (e.g. "BBCA")
    - Price must be greater than zero
- Un-pivot: convert each investor category column into its own rows.
- Calculate "Non-Public shares ownership" by subtracting total shares (under "Sec. Num" column in the original data) with the total of other investor categories, which, for the purpose of this analysis, is defined as "public ownership." This definition of "public ownership" may or may not be different from the concept of [free float](https://en.wikipedia.org/wiki/Public_float).
	- I found a problem when doing this: sum of shares of all categories could be larger than the total number of shares. In such cases, I will take the maximum of the two amounts as the total number of shares. For example, if the total number of shares under the "Sec. Num" column is 1 billion, while the sum of all investor categories is 1.5 billion, then I will take 1.5 billion as the total number of shares. Therefore, in that case, non-public ownership is zero.
- Outlier treatment: I found only one case of outlier: total number of shares of "BCIC" stock on February 2010 until September 2018 were too big. Based on manual inspection, I decided that the number need to be divided by 10 thousand.

Here's an example of the data after transformation:

|Date|Code|Price|Local/Foreign|Category|Shares Owned|
|---|---|---|---|---|---|
|3/31/2009|AALI|14100|Local|Insurance|38323000|
|3/31/2009|AALI|14100|Local|Corporate|9541429|
|3/31/2009|AALI|14100|Local|Pension Fund|13832600|
|3/31/2009|AALI|14100|Local|Financial Institution|66100|
|3/31/2009|AALI|14100|Local|Individual|47069158|
|3/31/2009|AALI|14100|Local|Mutual Fund|40032100|
|3/31/2009|AALI|14100|Local|Securities Company|5100261|
|3/31/2009|AALI|14100|Local|Foundation|1730000|
|3/31/2009|AALI|14100|Local|Others|3910004|
|3/31/2009|AALI|14100|Foreign|Insurance|0|
|3/31/2009|AALI|14100|Foreign|Corporate|5078177|
|3/31/2009|AALI|14100|Foreign|Pension Fund|737500|
|3/31/2009|AALI|14100|Foreign|Financial Institution|45338113|
|3/31/2009|AALI|14100|Foreign|Individual|350500|
|3/31/2009|AALI|14100|Foreign|Mutual Fund|1172000|
|3/31/2009|AALI|14100|Foreign|Securities Company|1000500|
|3/31/2009|AALI|14100|Foreign|Foundation|0|
|3/31/2009|AALI|14100|Foreign|Others|103579144|
|3/31/2009|AALI|14100||Non-Public|1257884414|
|…|…|…|…|…|…|

For the example above, we can calculate AALI's percentage of non-public ownership as follows: 1,257,884,414 shares (non-public shares) divided by 1,574,745,000 (sum of all the shares, including non-public), which equals to ~80%. That means AALI's public ownership is only 100% - 80% = 20%; the 80% was actually owned by PT Astra International Tbk, according to [this web page](https://www.idnfinancials.com/aali/pt-astra-agro-lestari-tbk#shareholders).

To calculate each investor category's ownership value (in rupiah), we only need to multiply the number of shares with the stock price. Going back to the last table, we can see that "Foreign Mutual Fund" ownership in AALI was equal to 1,172,000 x 14,100 = Rp16,525,200,000.

Here are some characteristics of the final data:

|Characteristic|Value|
|-|-|
|Number of stocks| 951 stocks|
|Earliest date| 3/31/2009|
|Latest date| 10/31/2023|
|Number of months| 176 months|
|Number of years| 15 years|

### Sector Data

In this analysis, I used Bloomberg Industry Classification System (BICS) to classify the sector of each stock. On a PC with Bloomberg, I retrieved the data using the following Excel formula: `=BQL(<stock ticker>,"bics_level_1_sector_name()")`. Here's an example of the output:

|Code|Sector|
|---|---|
|AALI|Consumer Staples|
|ABBA|Communications|
|ABDA|Financials|
|ACES|Consumer Discretionary|
|...|...|

Meanwhile, here's the complete list of all sectors:

1. Communications
1. Consumer Discretionary
1. Consumer Staples
1. Energy
1. Financials
1. Health Care
1. Industrials
1. Materials
1. Real Estate
1. Technology
1. Utilities

Note that the classified sectors are only based on the latest data, and I didn't account for changes in the stock's sector classification in the past (which is quite possible). Also, keep in mind that there are other alternative sector classification systems, such as the [Global Industry Classification Standard (GICS)](https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard) and the [Industry Classification Benchmark (ICB)](https://en.wikipedia.org/wiki/Industry_Classification_Benchmark), and each classification system might classify the stocks differently.

## Insights

In this section, we will look at the visualizations and hopefully gain some insights.

Note: I use [Tableau Public](https://public.tableau.com/app/discover) to make the visualizations in this post. And, since it allows the charts to be embedded on other websites, you can hover your mouse on each chart to get more details.

### Number of stocks and size of market cap over time

The number of stocks in Indonesia depends on how many stocks that went public or went private (i.e. delisted from the stock market) each year. If the number of companies that went public exceeded the number of companies that went private, the number of stocks will increase, and the vice versa is also true. Below is a chart depicting number of Indonesia's public companies each year:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/Num_ofStocks'
      hide-tabs>
    </tableau-viz>
</div>

For each year, I counted all stocks that exist in the data for any time in that year. Note that the KSEI data apparently still include stocks that were being suspended, which might explain why the number of stocks in the chart above are higher than the actual number of stocks that you can trade in the market.

As can be seen, the number of stocks consistently increased every year, starting from 407 stocks in 2009 to 913 stocks in 2023; an average increase of 36.14 stocks per year. We can also observe how the annual increase in the number of stocks started to pick up significantly in the year 2017. From 2010 to 2016, the increase in the number of stocks never went above 30, but starting in 2017 the increase was always greater than 30. There might be a structural reason why this is the case, but that is a discussion for another day.

Below is a chart depicting total market cap of Indonesia's stock market, along with the size of public and non-public ownership:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/PublicvsNon-Public'
      hide-tabs>
    </tableau-viz>
</div>

The changes in the total market cap are caused by market price fluctuations and stocks entering and leaving the market (i.e. companies that went public or went private). From the chart above we can gain several insights: (1) the total market cap is fluctuating every month, but in general it's increasing over time, (2) there was a large dip in total market cap in 2020-2021 period, around the time of the Covid-19 pandemic, which then eventually recovered, and (3) the size of public ownership appears to have increased faster than non-public ownership starting in 2021. We can see the ratio of total public ownership to total market cap over time in the chart below:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/Public'
      width='100%' height='350'
      hide-tabs>
    </tableau-viz>
</div>

As it turns out, the proportion of public to total ownership was decreasing from 2009 to 2018, but increasing from 2018 until present.

### Foreign vs local investment in the stock market

How has foreign ownership in Indonesia's stock market changed over time? Below is a chart that depicts ownership value of local and foreign investors in Indonesia's stock market:

<div> <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ForeignvsLocal'
      hide-tabs>
    </tableau-viz>
</div>

From the chart above, we can see that from 2009 to 2018 ownership of foreign investors had always been *higher* than local investors. However, during 2018-2020, local ownership started to catch up and finally exceeded foreign ownership. That dynamics between local and foreign ownership can also be seen in the following chart, which shows the ratio of foreign ownership to total public ownership:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/Foreign'
      hide-tabs>
    </tableau-viz>
</div>

As you can see, starting around 2020 the ratio has been below 50%, which means foreign ownership is less than local ownership. There's also one interesting phenomenon based on the chart above: in September 2013 foreign ownership ratio spiked from 56% to 63%, but then dropped again in November 2016.

### Investment of each investor category

How has the ownership of each investor category in Indonesia's stock market changed over time? To get a general sense of the answer, I have created stacked area charts below. The first is for local investors, while the second one is for foreign investors:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/LocalbyCategory'
      hide-tabs>
    </tableau-viz>
</div>

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ForeignbyCategory'
      hide-tabs>
    </tableau-viz>
</div>

From the above two charts we can sort of feel the position of each investor group over time. For instance, local investor ownership is dominated by the corporate and individual category, with the individual that shows a significant increase over time. Meanwhile, foreign investor ownership appears to be almost equally distributed between corporate, mutual fund, and others. The fact that the 'others' category is this big for foreign investors is also an interesting fact.

To see how the proportion of each investor category changes over time, I have also plotted the charts below:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/LocalbyCategory2'
      hide-tabs>
    </tableau-viz>
</div>

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ForeignbyCategory2'
      hide-tabs>
    </tableau-viz>
</div>

As of October 2023, the percentage of ownership for each investor category is as follows:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/LatestbyCategory?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link'
      hide-tabs>
    </tableau-viz>
</div>

Based on the chart above we can make some observations:

- Both local and foreign ownership are dominated by corporate investors.
- Local individuals have much larger investment than foreign individuals.
- Foreign mutual funds have much larger investment than local mutual funds.
- Local ownership is concentrated on corporate and individual categories, while foreign ownership is more spread out.

### Sectoral allocations for local and foreign mutual funds

In this section, we'll look at a particular investor category: local and foreign mutual funds. According to [Wikipedia](https://en.wikipedia.org/wiki/Mutual_fund), a mutual fund is an investment fund that pools money from many investors to purchase securities. Considering that some mutual funds are actively managed, their sector allocations should give us a glimpse into what sectors are favored over the past decade or so. You can see the chart below, which shows sectoral allocation of local mutual funds over time:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/LMFSectorWgt'
      hide-tabs>
    </tableau-viz>
</div>

As you can see, local mutual funds have large allocations for the financial sector, consumer staples, and materials sector. We can also observe how there appears to be no dramatic changes in the sectoral allocation since 2009, except, perhaps, in the materials sector, which decreased from around 30% to around 15%. The financial sector also seen gradual increase, from around 20% in 2009 to around 35% in 2023. Meanwhile, some sectors like technology and energy almost got no allocations (with weights of around 1% or less).

Below is the chart for foreign mutual funds:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/FMFSectorWgt'
      hide-tabs>
    </tableau-viz>
</div>

You can immediately see how different is the foreign mutual funds' sectoral allocations than local mutual funds'. For instance, the financial sector get a much bigger allocation (>50%), and it has been increasing over time. And, as the financial sector got larger allocation, sectors like utilities, industrials, and real estate got smaller allocations.

Here are the sectoral allocations for local and foreign mutual funds as of October 2023:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/LMFSectorWgtLatest'
      hide-tabs>
    </tableau-viz>
</div>

### Concentration level of each investor category

How concentrated is the aggregate holding of each investor category? To answer that question, we can use a metric known as "effective number of stocks" (ENS), and it's calculated as follows:

$$
ENS = \frac{1}{\sum_{i=1}^n w_i^2}
$$

Where $w_i$ is the weight of stock $i$ in a portfolio of $n$ stocks. This measure ranges from 1 (when the holding is very concentrated) to $n$ (when the holding is very diversified). In other words, the higher the ENS, the more diversified that portfolio is, and vice versa. For more information regarding this measure, you can see [this article](https://breakingdownfinance.com/finance-topics/finance-basics/effective-number-of-stocks/) and [this Wikipedia page](https://en.wikipedia.org/wiki/Effective_number_of_parties).

Below is a chart that shows the ENS for each investor category over time:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ENS'
      width='100%' height='1100px'
      hide-tabs>
    </tableau-viz>
</div>

Meanwhile, here's the ENS for each investor category by the end of October 2023:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ENSLatest'
      hide-tabs>
    </tableau-viz>
</div>

From the previous two charts, we can gain the following insights:

- Overall, the effective number of stocks of each investor category is relatively small, ranging from 0 to 30 stocks only (out of hundreds of stocks in the market), except for local securities company with 48.4 stocks.
- Local mutual funds are much more diversified than foreign mutual funds, as indicated by higher ENS (27.5 stocks vs 7.5 stocks in October 2023). Moreover, this phenomenon has always been true from 2009 until present. Our previous observation that foreign mutual funds are more concentrated in the financial sector also supports this finding. You may also recall that the foreign mutual funds' investment is much larger (about 5 times larger) than local mutual funds. In other words, foreign mutual funds have larger investment than local mutual funds, but their holding is much more concentrated. The reason for this phenomenon is probably due to limited number of stocks that they can invest in, which is constrained by criteria such as minimum market cap and minimum liquidity.
- Like mutual funds, we can also see that the ENS for insurance, foundation, and pension fund categories are always higher for local investors than foreign investors. However, the opposite is true for financial institution and others category, which has higher ENS for foreign investors.

### Concentration level of Indonesia's stock market

On the previous section we have seen the concentration level of each investor category's holding. In this section, we will examine the concentration level of Indonesia' stock market as a whole. Below I have plotted the ENS of the whole stock market:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/ENSOverall'
      width='100%' height='350'
      hide-tabs>
    </tableau-viz>
</div>

As you can see, the ENS of Indonesia's stock market is only about 30-40 stocks, which is much lower than the actual hundreds of stocks in the market. Moreover, this number is relatively stable over time, despite the increasing number of stocks each year. This indicates that the market cap is very concentrated.

To illustrate the point further, I have created the chart below, which shows each stock's market cap as the area of each rectangle:

<div>
    <tableau-viz id="tableauViz"       
      src='https://public.tableau.com/views/IndonesiasEquityMarketInvestors/TopMarketCap'
      hide-tabs>
    </tableau-viz>
</div>

Hopefully you can immediately see the disparities in the stock market cap from the chart above. Notice how very few stocks dominate the above chart's area: the nine largest stocks (from BBCA to ASII) are already enough to fill about 40-50% of the chart. In the future, it may be interesting to compare the concentration level of Indonesia's stock market vs other countries in the emerging market.

## Conclusion and Recommendation

In this post, we have seen some interesting characteristics and trends of Indonesia's stock market, particularly regarding its size, investor positions, sectoral allocations, and level of concentration. Hopefully you can get a better picture of the stock market and can make better investment decision after reading this article. I also encourage you to download the KSEI data to explore more insights and dig deeper information.

## References

- You can see the Tableau page for all visualizations I made [here](https://public.tableau.com/app/profile/abang.farhan/viz/IndonesiasEquityMarketInvestors/Num_ofStocks).
- KSEI holding composition data: <https://www.ksei.co.id/archive_download/holding_composition>
- More about effective number of stocks see <https://breakingdownfinance.com/finance-topics/finance-basics/effective-number-of-stocks/>. Also check out <https://en.wikipedia.org/wiki/Herfindahl-Hirschman_index> and <https://en.wikipedia.org/wiki/Effective_number_of_parties>
- About Bloomberg Industry Classification System see [this fact sheet](https://data.bloomberglp.com/professional/sites/10/Classification-Data-Fact-Sheet.pdf) or [this web page](https://www.bloomberg.com/professional/product/reference-data/).
