---
layout: post
title: Bond Effective Interest Calculator
category: blog
tags: [finance,math]
---

<label for="face_value">Face value:</label>
<input type="text" id="face_value" name="face_value" value="100000000"/>
<br/>
<label for="market_value">Market value:</label>
<input type="text" id="market_value" name="market_value" value="113592966"/>
<br/>
<label for="coupon_rate">Annual coupon rate:</label>
<input type="text" id="coupon_rate" name="coupon_rate" value="0.12"/>
<br/>
<label for="settlement_date">Settlement date:</label>
<input type="date" id="settlement_date" name="settlement_date" value="2024-01-19"/>
<br/>
<label for="maturity_date">Maturity date:</label>
<input type="date" id="maturity_date" name="maturity_date" value="2026-09-15"/>
<br/>
<input type="button" onclick="calculate_EIR()" value="Run"/>

Answer: <span id="eir_output"/>

<script>
function calculate_EIR() {
  const days_in_year = 365
  const fv = parseFloat(face_value.value)
  const mv = parseFloat(market_value.value)
  const cr = parseFloat(coupon_rate.value) / days_in_year
  const daily_coupon = cr * fv
  const settlement = new Date(settlement_date.value)
  const maturity = new Date(maturity_date.value)
  const one_day = 24 * 60 * 60 * 1000
  const days = Math.floor((maturity - settlement) / one_day)

  // TODO validate user input
  // TODO show message if solution if found or not found
  const max_iter = 200
  const tol = 1e-4
  let alpha = 1 + cr
  for(let i=0; i<100; i++) {
    const f = fn(fv, mv, alpha, daily_coupon, days)
    const df = dfn(mv, alpha, daily_coupon, days)
    alpha = alpha - f / df
    if(Math.abs(f) < tol) {
      break
    }
  }
  console.log(fn0(mv, alpha, daily_coupon, days))
  const EIR = (alpha - 1) * days_in_year
  eir_output.innerText = EIR
}

function fn0(mv, alpha, daily_coupon, days) {
  return Math.pow(alpha, days) * mv
    - daily_coupon * (1 - Math.pow(alpha, days)) / (1 - alpha)
}

function fn(fv, mv, alpha, daily_coupon, days) {
  return fn0(mv, alpha, daily_coupon, days) - fv
}

function dfn(mv, alpha, daily_coupon, days) {
  return Math.pow(alpha, days - 1) * mv * days
    - daily_coupon * (1 - Math.pow(alpha, days - 1)) / (1 - alpha * alpha)
}
</script>
