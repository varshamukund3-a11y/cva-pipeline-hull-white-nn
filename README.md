# CVA Pipeline: Hull-White Monte Carlo + Neural Network Surrogate Pricer

A counterparty credit risk pipeline that simulates interest rate paths, prices an interest rate swap under a stochastic short-rate model, and computes Expected Exposure (EE), Potential Future Exposure (PFE), and Credit Valuation Adjustment (CVA) — with a neural network trained as a fast surrogate for the analytical pricer.

## Motivation

CVA desks need to reprice trades across thousands of Monte Carlo paths and hundreds of time steps. Closed-form pricers exist for simple products (like the swap here), but for path-dependent or exotic trades, repeated analytical repricing becomes a computational bottleneck. This project demonstrates the standard workaround: train a neural network on the analytical pricer's input/output space, then swap it in as a fast surrogate inside the Monte Carlo exposure engine — and validate that the surrogate doesn't materially distort the resulting risk metrics.

## Pipeline

1. **Rate simulation** — 5,000 short-rate paths over 3 years simulated under the Hull-White one-factor model (mean reversion `a = 0.1`, volatility `σ = 0.015`, `r₀ = 3%`).
2. **Analytical pricer** — closed-form Hull-White zero-coupon bond price used to value a payer interest rate swap (fixed 3%, notional $10M) at any `(t, r_t)`.
3. **NN surrogate pricer** — a 3-layer MLP (128-128-64, ReLU) trained on 50,000 `(t, r) → price` samples from the analytical pricer, used to reprice the swap across every simulated path/time-step combination.
4. **Exposure & risk metrics** — Expected Exposure, 95% PFE, and CVA computed from both the analytical and NN-based exposure profiles, so accuracy loss (if any) from the surrogate can be measured directly in risk-metric terms, not just price terms.

## Results

| Metric | NN Pricer | Analytical |
|---|---:|---:|
| Peak Expected Exposure | $94,928 | $100,717 |
| Maximum 95% PFE | $398,474 | — |
| CVA | $1,093.91 | $1,193.08 |
| CVA Error | $99.16 | — |

**NN pricer accuracy on held-out test set:** R² = 0.9992, MAE = $10,782 (on swap values ranging roughly -$1.3M to +$2.0M).

The NN surrogate reproduces the shape of the exposure profile closely, including the sawtooth resets at each coupon date, with a modest underestimate of peak exposure near reset dates that shows up in the ~8% CVA gap. Full diagnostic charts are in [`CVA_with_NN.png`](CVA_with_NN.png).

## Assumptions & Known Limitations

- **Discounting:** CVA is discounted using `P(0,t)` implied by the Hull-White model at `r₀`, consistent with the same model used to simulate exposure (rather than an arbitrary flat rate).
- **Floating leg convention:** valued as `1 - P(t,T)`, which is exact at a reset date but assumes reset-to-par at the *next* payment date between resets — it ignores stub accrual since the last reset. Standard simplification for a single-curve pricer; would need a proper accrual schedule for production use.
- **Single-curve, no OIS/collateral discounting** — this is a simplified, unilateral CVA (no DVA, no collateral/CSA modeling).
- **Constant hazard rate** for default probability (`λ = 2%/year`), not calibrated to CDS spreads.
- Hull-White allows negative short rates by construction; a small share of simulated paths dip slightly negative, which is expected model behavior, not a bug.

## Repo Structure
├── CVA_with_NN.py       # main script: simulation, pricer, NN, exposure, CVA, plots
├── CVA_with_NN.png       # output figure (4-panel diagnostic chart)
├── requirements.txt
└── README.md
## How to Run

```bash
pip install -r requirements.txt
python CVA_with_NN.py
```

Outputs a 4-panel figure (`CVA_with_NN.png`) comparing NN vs. analytical Expected Exposure, NN exposure profiles with PFE, NN pricer accuracy on a held-out test set, and sample simulated short-rate paths — plus a printed summary of peak exposure, PFE, and CVA for both pricers.

## Possible Extensions

- Extend the NN surrogate to a portfolio of trades (netting sets) rather than a single swap
- Add wrong-way risk by correlating the hazard rate with the short rate
- Calibrate the hazard curve to market CDS spreads instead of a flat assumption
- Benchmark NN vs. analytical pricer **speed** directly (repricing calls/sec) to quantify the computational case for the surrogate, not just accuracy

