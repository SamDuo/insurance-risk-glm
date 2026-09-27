# Decision memo: next auto rating model

**To:** Pricing committee (portfolio exercise on public data)
**From:** Sam Duong
**Decision needed:** Replace the rating GLM with the gradient-boosted challenger?

## Recommendation

**Adopt GLM+ now, and do not deploy the gradient-boosted model as the rating plan.** Keep it
as the standing challenger, and bring its next-strongest pattern into the GLM next cycle.

## Why

1. **The challenger ranks risk better, but at a price the book cannot absorb.** Its Gini lead
   over the current GLM is +0.055 (95% CI +0.018 to +0.088), measured on 135,811 held-out
   policies with the uncertainty computed over whole risk groups. Switching would move 74% of
   premiums by more than 10%, and 40% by more than 25%. Its severity model is also worse than the
   GLM's, and a tree ensemble is much harder to file and explain.
2. **GLM+ keeps most of the explainability and gets a real share of the gain.** It uses
   bonus-malus bands plus three indicators (older driver above the bonus-malus floor ×1.43,
   mature driver at the entry level ×0.73, new brand-B12 vehicle ×0.79). Gini +0.013
   (CI +0.005 to +0.023), AIC lower by 851, and on the double-lift chart it tracks actual
   losses where it disagrees with the GLM. It moves 12% of premiums by more than 25%, which a
   capped phase-in can handle.
3. **The gap that remains tells us where to look.** The challenger still leads GLM+ by 0.042
   Gini. Its next-strongest interaction is bonus-malus × region, which GLM+ does not contain yet.

## Before filing

- **Phase in the dislocation.** Cap individual changes (for example ±15% per renewal) and
  report the policies affected.
- **Recheck the level.** All three models predicted 9.5% more loss than the holdout produced,
  because the holdout had fewer large claims than training. It is the same for every model, but
  the rate level should be set on full-book experience, not on a 20% holdout.
- **Confirm the claim definition.** Pricing uses paid claims because 9,116 policies report
  claims with no payment. Confirm with claims operations that these closed at zero cost.
  Mixing reported claims with paid severity overstated losses by 48%.

## How we know

A Poisson × Gamma GLM fitted in PySpark, with coefficients matched to statsmodels within
4×10⁻¹¹. It was compared with XGBoost on a holdout split by risk group, since 38% of policies
share a profile with another. Full method and charts are in the [README](README.md).
