# Interim Summary – AI Lab I-II

## Project
Constraint-Aware Emotion Detection (GoEmotions)

## Comparative Summary of Main Configurations

| Setup | Threshold | Micro-F1 | Macro-F1 | Tail Recall | Notes |
|---|---:|---:|---:|---:|---|
| Baseline (2000, 1 epoch) | 0.50 | 0.0000 | 0.0000 | 0.0000 | Initial failure case |
| Baseline (5000, 3 epochs) | 0.20 | 0.5348 | 0.3106 | 0.0986 | Best overall Micro-F1 for plain baseline |
| Baseline (5000, 3 epochs) | 0.05 | 0.3533 | 0.2657 | 0.3029 | Tail-oriented threshold |
| Oversampling | 0.25 | 0.5573 | 0.3555 | 0.2464 | Best Micro-F1 observed with oversampling |
| Weighted loss | 0.70 | 0.4560 | 0.4219 | 0.5714 | Strong tail-focused setup |
| Weighted-only seed summary | 0.70 | 0.4496 ± 0.0118 | 0.4231 ± 0.0049 | 0.6267 ± 0.0750 | Mean ± std across 3 runs |
| Oversampling + weighted loss | 0.70 | 0.4579 | 0.4302 | 0.7216 | Best overall trade-off |
| OW + per-class threshold | default=0.7, tail=0.6 | 0.4518 | 0.4163 | 0.7312 | Slightly higher tail recall, slightly lower overall performance |
| OW + held-out calibration | 0.70 | 0.4425 | 0.4358 | 0.8472 | Threshold selected on calibration half |
| OW + seed repeat (seed=123) | 0.70 | 0.4690 | 0.4289 | 0.6904 | Stability check |
| OW + seed repeat (seed=7) | 0.70 | 0.4747 | 0.4522 | 0.7060 | Stability check |
| OW + seed repeat (seed=42) | 0.70 | 0.4578 | 0.4361 | 0.7216 | Stability check |
| OW seed summary | 0.70 | 0.4648 ± 0.0084 | 0.4369 ± 0.0107 | 0.7099 ± 0.0149 | Mean ± std across 4 runs |

## Main Findings So Far

- Threshold selection is important, but threshold tuning alone is not sufficient for strong tail-label performance.
- Small training subsets lead to unstable tail-label behavior because several rare labels have extremely low support.
- Increasing the training subset size substantially improves the baseline.
- Oversampling improves tail recall while preserving strong overall performance.
- Class-weighted loss strongly improves tail sensitivity, but shifts the optimal threshold into a higher range.
- The combination of oversampling and class-weighted loss currently provides the best overall trade-off between Macro-F1 and tail recall.
- Per-class thresholding adds only marginal gains beyond a well-chosen global threshold.
- Threshold calibration appears stable across different validation splits.
- The current best setup also appears reasonably robust across random seeds.
- Tail labels are heterogeneous: some rare categories are learned relatively well, while others remain difficult.
- The weighted-only configuration remains a strong comparison point, but the current evidence suggests that OW is both stronger on average and more stable, especially in tail recall.

## Current Best Configuration

- Model: DistilBERT
- Training strategy: oversampling + class-weighted loss
- Decision threshold: 0.7
- Current multi-seed summary:
  - Micro-F1: 0.4648 ± 0.0084
  - Macro-F1: 0.4369 ± 0.0107
  - Tail Recall: 0.7099 ± 0.0149

## Additional Tail-Focused Findings

### Tail-label heterogeneity
Some rare labels are learned relatively well:
- pride
- remorse
- fear

Some remain difficult or unstable:
- grief
- relief

This suggests that rarity alone does not fully explain difficulty.

### Tail-oriented threshold selection
Selecting the threshold by Tail-F1 still yields the same optimum (0.7) as selection by Macro-F1.

Held-out result in this setting:
- Micro-F1: 0.4425
- Macro-F1: 0.4358
- Tail Precision: 0.4044
- Tail Recall: 0.8472
- Tail F1: 0.5475

## Open Questions

- How stable are the results across more random seeds?
- How much of the tail-label behavior is still influenced by the small validation subset?
- Would a larger evaluation slice change the ranking of the best setups?
- Can calibration or milder weighting further improve the precision–recall balance?

## Suggested Next Steps

- Focus primarily on the OW configuration as the main experimental direction.
- Keep weighted-only as the main comparison / control setup.
- Extend the multi-seed evaluation further only for the strongest 1–2 configurations.
- Prepare a cleaner comparison of the final candidate setups for later write-up.