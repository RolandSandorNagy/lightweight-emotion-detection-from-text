# Research Log – AI Lab I-II

## Project
Constraint-Aware Emotion Detection (GoEmotions)

---

## Experiment 0 – DistilBERT single-label smoke test

**Date:** 2026-04-01

### Goal
Verify that the Hugging Face training pipeline works end-to-end on a simplified single-label setup.

### Setup
- Dataset: GoEmotions (simplified)
- Filtering: only single-label samples
- Model: distilbert-base-uncased
- Task: single-label classification
- Train size: 1000
- Validation size: 200
- Epochs: 1
- Batch size: 16

### Metrics
- Accuracy
- Macro-F1

### Results
- Accuracy: not recorded
- Macro-F1: not recorded

### Observations
- The training and evaluation pipeline ran successfully end-to-end.
- The setup was useful as an initial sanity check for tokenization, model loading, training, and evaluation.
- This experiment was intentionally simplified and does not match the final research setting.

### Conclusion
This experiment served as a sanity check only. The setup is not aligned with the final multi-label research task.

---

## Experiment 1 – DistilBERT multi-label baseline (initial run)

**Date:** 2026-04-03

### Goal
Establish the first multi-label baseline aligned with the main research objective.

### Setup
- Dataset: GoEmotions (full 28-label multi-label version)
- Model: distilbert-base-uncased
- Task: multi-label classification
- Label encoding: multi-hot vectors
- Loss: BCEWithLogitsLoss
- Output interpretation: sigmoid + fixed threshold 0.5
- Tail labels:
  - grief (16)
  - pride (21)
  - relief (23)
  - nervousness (19)
  - embarrassment (12)
  - remorse (24)
  - fear (14)
  - desire (8)
- Train size: 2000
- Validation size: 500
- Epochs: 1
- Batch size: 16

### Metrics
- Micro-F1
- Macro-F1
- Tail Recall

### Results
- Micro-F1: 0.0
- Macro-F1: 0.0
- Tail Recall: 0.0
- Training loss: 0.17172127962112427
- Eval loss: 0.2711769943237305
- Max predicted probability on validation set: 0.33828905
- Mean predicted probability on validation set: 0.07850569

### Observations
- The multi-label training and evaluation pipeline ran successfully end-to-end.
- Training loss decreased during training, suggesting that optimization itself was functioning.
- The all-zero classification metrics were caused by the fixed threshold of 0.5, not by an implementation bug.
- Validation probabilities remained well below 0.5 after 1 epoch on the small subset.
- This indicates that the default threshold is too strict for this early baseline setting.

### Conclusion
This is the first successful end-to-end multi-label baseline run. The result shows that fixed thresholding at 0.5 is unsuitable in this setting, and threshold selection is likely to be a critical part of the research problem.

---

## Experiment 2 – Threshold sweep on initial multi-label baseline

**Date:** 2026-04-03

### Goal
Test how sensitive the initial multi-label baseline is to threshold selection, especially with respect to tail-label recall.

### Setup
- Same trained model as in Experiment 1
- No retraining
- Validation set: 500 samples
- Thresholds tested: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30

### Metrics
- Micro-F1
- Macro-F1
- Tail Recall

### Results
- Threshold 0.05:
  - Micro-F1: 0.0881
  - Macro-F1: 0.0713
  - Tail Recall: 0.8750
- Threshold 0.10:
  - Micro-F1: 0.1929
  - Macro-F1: 0.0247
  - Tail Recall: 0.0000
- Threshold 0.15:
  - Micro-F1: 0.2844
  - Macro-F1: 0.0171
  - Tail Recall: 0.0000
- Threshold 0.20:
  - Micro-F1: 0.2844
  - Macro-F1: 0.0171
  - Tail Recall: 0.0000
- Threshold 0.25:
  - Micro-F1: 0.2844
  - Macro-F1: 0.0171
  - Tail Recall: 0.0000
- Threshold 0.30:
  - Micro-F1: 0.2844
  - Macro-F1: 0.0171
  - Tail Recall: 0.0000

### Observations
- Threshold selection has a very strong effect on the resulting metrics.
- A very low threshold (0.05) dramatically improves tail recall.
- However, this comes with lower Micro-F1.
- For thresholds 0.10 and above, tail recall collapses to 0.0.
- This suggests that rare-label detection is highly sensitive to decision thresholding in the current baseline setting.

### Conclusion
Thresholding is not a minor implementation detail in this task, but a central part of model behavior. In this initial baseline, lower thresholds recover rare-label recall, while higher thresholds favor overall Micro-F1 at the cost of missing tail labels completely.

---

## Experiment 3 – Fine-grained threshold sweep

**Date:** 2026-04-03

### Goal
Identify a more precise threshold region where tail recall can be preserved while improving overall performance.

### Setup
- Same trained model as previous experiments
- No retraining
- Thresholds tested: 0.05–0.10 (step 0.01)

### Results
- Threshold 0.05:
  - Micro-F1: 0.0881
  - Macro-F1: 0.0713
  - Tail Recall: 0.8750
- Threshold 0.06:
  - Micro-F1: 0.0964
  - Macro-F1: 0.0652
  - Tail Recall: 0.5000
- Threshold 0.07:
  - Micro-F1: 0.1279
  - Macro-F1: 0.0572
  - Tail Recall: 0.0000
- Threshold 0.08:
  - Micro-F1: 0.1584
  - Macro-F1: 0.0450
  - Tail Recall: 0.0000
- Threshold 0.09:
  - Micro-F1: 0.1836
  - Macro-F1: 0.0342
  - Tail Recall: 0.0000
- Threshold 0.10:
  - Micro-F1: 0.1929
  - Macro-F1: 0.0247
  - Tail Recall: 0.0000

### Observations
- Tail recall exhibits a sharp drop between thresholds 0.06 and 0.07.
- Even a small increase in threshold can completely eliminate detection of rare emotion labels.
- Lower thresholds significantly improve tail recall but reduce Micro-F1.
- The model outputs for tail labels are clustered in a very low probability range.

### Conclusion
The results confirm that threshold selection is a highly sensitive and critical component in multi-label emotion detection. Small threshold changes can drastically affect rare-label performance, suggesting that fixed global thresholds are suboptimal.

---

## Experiment 4 – Per-class thresholding

**Date:** 2026-04-03

### Goal
Test whether assigning different thresholds to tail and non-tail labels improves the trade-off between overall performance and rare-label recall.

### Setup
- Same trained model as previous experiments
- Tail labels: threshold = 0.05
- Non-tail labels: threshold = 0.1

### Results
- Micro-F1: 0.0860
- Macro-F1: 0.0300
- Tail Recall: 0.8750

### Observations
- Tail recall remains high, matching the performance of the global threshold = 0.05 setting.
- However, Micro-F1 does not improve compared to the low global threshold baseline.
- This suggests that non-tail label predictions are also low-confidence, and a threshold of 0.1 is still too strict for many labels.

### Conclusion
Per-class thresholding alone is not sufficient to improve the balance between overall performance and tail recall in this setting. The issue appears to be related to generally low prediction confidence rather than thresholding alone.

---

## Experiment 5 – DistilBERT multi-label baseline (3 epochs)

**Date:** 2026-04-03

### Goal
Test whether longer training improves probability calibration and reduces extreme threshold sensitivity.

### Setup
- Dataset: GoEmotions (full 28-label multi-label version)
- Model: distilbert-base-uncased
- Task: multi-label classification
- Label encoding: multi-hot vectors
- Loss: BCEWithLogitsLoss
- Train size: 2000
- Validation size: 500
- Epochs: 3
- Batch size: 16
- Tail labels:
  - grief (16)
  - pride (21)
  - relief (23)
  - nervousness (19)
  - embarrassment (12)
  - remorse (24)
  - fear (14)
  - desire (8)

### Results
- Eval loss: 0.14661844074726105
- Eval Micro-F1 (threshold = 0.5): 0.0510
- Eval Macro-F1 (threshold = 0.5): 0.0063
- Eval Tail Recall (threshold = 0.5): 0.0000
- Max predicted probability on validation set: 0.527259
- Mean predicted probability on validation set: 0.064281803

### Threshold sweep
- Threshold 0.05:
  - Micro-F1: 0.1826
  - Macro-F1: 0.0737
  - Tail Recall: 0.0000
- Threshold 0.10:
  - Micro-F1: 0.2933
  - Macro-F1: 0.0510
  - Tail Recall: 0.0000
- Threshold 0.15:
  - Micro-F1: 0.3557
  - Macro-F1: 0.0521
  - Tail Recall: 0.0000
- Threshold 0.20:
  - Micro-F1: 0.3519
  - Macro-F1: 0.0503
  - Tail Recall: 0.0000
- Threshold 0.25:
  - Micro-F1: 0.3447
  - Macro-F1: 0.0510
  - Tail Recall: 0.0000
- Threshold 0.30:
  - Micro-F1: 0.3283
  - Macro-F1: 0.0528
  - Tail Recall: 0.0000

### Fine-grained threshold sweep
- Threshold 0.05:
  - Micro-F1: 0.1826
  - Macro-F1: 0.0737
  - Tail Recall: 0.0000
- Threshold 0.06:
  - Micro-F1: 0.2070
  - Macro-F1: 0.0684
  - Tail Recall: 0.0000
- Threshold 0.07:
  - Micro-F1: 0.2450
  - Macro-F1: 0.0608
  - Tail Recall: 0.0000
- Threshold 0.08:
  - Micro-F1: 0.2696
  - Macro-F1: 0.0573
  - Tail Recall: 0.0000
- Threshold 0.09:
  - Micro-F1: 0.2856
  - Macro-F1: 0.0553
  - Tail Recall: 0.0000
- Threshold 0.10:
  - Micro-F1: 0.2933
  - Macro-F1: 0.0510
  - Tail Recall: 0.0000

### Per-class thresholding
- Tail labels: threshold = 0.05
- Non-tail labels: threshold = 0.10
- Micro-F1: 0.2931
- Macro-F1: 0.0510
- Tail Recall: 0.0000

### Observations
- Increasing the number of epochs substantially improved Micro-F1 compared to the 1-epoch run.
- The model now produces higher-confidence outputs overall.
- However, tail recall remains 0.0 across all tested threshold settings.
- This indicates that the current baseline does not learn the rare labels sufficiently.
- Threshold tuning alone is not enough to recover tail-label performance in this setting.

### Conclusion
Longer training improves overall performance but does not solve rare-label detection. The next step should focus on tail-aware learning rather than thresholding alone.

---

## Experiment 6 – Tail label support analysis

### Goal
Verify whether the lack of tail-label performance is due to insufficient data in the sampled subsets.

### Results
Train subset (2000 samples):
- grief: 7
- pride: 3
- relief: 8
- nervousness: 6
- embarrassment: 10
- remorse: 21
- fear: 21
- desire: 26

Validation subset (500 samples):
- grief: 1
- pride: 4
- relief: 1
- nervousness: 2
- embarrassment: 5
- remorse: 13
- fear: 8
- desire: 8

### Observations
- Several tail labels have extremely low support in both training and validation subsets.
- Some labels appear only 1–2 times in validation, making recall highly unstable.

### Conclusion
The current subset size is insufficient for reliable evaluation of tail-label performance.

---

## Experiment 7 – DistilBERT multi-label baseline with larger training subset

**Date:** 2026-04-03

### Goal
Test whether increasing the training subset size improves rare-label learning and reduces the extreme instability observed in earlier runs.

### Setup
- Dataset: GoEmotions (full 28-label multi-label version)
- Model: distilbert-base-uncased
- Task: multi-label classification
- Label encoding: multi-hot vectors
- Loss: BCEWithLogitsLoss
- Train size: 5000
- Validation size: 500
- Epochs: 3
- Batch size: 16
- Tail labels:
  - grief (16)
  - pride (21)
  - relief (23)
  - nervousness (19)
  - embarrassment (12)
  - remorse (24)
  - fear (14)
  - desire (8)

### Eval results at default threshold 0.5
- Eval loss: 0.1073664128780365
- Eval Micro-F1: 0.4208605482717521
- Eval Macro-F1: 0.14515260309074549
- Eval Tail Recall: 0.0
- Max predicted probability on validation set: 0.91006434
- Mean predicted probability on validation set: 0.04407998

### Threshold sweep
- Threshold 0.05:
  - Micro-F1: 0.3533
  - Macro-F1: 0.2657
  - Tail Recall: 0.3029
- Threshold 0.10:
  - Micro-F1: 0.4831
  - Macro-F1: 0.3388
  - Tail Recall: 0.1839
- Threshold 0.15:
  - Micro-F1: 0.5196
  - Macro-F1: 0.3102
  - Tail Recall: 0.0986
- Threshold 0.20:
  - Micro-F1: 0.5348
  - Macro-F1: 0.3106
  - Tail Recall: 0.0986
- Threshold 0.25:
  - Micro-F1: 0.5302
  - Macro-F1: 0.2844
  - Tail Recall: 0.0481
- Threshold 0.30:
  - Micro-F1: 0.5077
  - Macro-F1: 0.2449
  - Tail Recall: 0.0000

### Fine-grained threshold sweep
- Threshold 0.05:
  - Micro-F1: 0.3533
  - Macro-F1: 0.2657
  - Tail Recall: 0.3029
- Threshold 0.06:
  - Micro-F1: 0.3923
  - Macro-F1: 0.2955
  - Tail Recall: 0.2716
- Threshold 0.07:
  - Micro-F1: 0.4266
  - Macro-F1: 0.3189
  - Tail Recall: 0.2620
- Threshold 0.08:
  - Micro-F1: 0.4452
  - Macro-F1: 0.3232
  - Tail Recall: 0.2151
- Threshold 0.09:
  - Micro-F1: 0.4661
  - Macro-F1: 0.3299
  - Tail Recall: 0.1839
- Threshold 0.10:
  - Micro-F1: 0.4831
  - Macro-F1: 0.3388
  - Tail Recall: 0.1839

### Per-class thresholding
- Tail labels: threshold = 0.05
- Non-tail labels: threshold = 0.10
- Micro-F1: 0.4818
- Macro-F1: 0.3373
- Tail Recall: 0.3029

### Tail label support
Train subset (5000 samples):
- grief: 12
- pride: 9
- relief: 18
- nervousness: 23
- embarrassment: 28
- remorse: 69
- fear: 67
- desire: 78

Validation subset (500 samples):
- grief: 1
- pride: 4
- relief: 1
- nervousness: 2
- embarrassment: 5
- remorse: 13
- fear: 8
- desire: 8

### Observations
- Increasing training subset size substantially improved both Micro-F1 and Macro-F1.
- The model now produces much higher-confidence outputs than in earlier runs.
- Tail recall is no longer uniformly zero once lower thresholds are used.
- The results show a clear trade-off between overall performance and rare-label recall.
- Per-class thresholding improves tail recall substantially while preserving relatively strong overall performance.
- Extremely low support for some tail labels in the validation subset still makes tail evaluation noisy.

### Conclusion
The earlier tail-label failure was not solely a thresholding issue, but was strongly affected by insufficient training support. With a larger training subset, thresholding becomes meaningfully effective, and per-class thresholding emerges as a promising strategy for balancing overall performance and rare-label recall.

---

## Experiment 8 – Tail-example oversampling

**Date:** 2026-04-03

### Goal
Test whether oversampling tail-label examples improves rare emotion detection.

### Setup
- Base training subset: 5000 samples
- Tail-containing examples duplicated once (2× frequency)
- Oversampled train size: 5295
- Same model and training setup as Experiment 7

### Eval results at default threshold 0.5
- Eval loss: 0.10371033847332001
- Eval Micro-F1: 0.4400
- Eval Macro-F1: 0.2177
- Eval Tail Recall: 0.1647
- Max predicted probability on validation set: 0.9287127
- Mean predicted probability on validation set: 0.04417675

### Results (threshold sweep)

- Threshold 0.05:
  - Micro-F1: 0.3625
  - Macro-F1: 0.2935
  - Tail Recall: 0.4938

- Threshold 0.10:
  - Micro-F1: 0.4968
  - Macro-F1: 0.3525
  - Tail Recall: 0.2969

- Threshold 0.15:
  - Micro-F1: 0.5375
  - Macro-F1: 0.3632
  - Tail Recall: 0.2812

- Threshold 0.20:
  - Micro-F1: 0.5497
  - Macro-F1: 0.3612
  - Tail Recall: 0.2464

- Threshold 0.25:
  - Micro-F1: 0.5573
  - Macro-F1: 0.3555
  - Tail Recall: 0.2464

- Threshold 0.30:
  - Micro-F1: 0.5470
  - Macro-F1: 0.3346
  - Tail Recall: 0.2464
  
### Observations
- Oversampling significantly improves tail recall across all thresholds.
- Unlike earlier experiments, tail recall remains non-zero even at higher thresholds.
- Overall performance (Micro-F1) also improves slightly.
- This suggests that oversampling improves the learned representation of rare labels, not just threshold sensitivity.

### Conclusion
Oversampling is an effective and simple strategy for improving rare-label detection, outperforming threshold tuning alone.

---

## Experiment 9 – Tail precision / recall / F1 analysis on the 5000-sample baseline

**Date:** 2026-04-03

### Goal
Evaluate whether the improved tail recall at low threshold corresponds to meaningful rare-label detection or only to excessive false positives.

### Setup
- Same trained model as in Experiment 7
- No retraining
- Dataset setting: 5000-sample training subset, no oversampling
- Evaluation threshold: 0.05
- Tail labels:
  - grief (16)
  - pride (21)
  - relief (23)
  - nervousness (19)
  - embarrassment (12)
  - remorse (24)
  - fear (14)
  - desire (8)

### Results
- Tail Precision: 0.2144
- Tail Recall: 0.2873
- Tail F1: 0.1228

### Observations
- Low-threshold prediction improves tail recall, but precision remains limited.
- However, tail precision does not collapse completely, suggesting that the model is learning at least some meaningful signal for rare labels.
- This indicates that the observed tail recall is not purely a threshold artifact.

### Conclusion
The 5000-sample baseline achieves non-trivial tail-label detection at low threshold, but the precision-recall trade-off remains substantial. Further improvement likely requires tail-aware learning beyond threshold selection alone.

---

## Experiment 10 – Class-weighted loss on the 5000-sample training subset

**Date:** 2026-04-10

### Goal
Test whether class-weighted loss improves rare-label detection more effectively than thresholding or oversampling.

### Setup
- Dataset: GoEmotions (full 28-label multi-label version)
- Model: distilbert-base-uncased
- Task: multi-label classification
- Label encoding: multi-hot vectors
- Train size: 5000
- Validation size: 500
- Epochs: 3
- Batch size: 16
- Loss: BCEWithLogitsLoss with class-dependent positive weights
- Tail labels:
  - grief (16)
  - pride (21)
  - relief (23)
  - nervousness (19)
  - embarrassment (12)
  - remorse (24)
  - fear (14)
  - desire (8)

### Eval results at default threshold 0.5
- Eval loss: 0.9538329839706421
- Eval Micro-F1: 0.3719
- Eval Macro-F1: 0.3455
- Eval Tail Recall: 0.7156
- Max predicted probability on validation set: 0.9778
- Mean predicted probability on validation set: 0.2265

### Threshold sweep
- Threshold 0.05:
  - Micro-F1: 0.1014
  - Macro-F1: 0.0884
  - Tail Recall: 0.9688
- Threshold 0.10:
  - Micro-F1: 0.1329
  - Macro-F1: 0.1123
  - Tail Recall: 0.9281
- Threshold 0.15:
  - Micro-F1: 0.1637
  - Macro-F1: 0.1402
  - Tail Recall: 0.8969
- Threshold 0.20:
  - Micro-F1: 0.1923
  - Macro-F1: 0.1664
  - Tail Recall: 0.7469
- Threshold 0.25:
  - Micro-F1: 0.2224
  - Macro-F1: 0.1975
  - Tail Recall: 0.7469
- Threshold 0.30:
  - Micro-F1: 0.2555
  - Macro-F1: 0.2341
  - Tail Recall: 0.7469

### Observations
- Class-weighted loss strongly increases tail-label recall.
- However, this comes at a substantial cost in Micro-F1 compared to the oversampling-based runs.
- The model outputs are much more positive overall than in previous experiments, suggesting overcorrection toward rare labels.
- This indicates that weighting is effective for tail sensitivity, but currently too aggressive for balanced performance.

### Conclusion
Class-weighted loss successfully improves rare-label recall, but in its current form it overcompensates and degrades overall performance. Compared with oversampling, it appears less balanced, even though it achieves much higher tail recall.

---

## Experiment 11 – High-threshold analysis on class-weighted model

### Goal
Investigate whether higher thresholds improve the balance between overall performance and tail recall for the class-weighted model.

### Setup
- Same trained model as in Experiment 10
- No retraining
- Thresholds tested: 0.35–0.7

### Results (selected)

- Threshold 0.5:
  - Micro-F1: 0.3719
  - Macro-F1: 0.3455
  - Tail Recall: 0.7156

- Threshold 0.6:
  - Micro-F1: 0.4228
  - Macro-F1: 0.3881
  - Tail Recall: 0.5810

- Threshold 0.7:
  - Micro-F1: 0.4560
  - Macro-F1: 0.4219
  - Tail Recall: 0.5714

### Observations
- Increasing the threshold significantly improves Micro-F1 and Macro-F1.
- Tail recall remains high even at relatively large thresholds.
- The optimal threshold for the weighted model is substantially higher than for the baseline models.
- This indicates that class-weighted training shifts the calibration of model outputs.

### Conclusion
Contrary to initial expectations, the class-weighted model is not inherently worse than the oversampling-based approach. When evaluated with appropriate (higher) thresholds, it achieves substantially better macro-F1 and tail recall, while maintaining competitive micro-F1. This suggests that class-weighted loss is a strong candidate for tail-focused scenarios.

---

## Experiment 12 – Oversampling + class-weighted loss

**Date:** 2026-04-10

### Goal
Test whether combining oversampling with class-weighted loss further improves rare-label detection compared to either method alone.

### Setup
- Base training subset: 5000 samples
- Tail-containing examples duplicated once (oversampling)
- Oversampled train size: 5295
- Model: distilbert-base-uncased
- Task: multi-label classification
- Loss: BCEWithLogitsLoss with class-dependent positive weights
- Epochs: 3
- Batch size: 16

### Eval results (threshold = 0.5)
- Micro-F1: 0.3924
- Macro-F1: 0.3686
- Tail Recall: 0.7562

### Threshold sweep (selected)

- Threshold 0.5:
  - Micro-F1: 0.3924
  - Macro-F1: 0.3686
  - Tail Recall: 0.7562

- Threshold 0.6:
  - Micro-F1: 0.4322
  - Macro-F1: 0.3982
  - Tail Recall: 0.7312

- Threshold 0.7:
  - Micro-F1: 0.4579
  - Macro-F1: 0.4302
  - Tail Recall: 0.7216

### Observations
- Combining oversampling with weighted loss produces the highest macro-F1 observed so far.
- Tail recall remains very high even at larger thresholds.
- Compared to weighted loss alone, the combination slightly improves stability and overall performance.
- The model still requires higher thresholds for optimal performance.

### Conclusion
Oversampling and class-weighted loss are complementary techniques. Their combination yields the strongest performance for rare-label detection, achieving the best macro-F1 and tail recall among all tested methods.

---

## Experiment 13 – Per-class thresholding on oversampling + weighted model

**Date:** 2026-04-10

### Goal
Evaluate whether per-class thresholding further improves performance compared to a global threshold on the best-performing model.

### Setup
- Model: oversampling + class-weighted loss (Experiment 12)
- No retraining
- Default threshold for non-tail labels: 0.6–0.7
- Lower thresholds for tail labels: 0.5–0.6

### Results

- default=0.7, tail=0.5:
  - Micro-F1: 0.4458
  - Macro-F1: 0.4087
  - Tail Recall: 0.7562

- default=0.7, tail=0.6:
  - Micro-F1: 0.4518
  - Macro-F1: 0.4163
  - Tail Recall: 0.7312

- default=0.6, tail=0.5:
  - Micro-F1: 0.4276
  - Macro-F1: 0.3905
  - Tail Recall: 0.7562

### Observations
- Per-class thresholding slightly increases tail recall.
- However, this comes with a small decrease in macro-F1 and micro-F1.
- The best global threshold setting remains competitive with or better than per-class configurations.

### Conclusion
Per-class thresholding provides only marginal gains over a well-tuned global threshold. The primary performance improvements come from training-time strategies (oversampling and class-weighted loss), while thresholding plays a secondary role.

---

## Experiment 14 – Threshold calibration on held-out validation split

**Date:** 2026-04-17

### Goal
Evaluate whether threshold tuning on the validation set leads to overfitting, by separating threshold selection from final evaluation.

### Setup
- Model: oversampling + class-weighted loss (best performing model)
- Validation set split into two halves:
  - Calibration split (50%) for threshold selection
  - Held-out split (50%) for evaluation
- Thresholds tested: 0.3–0.7

### Calibration results
- Best threshold (by Macro-F1): 0.7
- Calibration Macro-F1: 0.4105
- Calibration Micro-F1: 0.4741
- Calibration Tail Recall: 0.7083

### Held-out evaluation
- Selected threshold: 0.7
- Micro-F1: 0.4425
- Macro-F1: 0.4358
- Tail Recall: 0.8472

### Observations
- The selected threshold remains consistent with earlier experiments.
- Performance on the held-out split does not degrade compared to the calibration split.
- Tail recall is even higher on the held-out data.
- This suggests that threshold tuning captures generalizable model behavior rather than overfitting.

### Conclusion
Separating threshold selection from evaluation confirms that the observed threshold effects are robust. The model maintains strong tail-label performance under a proper validation protocol.

---

## Experiment 15 – Threshold calibration stability across alternative validation split

**Date:** 2026-04-17

### Goal
Verify whether threshold selection remains stable across different validation splits.

### Setup
- Same model as Experiment 14 (oversampling + class-weighted loss)
- Alternative split strategy:
  - Calibration set: every second sample
  - Held-out set: complementary samples
- Thresholds tested: 0.3–0.7

### Calibration results
- Best threshold (by Macro-F1): 0.7
- Calibration Macro-F1: 0.3750
- Calibration Micro-F1: 0.4456
- Calibration Tail Recall: 0.4915

### Held-out evaluation
- Selected threshold: 0.7
- Micro-F1: 0.4700
- Macro-F1: 0.4266
- Tail Recall: 0.9500

### Observations
- The optimal threshold remains identical to Experiment 14 (0.7).
- Performance remains stable across different split strategies.
- Tail recall remains very high and even increases on the held-out set.
- Results suggest that threshold selection is not sensitive to a specific validation partition.

### Conclusion
Threshold calibration is robust across different validation splits. The observed improvements are not artifacts of a specific data partition, but reflect stable model behavior.

---

## Experiment 16 – Tail precision / recall / F1 analysis on OW model

**Date:** 2026-04-20

### Goal
Evaluate the precision–recall trade-off for tail labels on the best-performing model (oversampling + class-weighted loss).

### Setup
- Model: oversampling + class-weighted loss (Experiment 12)
- No retraining
- Thresholds tested: 0.3–0.7
- Metrics: tail precision, tail recall, tail F1

### Results

- Threshold 0.3:
  - Tail Precision: 0.1364
  - Tail Recall: 0.7562
  - Tail F1: 0.2312

- Threshold 0.4:
  - Tail Precision: 0.1939
  - Tail Recall: 0.7562
  - Tail F1: 0.3087

- Threshold 0.5:
  - Tail Precision: 0.2616
  - Tail Recall: 0.7562
  - Tail F1: 0.3888

- Threshold 0.6:
  - Tail Precision: 0.2912
  - Tail Recall: 0.7312
  - Tail F1: 0.4166

- Threshold 0.7:
  - Tail Precision: 0.3499
  - Tail Recall: 0.7216
  - Tail F1: 0.4713

### Observations
- Increasing the threshold steadily improves tail precision while only slightly reducing recall.
- Tail recall remains consistently high across all tested thresholds.
- Tail F1 improves monotonically with increasing threshold, reaching its maximum at 0.7.
- This indicates that the model captures meaningful signal for rare labels rather than relying on overprediction.

### Conclusion
The oversampling + class-weighted model achieves a strong and stable precision–recall trade-off for tail labels. Unlike earlier experiments, improvements in tail recall are not driven by excessive false positives, but by genuinely improved discrimination.

---


## Experiment 17 – Seed repeat of the best model

**Date:** 2026-04-23

### Goal
Check whether the best-performing setup remains stable under a different random seed.

### Setup
- Model: oversampling + class-weighted loss
- Same setup as Experiment 12
- Random seed: 123
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.7968
- Eval Micro-F1 (threshold = 0.5): 0.3830
- Eval Macro-F1 (threshold = 0.5): 0.3543
- Eval Tail Recall (threshold = 0.5): 0.7216

### Thresholded evaluation
- Threshold: 0.7
- Micro-F1: 0.4690
- Macro-F1: 0.4289
- Tail Recall: 0.6904
- Max predicted probability on validation set: 0.9790
- Mean predicted probability on validation set: 0.2074

### Observations
- Results remain very close to the original best run.
- Macro-F1 and tail recall stay high under a different seed.
- This suggests that the best-performing configuration is reasonably stable and not the result of a lucky initialization.

### Conclusion
The oversampling + class-weighted setup appears robust across seeds, with consistent threshold behavior and stable rare-label performance.

---

## Experiment 18 – Seed repeat of the best model (seed = 7)

**Date:** 2026-04-27

### Goal
Further evaluate the stability of the best-performing setup under a different random seed.

### Setup
- Model: oversampling + class-weighted loss
- Same setup as Experiment 12
- Random seed: 7
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.7843
- Eval Micro-F1 (threshold = 0.5): not recorded separately
- Eval Macro-F1 (threshold = 0.5): not recorded separately
- Eval Tail Recall (threshold = 0.5): not recorded separately

### Thresholded evaluation
- Threshold: 0.7
- Micro-F1: 0.4747
- Macro-F1: 0.4522
- Tail Recall: 0.7060
- Max predicted probability on validation set: 0.9835
- Mean predicted probability on validation set: 0.2087

### Observations
- Results remain in the same range as previous runs.
- Macro-F1 is slightly higher than in the earlier best run.
- Tail recall remains high.

### Conclusion
The best-performing setup remains stable under seed = 7 and continues to show strong macro-F1 and tail performance.

---

## Experiment 19 – Seed repeat of the best model (seed = 42)

**Date:** 2026-04-27

### Goal
Further evaluate the stability of the best-performing setup under another random seed.

### Setup
- Model: oversampling + class-weighted loss
- Same setup as Experiment 12
- Random seed: 42
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.7857
- Eval Micro-F1 (threshold = 0.5): not recorded separately
- Eval Macro-F1 (threshold = 0.5): not recorded separately
- Eval Tail Recall (threshold = 0.5): not recorded separately

### Thresholded evaluation
- Threshold: 0.7
- Micro-F1: 0.4578
- Macro-F1: 0.4361
- Tail Recall: 0.7216
- Max predicted probability on validation set: 0.9804
- Mean predicted probability on validation set: 0.2044

### Observations
- Results remain very close to the earlier runs.
- Tail recall remains high.
- The overall behavior is consistent with the previous seed repeat.

### Conclusion
The best-performing setup appears stable under seed = 42 as well.

---

## Experiment 20 – Summary across seeds

**Date:** 2026-04-27

### Goal
Summarize the stability of the current best-performing setup across multiple random seeds.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Runs included:
  - original
  - seed = 123
  - seed = 7
  - seed = 42

### Results
- Micro-F1: 0.4648 ± 0.0084
- Macro-F1: 0.4369 ± 0.0107
- Tail Recall: 0.7099 ± 0.0149

### Observations
- Variation across seeds is small.
- Macro-F1 and tail recall remain consistently high.
- The current best setup is reasonably robust.

### Conclusion
The oversampling + class-weighted setup appears stable across seeds and remains the strongest candidate so far for the tail-focused scenario.

---

## Experiment 21 – Per-tail-label breakdown on the best model

**Date:** 2026-04-30

### Goal
Inspect which tail labels are learned well by the best-performing model, and which remain difficult.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Per-label precision / recall / F1 computed for the tail labels only

### Results
- pride: precision 0.7500, recall 0.7500, F1 0.7500
- remorse: precision 0.5000, recall 0.9231, F1 0.6486
- fear: precision 0.5455, recall 0.7500, F1 0.6316
- embarrassment: precision 0.4286, recall 0.6000, F1 0.5000
- nervousness: precision 0.2500, recall 1.0000, F1 0.4000
- desire: precision 0.2727, recall 0.7500, F1 0.4000
- grief: precision 0.0526, recall 1.0000, F1 0.1000
- relief: precision 0.0000, recall 0.0000, F1 0.0000

### Observations
- Tail-label performance is highly heterogeneous.
- Some rare labels (e.g. pride, remorse, fear) are learned relatively well.
- Others (especially grief and relief) remain difficult or unstable.
- This suggests that tail-label difficulty is not explained by frequency alone.

### Conclusion
The best-performing setup improves tail-label performance overall, but not uniformly. Some rare emotions remain substantially harder than others.

---

## Experiment 22 – Threshold calibration by Tail-F1

**Date:** 2026-04-30

### Goal
Check whether selecting the threshold by a tail-focused objective (Tail-F1) leads to a different optimum than selecting it by Macro-F1.

### Setup
- Model: oversampling + class-weighted loss
- Validation split into calibration and held-out halves
- Thresholds tested: 0.3–0.7
- Threshold selected by Tail-F1 on the calibration split

### Calibration results
- Threshold 0.3: Tail F1 = 0.2184
- Threshold 0.4: Tail F1 = 0.2747
- Threshold 0.5: Tail F1 = 0.3557
- Threshold 0.6: Tail F1 = 0.3765
- Threshold 0.7: Tail F1 = 0.4231

- Best threshold (by Tail-F1): 0.7

### Held-out evaluation
- Selected threshold: 0.7
- Micro-F1: 0.4425
- Macro-F1: 0.4358
- Tail Precision: 0.4044
- Tail Recall: 0.8472
- Tail F1: 0.5475

### Observations
- The best threshold remains 0.7 even when the calibration objective is tail-focused.
- Tail performance remains strong on the held-out split.
- This suggests that the selected threshold is robust not only for overall metrics, but also for the tail-focused scenario.

### Conclusion
Threshold selection appears stable across objectives: the same threshold emerges as optimal under both Macro-F1 and Tail-F1 based calibration. This strengthens confidence in the current best configuration.

---

## Experiment 23 – Weighted-only seed repeat (seed = 7)

**Date:** 2026-05-13

### Goal
Further evaluate the stability of the weighted-only configuration under a different random seed.

### Setup
- Model: class-weighted loss only
- Same base setup as Experiment 10
- Training subset: 5000 samples
- Validation subset: 500 samples
- Random seed: 7
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.9359
- Thresholded evaluation:
  - Micro-F1: 0.4568
  - Macro-F1: 0.4285
  - Tail Recall: 0.5966
- Max predicted probability on validation set: 0.9773
- Mean predicted probability on validation set: 0.2285

### Observations
- Results remain in the same range as the original weighted-only run.
- Macro-F1 stays strong, while tail recall remains clearly above the plain baseline.
- The weighted-only setup appears reasonably stable, but tail performance is still lower than in the OW setting.

### Conclusion
The weighted-only configuration remains a viable tail-focused setup under seed = 7, but does not yet match the strongest OW results.

---

## Experiment 24 – Weighted-only seed repeat (seed = 42)

**Date:** 2026-05-13

### Goal
Further evaluate the stability of the weighted-only configuration under another random seed.

### Setup
- Model: class-weighted loss only
- Same base setup as Experiment 10
- Training subset: 5000 samples
- Validation subset: 500 samples
- Random seed: 42
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.9301
- Thresholded evaluation:
  - Micro-F1: 0.4359
  - Macro-F1: 0.4189
  - Tail Recall: 0.7120
- Max predicted probability on validation set: 0.9758
- Mean predicted probability on validation set: 0.2278

### Observations
- This run again confirms that the weighted-only setup remains competitive at threshold 0.7.
- Tail recall is substantially higher here than in the original weighted-only run.
- However, performance appears somewhat more variable than in the OW configuration, especially for tail recall.

### Conclusion
The weighted-only setup remains strong across seeds, but its tail behavior appears less stable than OW.

---

## Experiment 25 – Weighted-only summary across runs

**Date:** 2026-05-13

### Goal
Summarize the stability of the weighted-only configuration across multiple runs.

### Setup
- Model: class-weighted loss only
- Evaluation threshold: 0.7
- Runs included:
  - original
  - seed = 7
  - seed = 42

### Results
- Micro-F1: 0.4496 ± 0.0118
- Macro-F1: 0.4231 ± 0.0049
- Tail Recall: 0.6267 ± 0.0750

### Observations
- The weighted-only configuration is fairly stable in Micro-F1 and Macro-F1.
- Tail Recall is more variable across seeds than in the OW setup.
- Compared to the current OW summary, weighted-only remains weaker on average in all three key metrics, and especially less stable in tail recall.

### Conclusion
The weighted-only configuration is a useful comparison point and remains a strong tail-focused baseline, but the current evidence still favors the OW setup as the main direction.

---

## Experiment 26 – OW seed repeat (seed = 99)

**Date:** 2026-05-20

### Goal
Further validate the selected main direction by testing the OW configuration under an additional random seed.

### Setup
- Model: oversampling + class-weighted loss
- Same setup as Experiment 12
- Random seed: 99
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.7563
- Thresholded evaluation:
  - Micro-F1: 0.4753
  - Macro-F1: 0.4431
  - Tail Recall: 0.6964
- Max predicted probability on validation set: 0.9836
- Mean predicted probability on validation set: 0.2040

### Observations
- The result remains in the same range as the earlier OW seed runs.
- Micro-F1 is among the strongest observed values so far.
- Macro-F1 remains strong, while tail recall stays high.
- This further supports that the selected OW direction is stable across seeds.

### Conclusion
The additional seed repeat strengthens confidence in OW as the main experimental direction. The configuration continues to provide a strong and stable trade-off across the key metrics.

---

## Experiment 27 – Updated OW summary across runs

**Date:** 2026-05-20

### Goal
Update the multi-run summary of the selected main direction after adding one more OW seed run.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Runs included:
  - original
  - seed = 123
  - seed = 7
  - seed = 42
  - seed = 99

### Results
- Micro-F1: 0.4669 ± 0.0086
- Macro-F1: 0.4381 ± 0.0097
- Tail Recall: 0.7072 ± 0.0143

### Observations
- The updated OW summary remains highly stable across seeds.
- The new seed run does not change the overall picture, but further reinforces it.
- Tail recall remains consistently high with low variance.

### Conclusion
The additional run strengthens confidence that OW is a stable and robust main direction.

---

## Experiment 28 – Direct comparison of OW and weighted-only summaries

**Date:** 2026-05-20

### Goal
Compare the two selected candidate directions directly using their current multi-run summaries.

### Setup
- Weighted-only summary:
  - Micro-F1: 0.4496
  - Macro-F1: 0.4231
  - Tail Recall: 0.6267
- OW summary:
  - Micro-F1: 0.4669
  - Macro-F1: 0.4381
  - Tail Recall: 0.7072

### Results
- Delta Micro-F1: +0.0173
- Delta Macro-F1: +0.0150
- Delta Tail Recall: +0.0805

### Observations
- OW outperforms weighted-only in all three key metrics.
- The largest improvement appears in Tail Recall.
- Combined with the lower variance seen in the OW runs, this supports OW as the stronger main direction.

### Conclusion
The current evidence favors OW over weighted-only both in average performance and in stability, especially for the tail-focused objective.

---

## Experiment 29 – Visual comparison of OW and weighted-only multi-run summaries

**Date:** 2026-05-21

### Goal
Create a visual comparison of the two selected candidate directions based on their multi-run summaries.

### Setup
- Weighted-only summary:
  - Micro-F1: 0.4496 ± 0.0118
  - Macro-F1: 0.4231 ± 0.0049
  - Tail Recall: 0.6267 ± 0.0750
- OW summary:
  - Micro-F1: 0.4669 ± 0.0086
  - Macro-F1: 0.4381 ± 0.0097
  - Tail Recall: 0.7072 ± 0.0143

### Observations
- OW outperforms weighted-only in all three key metrics.
- The largest difference appears in Tail Recall.
- The Tail Recall variance is substantially smaller for OW than for weighted-only.
- The visual comparison reinforces OW as the stronger and more stable main direction.

### Conclusion
The visual summary supports the same conclusion as the numeric comparison: OW is currently the preferred main direction, while weighted-only remains a useful control setup.

---

## Experiment 30 – Tail-label error breakdown on OW model

**Date:** 2026-05-21

### Goal
Inspect label-level error patterns for the tail labels on the selected main direction (OW).

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Metrics computed per tail label:
  - support
  - TP
  - FP
  - FN
  - precision
  - recall
  - F1

### Results
- pride: support 4, TP 3, FP 1, FN 1, precision 0.7500, recall 0.7500, F1 0.7500
- remorse: support 13, TP 12, FP 12, FN 1, precision 0.5000, recall 0.9231, F1 0.6486
- fear: support 8, TP 6, FP 5, FN 2, precision 0.5455, recall 0.7500, F1 0.6316
- embarrassment: support 5, TP 3, FP 4, FN 2, precision 0.4286, recall 0.6000, F1 0.5000
- nervousness: support 2, TP 2, FP 6, FN 0, precision 0.2500, recall 1.0000, F1 0.4000
- desire: support 8, TP 6, FP 16, FN 2, precision 0.2727, recall 0.7500, F1 0.4000
- grief: support 1, TP 1, FP 18, FN 0, precision 0.0526, recall 1.0000, F1 0.1000
- relief: support 1, TP 0, FP 7, FN 1, precision 0.0000, recall 0.0000, F1 0.0000

### Observations
- Tail-label performance is highly non-uniform.
- Some rare labels, especially pride, remorse, and fear, are handled relatively well.
- grief shows perfect recall but extremely poor precision, indicating severe overprediction.
- relief is not learned reliably at all in this slice.
- These results suggest that tail-label difficulty is not explained by rarity alone.

### Conclusion
The selected OW direction remains strong overall, but its tail-label behavior is heterogeneous. Some rare labels are already separated reasonably well, while others remain dominated by confusion and overprediction.

---

## Experiment 31 – False negative analysis for grief

**Date:** 2026-05-21

### Goal
Check whether the OW model misses true grief examples.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: grief

### Results
- Number of grief false negatives: 0

### Observations
- In this validation slice, the model did not miss the single true grief example.

### Conclusion
For grief, the main problem is not false negatives but false positives.

---

## Experiment 32 – False positive analysis for grief

**Date:** 2026-05-21

### Goal
Inspect why the OW model overpredicts grief.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: grief

### Results
- Number of grief false positives: 18

### Example patterns
Typical false positives included texts whose gold labels were closer to sadness or remorse, for example:
- “I've never been this sad in my life!” → true: sadness
- “Oh dear, The combination of drugs, alcohol and a cell phone has caused many regrets.” → true: remorse
- “I miss those days.” → true: sadness
- “I'm so sorry for your losses...” → true: caring, remorse

### Observations
- grief is often predicted together with sadness, remorse, and sometimes disappointment.
- The model appears to activate grief for broadly negative or loss-related language, even when the gold label is not grief.
- This suggests that the model has not learned a sufficiently narrow decision boundary for grief.

### Conclusion
The main grief failure mode is overprediction, not underdetection. grief appears to be semantically entangled with sadness- and remorse-like content.

---

## Experiment 33 – False negative analysis for relief

**Date:** 2026-05-21

### Goal
Inspect whether the OW model fails to detect true relief examples.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: relief

### Results
- Number of relief false negatives: 1

### Example
- “try going outside for a long painful walk”
  - true: relief
  - predicted: caring
  - relief probability: 0.1015

### Observations
- The single true relief example in this slice was missed with low confidence.
- This suggests that relief is not being robustly recognized in the current setting.

### Conclusion
relief remains a difficult label for the OW setup, with weak positive detection in this validation slice.

---

## Experiment 34 – False positive analysis for relief

**Date:** 2026-05-21

### Goal
Inspect why the OW model overpredicts relief.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: relief

### Results
- Number of relief false positives: 7

### Example patterns
Typical false positives included texts whose gold labels were closer to gratitude, joy, pride, admiration, or optimism, for example:
- “Any response helps and makes me feel like I’m not alone....thank you❤️” → true: gratitude
- “My blood pressure Edit: I'm just happy to be here.” → true: joy
- “I had a monster panic attack myself... I guess I should be proud, too!” → true: pride
- “Congrats, darling! ... I'm both happy for and proud of you” → true: excitement, joy, pride

### Observations
- relief is often predicted together with joy, pride, gratitude, admiration, or optimism.
- The model appears to treat relief as a broader “positive emotional release” category.
- This suggests strong semantic overlap in the learned representation.

### Conclusion
The main relief failure mode is semantic confusion with nearby positive labels rather than simple absence of signal.

---

## Experiment 35 – Label-specific threshold sweep for grief

**Date:** 2026-05-28

### Goal
Test whether the grief label can be improved by adjusting its threshold separately while keeping all other labels at the default OW threshold.

### Setup
- Model: oversampling + class-weighted loss
- Default threshold for all labels: 0.7
- Target label: grief
- grief threshold tested: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95

### Results
- At threshold 0.70:
  - Micro-F1: 0.4572
  - Macro-F1: 0.4277
  - Tail Recall: 0.7216
  - grief precision: 0.0526
  - grief recall: 1.0000
  - grief F1: 0.1000
  - TP: 1, FP: 18, FN: 0

- At threshold 0.90:
  - Micro-F1: 0.4610
  - Macro-F1: 0.4343
  - Tail Recall: 0.7216
  - grief precision: 0.1667
  - grief recall: 1.0000
  - grief F1: 0.2857
  - TP: 1, FP: 5, FN: 0

- At threshold 0.95:
  - Micro-F1: 0.4614
  - Macro-F1: 0.4241
  - Tail Recall: 0.5966
  - grief precision: 0.0000
  - grief recall: 0.0000
  - grief F1: 0.0000
  - TP: 0, FP: 0, FN: 1

### Observations
- Increasing the grief threshold substantially reduces false positives.
- Up to 0.90, recall remains unchanged while precision and F1 improve clearly.
- At 0.95, the single true grief example is lost entirely.
- This suggests that grief is at least partly a calibration problem rather than only a representation problem.

### Conclusion
Grief appears to benefit from targeted label-level threshold adjustment. A higher label-specific threshold improves grief precision and F1 without harming recall up to a point.

---

## Experiment 36 – Label-specific threshold sweep for relief

**Date:** 2026-05-28

### Goal
Test whether the relief label can be improved by adjusting its threshold separately while keeping all other labels at the default OW threshold.

### Setup
- Model: oversampling + class-weighted loss
- Default threshold for all labels: 0.7
- Target label: relief
- relief threshold tested: 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90

### Results
Across all tested thresholds:
- relief precision: 0.0000
- relief recall: 0.0000
- relief F1: 0.0000

False positives decreased as the threshold increased:
- FP = 45 at threshold 0.3
- FP = 7 at threshold 0.7
- FP = 0 at threshold 0.9

However:
- TP remained 0 throughout
- FN remained 1 throughout

### Observations
- Changing the relief threshold only changes the number of false positives.
- The model never recovers the single true relief example at any tested threshold.
- This suggests that relief is not primarily a threshold-calibration problem.

### Conclusion
Relief does not benefit from simple label-specific threshold adjustment. Its difficulty likely comes from weak representation and/or semantic overlap with nearby positive labels.

---

## Experiment 37 – Visual comparison of label-specific threshold sweeps for grief and relief

**Date:** 2026-05-28

### Goal
Visualize whether grief and relief respond differently to targeted label-level threshold adjustment.

### Setup
- Model: oversampling + class-weighted loss
- Label-specific threshold sweeps for grief and relief
- Precision, recall, and F1 plotted against threshold values

### Observations
- grief shows a meaningful precision/F1 improvement as the threshold increases up to around 0.90, while recall remains intact.
- relief remains at zero precision, recall, and F1 across all tested thresholds.
- The visual comparison reinforces that these two problematic tail labels fail for different reasons.

### Conclusion
The threshold-sweep plots support a differentiated view of tail-label difficulty: grief can be partly improved through calibration, while relief likely requires a different kind of intervention.

---

## Experiment 38 – Clipped positive weights for OW setup

**Date:** 2026-06-04

### Goal
Test whether reducing the most extreme class weights helps mitigate overprediction in the selected OW setup.

### Setup
- Model: oversampling + class-weighted loss
- Original positive weights:
  - min: 2.2013
  - max: 293.1667
  - mean: 54.2542
- Clipped positive weights:
  - min: 2.2013
  - max: 50.0
  - mean: 31.9938

### Observations
- Clipping substantially reduced the most extreme class weights.
- This provides a simple way to test whether some of the tail-label problems are caused by overly aggressive weighting.

### Conclusion
The clipped-weight setup provides a controlled variant of OW for testing whether milder weighting improves precision on problematic labels.

---

## Experiment 39 – OW with clipped class weights

**Date:** 2026-06-04

### Goal
Evaluate whether clipping the positive class weights improves the OW setup.

### Setup
- Model: oversampling + class-weighted loss
- Positive weights clipped to max = 50.0
- Random seed: 1234
- Evaluation threshold: 0.7

### Results
- Eval loss: 0.6602
- Thresholded evaluation:
  - Micro-F1: 0.4645
  - Macro-F1: 0.4292
  - Tail Recall: 0.4716
- Max predicted probability on validation set: 0.9852
- Mean predicted probability on validation set: 0.1995

### Observations
- Compared to the original OW setup, clipped weights slightly improved Micro-F1 and Macro-F1.
- However, Tail Recall dropped substantially.
- This suggests that the clipping was too aggressive for the tail-focused objective.

### Conclusion
Simple global clipping of class weights does not appear to be a good trade-off in its current form: it improves overall metrics slightly, but sacrifices too much tail recall.

---

## Experiment 40 – Overall comparison: original OW vs clipped-weight OW

**Date:** 2026-06-04

### Goal
Compare the overall behavior of the original OW setup and the clipped-weight variant.

### Setup
- Original OW:
  - Micro-F1: 0.4572
  - Macro-F1: 0.4277
  - Tail Recall: 0.7216
- Clipped-weight OW:
  - Micro-F1: 0.4645
  - Macro-F1: 0.4292
  - Tail Recall: 0.4716

### Results
- Delta Micro-F1: +0.0073
- Delta Macro-F1: +0.0015
- Delta Tail Recall: -0.2500

### Observations
- Clipping yields a small gain in overall metrics.
- However, the tail recall drop is large.
- For the current research focus, this trade-off is unfavorable.

### Conclusion
The clipped-weight OW variant is not preferable to the original OW setup under the current tail-focused objective.

---

## Experiment 41 – Grief threshold sweep on clipped-weight OW model

**Date:** 2026-06-05

### Goal
Check whether the loss of grief recall in the clipped-weight OW model can be recovered by lowering the grief-specific threshold.

### Setup
- Model: clipped-weight OW
- Default threshold for all labels: 0.7
- Target label: grief
- grief threshold tested: 0.30, 0.40, 0.50, 0.60, 0.70

### Results
- At threshold 0.70:
  - Micro-F1: 0.4645
  - Macro-F1: 0.4292
  - Tail Recall: 0.4716
  - grief precision: 0.0000
  - grief recall: 0.0000
  - grief F1: 0.0000
  - TP: 0, FP: 1, FN: 1

- At threshold 0.50:
  - Micro-F1: 0.4618
  - Macro-F1: 0.4340
  - Tail Recall: 0.5966
  - grief precision: 0.0714
  - grief recall: 1.0000
  - grief F1: 0.1333
  - TP: 1, FP: 13, FN: 0

- At threshold 0.40:
  - Micro-F1: 0.4612
  - Macro-F1: 0.4334
  - Tail Recall: 0.5966
  - grief precision: 0.0625
  - grief recall: 1.0000
  - grief F1: 0.1176
  - TP: 1, FP: 15, FN: 0

### Observations
- Lowering the grief-specific threshold on the clipped model does recover the true grief example.
- However, this also reintroduces many false positives.
- Even after recovery, the clipped model remains weaker on grief than the original OW model with targeted thresholding.

### Conclusion
The clipped-weight OW model can recover grief recall through label-specific threshold adjustment, but this does not outperform the original OW setup. For grief, targeted calibration on the original OW model appears more promising than global weight clipping.

---

## Experiment 42 – Comparison of grief threshold sweeps: original OW vs clipped-weight OW

**Date:** 2026-06-05

### Goal
Compare whether grief is handled better by the original OW setup or by the clipped-weight OW setup after label-specific threshold adjustment.

### Setup
- Compare grief threshold sweeps on:
  - original OW
  - clipped-weight OW

### Observations
- In the original OW model, the best grief trade-off appears around threshold 0.90:
  - precision 0.1667
  - recall 1.0000
  - F1 0.2857
  - FP = 5

- In the clipped-weight OW model, the best grief recovery appears around threshold 0.50:
  - precision 0.0714
  - recall 1.0000
  - F1 0.1333
  - FP = 13

### Conclusion
For grief, the original OW model with targeted threshold adjustment remains clearly better than the clipped-weight OW variant. This suggests that simple global clipping is not the right fix for this label.

---

## Experiment 43 – Relief false positive true-label frequency analysis

**Date:** 2026-06-15

### Goal
Quantify which true labels appear most often among the relief false positives in the OW model.

### Setup
- Model: oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: relief
- Analysis restricted to false positive relief predictions

### Results
Number of relief false positives: 7

True-label frequencies among relief false positives:
- pride: 3
- joy: 2
- admiration: 1
- caring: 1
- excitement: 1
- gratitude: 1
- remorse: 1

### Observations
- relief false positives are dominated by nearby positive labels, especially pride and joy.
- The qualitative examples also show overlap with gratitude, admiration, and excitement.
- This supports the earlier hypothesis that relief is not mainly a threshold problem, but a semantic confusion problem with nearby positive emotional categories.

### Conclusion
The current OW model tends to use relief as a broader “positive emotional release” label, which causes systematic confusion with pride-, joy-, and gratitude-like examples.

---

## Experiment 44 – Grief best-threshold summary

**Date:** 2026-06-15

### Goal
Summarize the current best label-specific threshold candidate for grief on the original OW model.

### Setup
- Model: original oversampling + class-weighted loss
- Grief label-specific threshold sweep reused from Experiment 35
- Selection criteria:
  - best grief F1
  - best macro-F1 while keeping full grief recall

### Results
Both selection criteria identified the same threshold:

- Recommended grief threshold: 0.90
- Micro-F1: 0.4610
- Macro-F1: 0.4343
- Tail Recall: 0.7216
- grief precision: 0.1667
- grief recall: 1.0000
- grief F1: 0.2857
- TP: 1
- FP: 5
- FN: 0

### Observations
- The same threshold is selected both by grief-F1 and by macro-F1 under full grief recall.
- This gives a clear and stable current candidate for grief-specific calibration.
- Compared to the clipped-weight branch, the original OW model remains better for grief when targeted thresholding is used.

### Conclusion
For grief, the current best direction is the original OW setup with label-specific threshold 0.90.

---

## Experiment 45 – Targeted grief calibration candidate

**Date:** 2026-06-15

### Goal
Test whether applying the currently best grief-specific threshold candidate improves the overall OW setup compared to the default global threshold.

### Setup
- Model: original oversampling + class-weighted loss
- Baseline setup: global threshold = 0.70
- Candidate setup: grief threshold = 0.90, all other labels = 0.70

### Results
- OW global 0.70:
  - Micro-F1: 0.4572
  - Macro-F1: 0.4277
  - Tail Recall: 0.7216
  - grief precision: 0.0526
  - grief recall: 1.0000
  - grief F1: 0.1000
  - grief TP/FP/FN: 1 / 18 / 0

- OW grief 0.90:
  - Micro-F1: 0.4610
  - Macro-F1: 0.4343
  - Tail Recall: 0.7216
  - grief precision: 0.1667
  - grief recall: 1.0000
  - grief F1: 0.2857
  - grief TP/FP/FN: 1 / 5 / 0

### Observations
- Applying grief = 0.90 improves both Micro-F1 and Macro-F1 while preserving Tail Recall.
- grief precision and grief F1 improve substantially.
- False positives for grief drop strongly, while the true positive is preserved.

### Conclusion
A targeted grief-specific threshold of 0.90 is currently a strong candidate calibration setting on top of the original OW model.

---

## Experiment 46 – Relief false positive co-predicted label analysis

**Date:** 2026-06-15

### Goal
Quantify which labels the model most often predicts together with relief in relief false positive cases.

### Setup
- Model: original oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Analysis restricted to relief false positives

### Results
Co-predicted label frequencies among relief false positives:
- joy: 5
- pride: 5
- admiration: 4
- gratitude: 4
- excitement: 2
- optimism: 1
- remorse: 1
- sadness: 1

### Observations
- relief false positives are most strongly associated with joy-, pride-, admiration-, and gratitude-like predictions.
- The qualitative examples support the same pattern.
- This indicates that the model tends to use relief as part of a broader positive emotional release cluster.

### Conclusion
The current OW model’s relief errors are consistent with semantic confusion among nearby positive labels, rather than simple threshold miscalibration.

---

## Experiment 47 – Relief positive-cluster summary

**Date:** 2026-06-30

### Goal
Quantify whether the relief false positive cases in the OW model are concentrated around a nearby positive label cluster.

### Setup
- Model: original oversampling + class-weighted loss
- Evaluation threshold: 0.7
- Target label: relief
- Positive cluster used for analysis:
  - joy
  - pride
  - admiration
  - gratitude
  - excitement
  - optimism

### Results
- In relief false positive cases, the proportion of true labels belonging to the positive cluster is:
  - 8 / 10 = 0.800
- In relief false positive cases, the proportion of co-predicted labels belonging to the positive cluster is:
  - 21 / 23 = 0.913

### Observations
- Most relief false positives occur in examples whose true labels already belong to a nearby positive cluster.
- Even more strongly, the model’s co-predictions in these cases are overwhelmingly drawn from the same positive cluster.
- This supports the interpretation that relief errors are structured and non-random.

### Conclusion
The relief false positives are strongly concentrated around a nearby positive emotional cluster, suggesting that the main problem is semantic confusion rather than threshold miscalibration.

---

## Experiment 48 – Grief candidate delta summary

**Date:** 2026-06-30

### Goal
Summarize the overall effect of applying the grief-specific threshold candidate (`grief = 0.90`) compared to the baseline global threshold setup.

### Setup
- Baseline: original OW model with global threshold = 0.70
- Candidate: original OW model with grief threshold = 0.90 and all other labels = 0.70

### Results
Baseline (`OW_global_0.70`):
- Micro-F1: 0.4572
- Macro-F1: 0.4277
- Tail Recall: 0.7216
- grief precision: 0.0526
- grief recall: 1.0000
- grief F1: 0.1000
- grief TP/FP/FN: 1 / 18 / 0

Candidate (`OW_grief_0.90`):
- Micro-F1: 0.4610
- Macro-F1: 0.4343
- Tail Recall: 0.7216
- grief precision: 0.1667
- grief recall: 1.0000
- grief F1: 0.2857
- grief TP/FP/FN: 1 / 5 / 0

Delta (candidate − baseline):
- Micro-F1: +0.0038
- Macro-F1: +0.0066
- Tail Recall: +0.0000
- grief precision: +0.1140
- grief recall: +0.0000
- grief F1: +0.1857
- grief FP: -13

### Observations
- The grief-specific candidate improves both Micro-F1 and Macro-F1.
- Tail Recall remains unchanged.
- grief precision and grief F1 improve substantially.
- The number of grief false positives drops strongly while preserving the true positive example.

### Conclusion
The `grief = 0.90` label-specific calibration is currently a strong candidate refinement of the original OW setup.

---

## Related Work Note – Comparison to *Handling Class Imbalance in Dimensional Emotion Recognition with Parameter-Efficient Fine-Tuning*

**Date:** 2026-06-30

### Goal
Relate the current findings of this project to a recent thesis on imbalance handling in dimensional emotion recognition.

### Observations
- The reviewed thesis studies imbalance in VAD regression, while the current project studies imbalance in multi-label emotion classification.
- Despite the task difference, both works suggest that imbalance is not only a simple frequency problem: the structure of the emotion / label space also matters.
- In the reviewed thesis, class-frequency weighting is weaker than structure-aware weighting in the continuous affective space.
- This is consistent with the current project’s finding that weighted-only training helps, but does not fully resolve the hardest tail-label cases.
- The current results further suggest that problematic tail labels are heterogeneous: grief behaves more like a calibration problem, whereas relief behaves more like a semantic confusion problem with nearby positive labels.

### Conclusion
The reviewed thesis supports the broader interpretation that imbalance handling should not be treated purely as class-frequency correction. In the current project as well, label-space structure and confusion patterns appear central to the remaining tail-label errors.

---

## Experiment 49 – Full GoEmotions dataset and label-space audit

**Date:** 2026-09-10

### Goal
Re-evaluate the dataset assumptions behind the earlier tail-label experiments using the complete GoEmotions simplified dataset, and inspect whether the emotion labels show structured and potentially asymmetric relationships.

### Setup
- Dataset: complete GoEmotions simplified dataset
- Splits:
  - train: 43,410 examples
  - validation: 5,426 examples
  - test: 5,427 examples
- Number of labels: 28
- No model training was performed in this experiment.
- Analyses:
  - per-label support across train / validation / test
  - verification of the previously selected tail labels
  - number of labels per example
  - train-set label co-occurrence matrix
  - directional conditional co-occurrence probabilities
  - qualitative inspection of selected grief, relief, remorse, and sadness examples

### Results

#### Tail-label support on the complete dataset

| Label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| grief | 77 | 13 | 6 | 96 |
| pride | 111 | 15 | 16 | 142 |
| relief | 153 | 18 | 11 | 182 |
| nervousness | 164 | 21 | 23 | 208 |
| embarrassment | 303 | 35 | 37 | 375 |
| remorse | 545 | 68 | 56 | 669 |
| fear | 596 | 90 | 78 | 764 |
| desire | 641 | 77 | 83 | 801 |

The eight least frequent non-neutral labels on the complete training split are exactly the same eight labels used in the previous experiments:
- grief
- pride
- relief
- nervousness
- embarrassment
- remorse
- fear
- desire

#### Number of labels per example

Training split:
- 1 label: 36,308 examples (83.64%)
- 2 labels: 6,541 examples (15.07%)
- 3 labels: 532 examples (1.23%)
- 4 labels: 28 examples (0.06%)
- 5 labels: 1 example

Mean number of labels per example:
- train: 1.177
- validation: 1.176
- test: 1.166

This shows that although GoEmotions is a multi-label dataset, most examples contain only one annotated label.

#### Selected directional label relationships

- grief → sadness:
  - co-occurrence: 22
  - P(sadness | grief) = 0.2857
  - P(grief | sadness) = 0.0166

- remorse → sadness:
  - co-occurrence: 60
  - P(sadness | remorse) = 0.1101
  - P(remorse | sadness) = 0.0452

- nervousness → fear:
  - co-occurrence: 34
  - P(fear | nervousness) = 0.2073
  - P(nervousness | fear) = 0.0570

- pride → admiration:
  - co-occurrence: 23
  - P(admiration | pride) = 0.2072
  - P(pride | admiration) = 0.0056

- relief → joy:
  - co-occurrence: 9
  - P(joy | relief) = 0.0588
  - P(relief | joy) = 0.0062

- relief → gratitude:
  - co-occurrence: 10
  - P(gratitude | relief) = 0.0654
  - P(relief | gratitude) = 0.0038

- relief and pride did not co-occur in the training split.

### Qualitative observations

- grief + sadness examples are mostly clearly related to death, loss, missing someone, condolences, or other loss-related negative experiences.
- This supports the interpretation that grief and sadness are related, but grief appears to be a more specific category. The strong asymmetry between P(sadness | grief) and P(grief | sadness) is consistent with this interpretation.

- relief examples often describe a negative, uncertain, or stressful situation becoming resolved or less threatening.
- relief sometimes co-occurs with joy or gratitude, but the co-occurrence rates are relatively low.
- This is notable because earlier OW-model error analysis showed that relief false positives were often associated with joy, pride, gratitude, admiration, and other nearby positive labels.

- remorse examples frequently contain expressions such as “sorry”, “apologies”, or “regret”.
- Some of these examples appear closer to apology, politeness, sympathy, or acknowledgement of a mistake than to a narrow psychological interpretation of remorse.
- This suggests that the discrete annotation labels may contain semantic overlap and annotation-related ambiguity.

### Observations
- The earlier 5000/500 subset substantially underestimated the available support for the rarest labels. In particular, the full validation split contains 13 grief and 18 relief examples instead of only one example each.
- The previous tail-label selection remains valid on the complete training dataset.
- Strong label imbalance remains present even with the complete dataset.
- Most samples contain only one annotated label, which limits how directly label co-occurrence can be interpreted as the full relationship between underlying emotions.
- Several label relationships are clearly asymmetric, suggesting that simple symmetric similarity between labels may not fully describe the structure of the emotion space.
- The qualitative examples indicate that some discrete labels have overlapping or context-dependent meanings.

### Conclusion
The full-dataset audit confirms that the previously selected tail labels are appropriate, but also shows that the earlier detailed grief and relief conclusions were based on unnecessarily small validation support and should be re-evaluated using the complete dataset.

The label-space analysis provides initial evidence that emotion-label relationships are structured and often asymmetric. In addition, the qualitative examples suggest that some discrete labels do not correspond to completely isolated emotional categories.

These findings support the next research direction: first re-establish the baseline models on the complete GoEmotions dataset, and then investigate whether a representation or inference method that incorporates continuous or structured relationships between emotion labels can improve the handling of rare and semantically overlapping labels.

---

## Experiment 50 – Full-dataset plain DistilBERT baseline

**Date:** 2026-09-10

### Goal
Establish a stronger and more reliable plain DistilBERT baseline using the complete official GoEmotions train, validation and test splits.

The earlier baseline experiments were based mainly on a 5,000-example training subset and a 500-example validation subset, which provided very limited support for the rarest labels.

### Setup
- Dataset: GoEmotions simplified
- Train size: 43,410
- Validation size: 5,426
- Test size: 5,427
- Model: distilbert-base-uncased
- Task: 28-label multi-label classification
- Loss: BCEWithLogitsLoss
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Maximum sequence length: 128
- Random seed: 42
- Threshold selection: validation Macro-F1
- Test split used only after threshold selection

### Training results

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.093426 | 0.088851 |
| 2 | 0.081178 | 0.084007 |
| 3 | 0.072051 | 0.084055 |

The lowest validation loss was reached after epoch 2, so the corresponding checkpoint was selected as the final model.

### Validation threshold sweep

The best global threshold according to validation Macro-F1 was 0.15.

At threshold = 0.15:
- Micro-F1: 0.5791
- Macro-F1: 0.4712
- Tail Precision: 0.3757
- Tail Recall: 0.3482
- Tail F1: 0.3467

### Test results

Using the validation-selected threshold of 0.15:

- Micro-F1: 0.5739
- Macro-F1: 0.4626
- Tail Precision: 0.3380
- Tail Recall: 0.3374
- Tail F1: 0.3212

### Observations
- Training on the complete dataset produces a substantially stronger plain baseline than the earlier subset-based experiments.
- Validation performance stabilizes after approximately two epochs.
- The optimal global threshold for the plain model is 0.15, which is much lower than the threshold previously used for the weighted / OW models.
- Validation and test performance are relatively close, suggesting that the selected threshold generalizes reasonably well.
- Tail-label performance remains considerably weaker than overall performance, despite the much larger training set.
- This confirms that increased training support helps rare-label recognition substantially, but does not eliminate the tail-label problem.

### Conclusion
The complete-dataset plain DistilBERT model provides a substantially stronger and more reliable baseline than the earlier subset-based experiments.

The model achieves good overall performance, but tail-label performance remains substantially lower, leaving room for imbalance-aware methods such as oversampling and class-weighted loss.

The next step is to train the OW configuration on the same complete dataset and compare it directly against this baseline under the same validation/test protocol.

---

## Experiment 51 – Full-dataset OW baseline

**Date:** 2026-09-10

### Goal
Evaluate the oversampling + class-weighted loss (OW) configuration on the complete GoEmotions dataset using the same training and evaluation protocol as the full-dataset plain baseline.

### Setup
- Dataset: GoEmotions simplified
- Original train size: 43,410
- OW train size after tail-example oversampling: 45,943
- Validation size: 5,426
- Test size: 5,427
- Model: distilbert-base-uncased
- Task: 28-label multi-label classification
- Tail-containing examples duplicated once
- Loss: BCEWithLogitsLoss with class-dependent positive weights
- Positive weights computed from the oversampled training set
- pos_weight range:
  - min: 2.2119
  - max: 297.3312
  - mean: 54.5409
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Maximum sequence length: 128
- Random seed: 42
- Threshold selection: validation Macro-F1
- Test split used only after threshold selection

### Training results

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.629487 | 0.590996 |
| 2 | 0.506744 | 0.570996 |
| 3 | 0.412403 | 0.595926 |

The lowest validation loss was reached after epoch 2, so the corresponding checkpoint was selected as the final model.

### Validation threshold sweep

The best global threshold according to validation Macro-F1 was 0.90.

At threshold = 0.90:
- Micro-F1: 0.5025
- Macro-F1: 0.4895
- Tail Precision: 0.3972
- Tail Recall: 0.6254
- Tail F1: 0.4747

### Test results

Using the validation-selected threshold of 0.90:

- Micro-F1: 0.4913
- Macro-F1: 0.4721
- Tail Precision: 0.3308
- Tail Recall: 0.5986
- Tail F1: 0.4122

### Comparison with full-dataset plain baseline

| Metric | Plain | OW |
|---|---:|---:|
| Threshold | 0.15 | 0.90 |
| Micro-F1 | 0.5739 | 0.4913 |
| Macro-F1 | 0.4626 | 0.4721 |
| Tail Precision | 0.3380 | 0.3308 |
| Tail Recall | 0.3374 | 0.5986 |
| Tail F1 | 0.3212 | 0.4122 |

### Observations
- OW substantially improves rare-label recall on the complete dataset.
- Tail Recall increases from 0.3374 to 0.5986.
- Tail F1 also improves clearly, from 0.3212 to 0.4122.
- Macro-F1 improves slightly.
- Micro-F1 decreases substantially, indicating that the increased tail sensitivity comes at a cost in overall prediction quality.
- Tail Precision remains similar between the two models.
- The optimal global threshold for OW is much higher than for the plain model (0.90 vs 0.15), confirming that class-weighted training strongly shifts the output score distribution.
- The result shows that OW remains effective for the rare-label objective even when much more training data is available, but the trade-off with overall performance is stronger than the earlier subset-based experiments suggested.

### Conclusion
On the complete GoEmotions dataset, OW provides a clear improvement in rare-label detection compared with the plain DistilBERT baseline.

The strongest gain is in Tail Recall and Tail F1, while Micro-F1 decreases substantially. Therefore, OW should not be interpreted as a universally better model, but as a tail-focused configuration that trades some overall performance for substantially stronger rare-label sensitivity.

These results provide a stronger full-dataset baseline for the next stage of the project, where structured or continuous emotion-label relationships can be investigated as a possible way to improve difficult tail labels without relying only on more aggressive imbalance correction.

---

## Experiment 52 – Exploratory geometry of text, label, and joint text-label spaces

**Date:** 2026-09-21

### Motivation

The previous error analyses suggested that fine-grained GoEmotions labels are not independent: several rare labels are systematically confused with semantically related emotions (e.g. grief/sadness, nervousness/fear, pride/admiration).

Before designing a structured or continuous label representation, I ran an exploratory clustering analysis to compare:

1. the semantic structure of the texts alone,
2. the structure induced by the gold multi-label annotations,
3. and a joint text + label representation.

The goal was not to estimate the "true number of human emotions", but to investigate how different the geometry of the text space and annotation space is, and whether a stable intermediate structure appears when both information sources are combined.

---

### A – Text-only representation

Each training example was represented with a normalized `all-MiniLM-L6-v2` sentence embedding (384 dimensions).

The embeddings were reduced to 50 dimensions with PCA before clustering.

- Train examples: 43,410
- Original embedding size: 384
- PCA dimensions: 50
- Explained variance retained by PCA: 0.4708

HDBSCAN settings:

- `min_cluster_size = 100`
- `min_samples = 20`
- Euclidean distance
- EOM cluster selection

Result:

- Number of clusters: 2
- Noise examples: 38,951
- Noise ratio: 0.8973

The result was stable for `min_cluster_size` values from 50 to 500: HDBSCAN continued to return 2 clusters with approximately the same noise ratio.

K-Means was also tested for `k = 2..30`.

The highest silhouette score was only:

- `k = 2`: 0.0951

All other values were approximately 0.04–0.06.

This indicates that the general-purpose semantic text embedding space does not contain a strong discrete cluster structure corresponding naturally to many emotion categories. The HDBSCAN clusters also appeared to partially reflect lexical/style regularities rather than pure emotion structure.

This does **not** imply that emotions are inherently continuous; it only shows that a general semantic sentence embedding does not separate the GoEmotions examples into strong discrete emotion clusters.

---

### B – Label-only representation

Each example was represented only by its 28-dimensional multi-hot GoEmotions annotation vector.

There were:

- 43,410 examples
- 28 label dimensions
- 711 unique multi-label annotation patterns

Because approximately 84% of GoEmotions examples are single-label, many examples occupy exactly the same position in this space.

With HDBSCAN (`min_cluster_size = 100`):

- Number of clusters: 35
- Noise examples: 3,629
- Noise ratio: 0.0836

Parameter sweep:

| min_cluster_size | clusters | noise ratio |
|---:|---:|---:|
| 50 | 65 | 0.0666 |
| 100 | 35 | 0.0836 |
| 200 | 27 | 0.0815 |
| 300 | 23 | 0.0967 |
| 500 | 20 | 0.1144 |
| 750 | 16 | 0.1730 |
| 1000 | 13 | 0.1816 |

The label-only space is therefore strongly discrete compared with the text-only space.

However, this result should not be interpreted as discovering a latent number of emotions: many clusters correspond directly to frequent exact annotation patterns such as single labels or common label combinations.

The main observation is that the **annotation-space geometry is substantially more discrete than the general text-semantic geometry**.

---

### C – Initial joint text + label representation

An initial joint representation concatenated:

- PCA-reduced text embeddings
- standardized multi-hot label vectors

with a weighting parameter:

`joint = [text ; alpha * labels]`

The first version used `StandardScaler` independently on both blocks.

This produced an abrupt structural change at relatively small `alpha` values. Further inspection showed that standardizing highly imbalanced binary label dimensions strongly amplifies rare positive labels. For example, a positive value for an extremely rare label receives a very large z-score.

Therefore the apparent sharp transition in this initial experiment was considered partly a scaling artifact and was not used for substantive conclusions.

---

### C2 – L2-normalized joint representation

The joint representation was redesigned to avoid rare-label amplification.

Processing:

1. normalized MiniLM sentence embeddings
2. PCA to 50 dimensions
3. L2-normalize each text vector
4. convert annotations to 28-dimensional multi-hot vectors
5. L2-normalize each label vector
6. concatenate:

`joint(alpha) = [text_L2 ; alpha * label_L2]`

With this representation:

- mean text-vector norm = 1.0
- mean label-vector norm = 1.0

Combined vector norms matched the expected values:

- alpha = 0.0 → 1.0000
- alpha = 0.5 → 1.1180
- alpha = 1.0 → 1.4142
- alpha = 2.0 → 2.2361

This confirmed that `alpha` now controls the relative contribution of the label block in an interpretable way.

---

### Broad alpha sweep

HDBSCAN results:

| alpha | clusters | noise ratio |
|---:|---:|---:|
| 0.00 | 2 | 0.9602 |
| 0.10 | 2 | 0.9621 |
| 0.25 | 2 | 0.9616 |
| 0.50 | 2 | 0.9422 |
| 0.75 | 2 | 0.8116 |
| 1.00 | 13 | 0.7860 |
| 1.50 | 27 | 0.1539 |
| 2.00 | 38 | 0.0996 |
| 3.00 | 37 | 0.1209 |
| 5.00 | 36 | 0.1228 |
| 10.00 | 36 | 0.1228 |

Three broad regimes appeared:

1. **Text-dominated (`alpha < ~1`)**
   - very few clusters
   - very high noise

2. **Transition region (`alpha ≈ 1–1.5`)**
   - rapid increase in cluster coverage
   - emergence of many emotion-related clusters

3. **Label-dominated (`alpha >= ~2`)**
   - approximately 36–38 clusters
   - approximately 10–12% noise
   - structure approaches the label-only space

Cluster assignments at high alpha were highly stable:

- ARI(1.5, 2.0) = 0.926
- ARI(1.5, 3.0) = 0.946
- ARI(2.0, 3.0) = 0.970
- ARI(3.0, 5.0) = 0.992
- ARI(5.0, 10.0) = 1.000

---

### Detailed transition sweep

The transition region was examined more densely:

| alpha | clusters | noise ratio | clustered ratio |
|---:|---:|---:|---:|
| 0.75 | 2 | 0.8116 | 0.1884 |
| 0.85 | 2 | 0.8169 | 0.1831 |
| 0.90 | 9 | 0.9126 | 0.0874 |
| 0.95 | 3 | 0.5306 | 0.4694 |
| 1.00 | 13 | 0.7860 | 0.2140 |
| 1.05 | 18 | 0.7464 | 0.2536 |
| 1.10 | 19 | 0.6363 | 0.3637 |
| 1.20 | 22 | 0.4008 | 0.5992 |
| 1.30 | 24 | 0.2632 | 0.7368 |
| 1.40 | 24 | 0.1875 | 0.8125 |
| 1.50 | 27 | 0.1539 | 0.8461 |

The region around `alpha ≈ 0.9–1.0` was unstable, suggesting sensitivity of the density structure while the influence of text and labels becomes comparable.

Above approximately `alpha = 1.05`, a much more consistent pattern appeared.

---

### Stability of non-noise clusters

Because HDBSCAN labels unassigned examples as `-1`, ordinary ARI can be dominated by changes in the large noise set. Therefore ARI was also calculated only on examples assigned to a cluster in both adjacent solutions.

Non-noise ARI:

- 1.00 → 1.05: 0.870
- 1.05 → 1.10: 0.999
- 1.10 → 1.20: 0.998
- 1.20 → 1.30: 0.994
- 1.30 → 1.40: ~1.000
- 1.40 → 1.50: 0.989

This suggests that once examples enter the emerging cluster structure, the cluster organization itself remains highly stable. Increasing `alpha` mainly causes additional examples to become incorporated into an already largely stable structure, rather than repeatedly reorganizing the existing clusters.

Cluster-overlap heatmaps for `1.2 → 1.3` and `1.3 → 1.5` showed the same pattern: most established clusters mapped strongly onto a corresponding cluster at the next alpha value.

---

### Qualitative structure

UMAP visualizations showed a clear progression:

- low alpha: mostly continuous text-semantic space
- intermediate alpha: multiple emotion-related islands begin to emerge
- higher alpha: increasingly discrete annotation-like regions

At intermediate alpha values, several clusters were semantically interpretable rather than being arbitrary partitions.

Observed examples included clusters dominated by:

- fear
- disgust
- remorse
- amusement
- optimism
- caring

and mixed clusters containing semantically related labels such as:

- curiosity + confusion
- sadness + disappointment
- joy + excitement
- anger + annoyance
- gratitude + admiration

A sampled fear cluster was also semantically coherent at the text level, containing expressions such as being scared, terrified, horrified, or worried.

This suggests that at least some joint-space clusters are supported simultaneously by both text semantics and annotation structure.

---

### Interpretation

The experiments reveal a large geometric difference between the two representations:

- general text-semantic space is relatively continuous and weakly clusterable;
- gold annotation space is highly discrete;
- a joint representation exhibits a transition between these two geometries.

Importantly, the transition does not simply produce arbitrary new partitions. Above approximately `alpha = 1.05`, already-clustered examples retain highly stable group membership while additional examples progressively join the structure.

This is consistent with the hypothesis that fine-grained emotion labels may be better viewed as a **structured label space** rather than as 28 completely independent binary targets.

However, this experiment does **not** establish:

- the true number of human emotions,
- an optimal psychological emotion representation,
- or that `alpha ≈ 1.2–1.5` has any intrinsic psychological meaning.

`alpha` is an artificial weighting parameter of this exploratory representation.

The result should therefore be treated as evidence motivating further study of structured or learned emotion representations, not as a final emotion taxonomy.

---

### Main takeaway

The text-only and annotation-only representations have substantially different geometries.

A controlled L2-normalized joint representation reveals a stable intermediate structure in which semantically related emotions can form persistent regions. This provides additional motivation for investigating learned label embeddings or latent emotion representations that preserve fine-grained GoEmotions outputs while explicitly modelling relationships between labels.

### Next steps

- Continue the literature review on label embeddings, label dependency modelling, continuous/dimensional emotion representations, and long-tail emotion classification.
- Compare the observed empirical structure with structures proposed in prior emotion-representation literature.
- Design a learned label-embedding / latent emotion-space pilot rather than manually defining the relationships.
- Later validate promising supervised methods with multi-label stratified cross-validation and an untouched final test set.

---

## Experiment 53 – Empirical, semantic, and hybrid label-relation construction

**Date:** 2026-10-15

### Goal

Construct the label-relation matrices required by the proposed structured-label method and evaluate whether empirical co-occurrence and external semantic information provide complementary views of the GoEmotions label space.

### Setup

- Dataset: complete GoEmotions simplified dataset
- Train split: 43,410 examples
- Validation split: 5,426 examples
- Test split: 5,427 examples
- Number of classifier labels: 28
- Structural emotion space: 27 non-neutral labels
- Mean labels per training example: 1.1772

Tail labels:
- grief
- pride
- relief
- nervousness
- embarrassment
- remorse
- fear
- desire

Empirical structure:
- Computed from the official training split only
- Pairwise relation measure: Jaccard similarity of positive-example sets

Semantic structure:
- Natural-language emotion definitions from the original GoEmotions taxonomy
- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Encoder kept frozen
- Embedding dimensionality: 768
- Pairwise relation measure: cosine similarity
- Negative similarities clipped to zero
- Neutral excluded from the semantic structure

Normalization:
- Empirical and semantic matrices normalized independently using off-diagonal min-max normalization
- Diagonal excluded from structural relations

Exploratory hybrid structure:

`S_hybrid = alpha * S_empirical + (1 - alpha) * S_semantic`

with:

`alpha = 0.5`

This alpha value is exploratory only and is not treated as the final selected hyperparameter.

### Dataset and tail-set checks

The full training split contained the following tail-label frequencies:

| Label | Train count |
|---|---:|
| grief | 77 |
| pride | 111 |
| relief | 153 |
| nervousness | 164 |
| embarrassment | 303 |
| remorse | 545 |
| fear | 596 |
| desire | 641 |

The eight least frequent non-neutral labels exactly matched the previously defined tail set.

The training-label matrix had shape:

`(43410, 28)`

and the mean number of labels per example was:

`1.1772`

The raw co-occurrence matrix passed the diagonal consistency check: each diagonal value matched the corresponding training-label frequency.

### Empirical-relation distribution

Off-diagonal Jaccard similarities:

- minimum: 0.000000
- 10th percentile: 0.000000
- 25th percentile: 0.001173
- median: 0.003380
- 75th percentile: 0.006623
- 90th percentile: 0.012424
- 95th percentile: 0.022561
- maximum: 0.071391
- mean: 0.005903
- standard deviation: 0.008921
- zero fraction: 0.1225

The empirical relation matrix is therefore sparse and strongly right-skewed.

The strongest empirical label pairs included:

1. anger – annoyance: 0.071391
2. confusion – curiosity: 0.063340
3. disappointment – sadness: 0.054021
4. fear – nervousness: 0.046832
5. admiration – gratitude: 0.042837
6. annoyance – disapproval: 0.041261
7. excitement – joy: 0.038756
8. admiration – approval: 0.036055
9. caring – optimism: 0.034911
10. remorse – sadness: 0.033131

Selected empirical tail-label neighbours included:

- grief:
  - sadness: 0.015930
  - remorse: 0.004847
  - nervousness: 0.004167
  - disappointment: 0.002981
  - embarrassment: 0.002639

- pride:
  - optimism: 0.005945
  - admiration: 0.005453
  - joy: 0.003854
  - excitement: 0.003122
  - gratitude: 0.002893

- relief:
  - joy: 0.005639
  - caring: 0.004862
  - gratitude: 0.003565
  - nervousness: 0.003165
  - excitement: 0.002991

- nervousness:
  - fear: 0.046832
  - sadness: 0.006077
  - disappointment: 0.004909
  - grief: 0.004167
  - caring: 0.003208

- remorse:
  - sadness: 0.033131
  - caring: 0.008653
  - disappointment: 0.008338
  - optimism: 0.008061
  - embarrassment: 0.007126

- fear:
  - nervousness: 0.046832
  - sadness: 0.007866
  - disgust: 0.007252
  - surprise: 0.005464
  - embarrassment: 0.004469

- desire:
  - optimism: 0.029657
  - love: 0.010000
  - admiration: 0.006965
  - realization: 0.006901
  - excitement: 0.006739

### Semantic-relation construction

All 27 non-neutral GoEmotions labels had corresponding official natural-language definitions.

The definitions were encoded using `all-mpnet-base-v2`.

Resulting semantic embedding shape:

`(27, 768)`

Mean embedding norm after normalization:

`1.0`

Semantic matrix checks:

- symmetric: True
- diagonal all ones: True
- negative cosine values before clipping: 0

Off-diagonal cosine similarity statistics:

- minimum: 0.080416
- 10th percentile: 0.250171
- 25th percentile: 0.316005
- median: 0.408023
- 75th percentile: 0.491609
- 90th percentile: 0.581750
- 95th percentile: 0.628028
- maximum: 0.852502
- mean: 0.410658
- standard deviation: 0.129999
- zero fraction: 0.0000

The semantic relation matrix is substantially denser than the empirical co-occurrence structure.

The strongest semantic label pairs included:

1. grief – sadness: 0.852502
2. fear – nervousness: 0.814040
3. anger – disgust: 0.781046
4. gratitude – joy: 0.748078
5. joy – love: 0.735515
6. joy – pride: 0.735230
7. approval – disapproval: 0.693542
8. excitement – optimism: 0.689041
9. anger – annoyance: 0.687259
10. gratitude – love: 0.686284

Selected semantic tail-label neighbours included:

- grief:
  - sadness: 0.852502
  - anger: 0.648707
  - disappointment: 0.639776
  - fear: 0.603060
  - desire: 0.592873

- pride:
  - joy: 0.735230
  - gratitude: 0.649770
  - admiration: 0.607310
  - optimism: 0.601285
  - excitement: 0.596545

- relief:
  - joy: 0.527868
  - nervousness: 0.500457
  - gratitude: 0.461755
  - optimism: 0.459826
  - fear: 0.452323

- nervousness:
  - fear: 0.814040
  - embarrassment: 0.588349
  - surprise: 0.518673
  - grief: 0.518189
  - anger: 0.511474

- embarrassment:
  - nervousness: 0.588349
  - fear: 0.557656
  - remorse: 0.525830
  - surprise: 0.468732
  - anger: 0.428670

- remorse:
  - sadness: 0.557688
  - grief: 0.553530
  - anger: 0.538095
  - embarrassment: 0.525830
  - desire: 0.496004

- fear:
  - nervousness: 0.814040
  - surprise: 0.656362
  - grief: 0.603060
  - anger: 0.592756
  - embarrassment: 0.557656

- desire:
  - disappointment: 0.641336
  - anger: 0.623836
  - optimism: 0.601925
  - grief: 0.592873
  - joy: 0.583525

### Empirical vs semantic agreement

The empirical and semantic relation matrices were positively but only moderately associated across the 351 unique unordered emotion-label pairs.

Results:

- Number of unique label pairs: 351
- Pearson correlation: r = 0.3219, p = 6.621e-10
- Spearman correlation: rho = 0.2725, p = 2.154e-07

This indicates that the two relation sources capture partly overlapping but substantially non-identical information about the GoEmotions label space.

The positive association shows that they are not unrelated, while the moderate correlation suggests that the semantic matrix is not simply reproducing empirical co-occurrence.

### Normalization

Both relation matrices were independently normalized using off-diagonal min-max normalization.

Empirical normalization range:
- minimum: 0.000000
- maximum: 0.071391

Semantic normalization range:
- minimum: 0.080416
- maximum: 0.852502

After normalization:

- empirical matrix symmetric: True
- empirical off-diagonal min: 0.0
- empirical off-diagonal max: 1.0
- empirical diagonal: 0

- semantic matrix symmetric: True
- semantic off-diagonal min: 0.0
- semantic off-diagonal max: 1.0
- semantic diagonal: 0

### Exploratory hybrid results

Using `alpha = 0.5`, the empirical and semantic matrices were combined with equal weight.

The strongest hybrid relationships were:

1. anger – annoyance: 0.892989
2. fear – nervousness: 0.803090
3. disappointment – sadness: 0.689849
4. excitement – joy: 0.636979
5. anger – disgust: 0.614626
6. grief – sadness: 0.611573
7. admiration – gratitude: 0.608019
8. confusion – curiosity: 0.599616
9. gratitude – joy: 0.560662
10. joy – love: 0.555320
11. desire – optimism: 0.545437
12. remorse – sadness: 0.541119
13. annoyance – disgust: 0.517298
14. amusement – joy: 0.514521
15. annoyance – disapproval: 0.478893
16. gratitude – love: 0.471420
17. admiration – approval: 0.466524
18. annoyance – disappointment: 0.458892
19. admiration – love: 0.457544
20. caring – optimism: 0.455660

Selected hybrid tail-label neighbours were:

- grief:
  - sadness: 0.611573
  - disappointment: 0.383115
  - anger: 0.368023
  - desire: 0.341633
  - remorse: 0.340331

- pride:
  - joy: 0.451045
  - gratitude: 0.388975
  - admiration: 0.379404
  - optimism: 0.378952
  - excitement: 0.356107

- relief:
  - joy: 0.329263
  - nervousness: 0.294181
  - gratitude: 0.271922
  - optimism: 0.261898
  - love: 0.245288

- nervousness:
  - fear: 0.803090
  - embarrassment: 0.343965
  - grief: 0.312682
  - relief: 0.294181
  - sadness: 0.290947

- embarrassment:
  - nervousness: 0.343965
  - fear: 0.340361
  - remorse: 0.338356
  - surprise: 0.266921
  - disgust: 0.263142

- remorse:
  - sadness: 0.541119
  - grief: 0.340331
  - embarrassment: 0.338356
  - disappointment: 0.317739
  - anger: 0.306354

- fear:
  - nervousness: 0.803090
  - surprise: 0.411252
  - anger: 0.348017
  - embarrassment: 0.340361
  - grief: 0.338462

- desire:
  - optimism: 0.545437
  - disappointment: 0.407531
  - love: 0.384786
  - anger: 0.364628
  - joy: 0.352684

### Observations

- The empirical and semantic relation matrices have substantially different distributions.
- Empirical Jaccard relations are sparse because most GoEmotions examples contain only one annotated label.
- Semantic definition embeddings provide dense label-to-label information even when empirical co-occurrence is weak or absent.
- Several intuitively related label pairs are strong under both relation sources, including fear–nervousness and grief–sadness.
- Other relationships receive substantially different strengths from the two sources, indicating that the semantic matrix is not redundant with empirical co-occurrence.
- The moderate empirical–semantic correlation supports the use of a hybrid representation rather than either relation source alone.
- The hybrid matrix produces interpretable high-scoring relationships and preserves several known semantically close emotion pairs.
- The current `alpha = 0.5` hybrid matrix is exploratory only. The final alpha value will be selected using the validation protocol defined in the methodology.
- The empirical and semantic matrices differ strongly in sparsity and scale before normalization, supporting the use of separate normalization before combination.
- The current min-max normalization behaves sensibly on the observed matrices, but alternative normalization strategies may still be considered later as robustness checks if needed.

### Conclusion

The first implementation of the proposed structured label representation behaves as intended.

Empirical co-occurrence and semantic definition similarity provide related but complementary information about the GoEmotions label space. Their moderate correlation and different score distributions support the motivation for combining them instead of relying on only one relation source.

The exploratory hybrid matrix yields interpretable relationships, including strong connections among several semantically related and previously observed confusable emotion pairs.

These results support proceeding to the next stage of the proposed method: constructing frequency-dependent tail-aware structural weights and integrating the resulting label structure into the training objective.

---

## Experiment 54 – Tail-aware structural rarity weights

**Date:** 2026-10-15

### Goal

Construct and inspect the frequency-dependent rarity weights used by the
proposed tail-aware structural regularization method.

The objective is to give rare emotion labels stronger access to structural
information without using the much more aggressive scaling produced by direct
inverse-frequency weighting.

### Setup

- Dataset: complete GoEmotions simplified training split
- Train size: 43,410 examples
- Structural label space: 27 non-neutral emotion labels
- Label frequency source: positive training-label counts
- Tail labels:
  - grief
  - pride
  - relief
  - nervousness
  - embarrassment
  - remorse
  - fear
  - desire

For each label `i`, the raw rarity weight was defined as:

`r_i = sqrt(n_max / n_i)`

where `n_i` is the positive training frequency of label `i` and `n_max` is the
largest positive frequency among the 27 structural emotion labels.

The weights were then normalized to have unit mean:

`r_i_normalized = r_i / mean(r)`

For comparison, direct inverse-frequency weighting was also calculated:

`r_i_inverse = n_max / n_i`

followed by the same mean normalization.

### Results

The mean of the normalized square-root rarity weights was:

`1.0000`

confirming that the weighting redistributes structural emphasis without
increasing the average weight.

Normalized square-root rarity weights:

| Label | Train count | Normalized weight | Tail |
|---|---:|---:|---|
| grief | 77 | 2.9922 | yes |
| pride | 111 | 2.4921 | yes |
| relief | 153 | 2.1227 | yes |
| nervousness | 164 | 2.0503 | yes |
| embarrassment | 303 | 1.5084 | yes |
| remorse | 545 | 1.1247 | yes |
| fear | 596 | 1.0755 | yes |
| desire | 641 | 1.0371 | yes |
| disgust | 793 | 0.9324 | no |
| excitement | 853 | 0.8990 | no |
| surprise | 1060 | 0.8065 | no |
| caring | 1087 | 0.7964 | no |
| realization | 1110 | 0.7881 | no |
| disappointment | 1269 | 0.7371 | no |
| sadness | 1326 | 0.7210 | no |
| confusion | 1368 | 0.7099 | no |
| joy | 1452 | 0.6891 | no |
| anger | 1567 | 0.6633 | no |
| optimism | 1581 | 0.6603 | no |
| disapproval | 2022 | 0.5839 | no |
| love | 2086 | 0.5749 | no |
| curiosity | 2191 | 0.5609 | no |
| amusement | 2328 | 0.5442 | no |
| annoyance | 2470 | 0.5283 | no |
| gratitude | 2662 | 0.5089 | no |
| approval | 2939 | 0.4843 | no |
| admiration | 4130 | 0.4086 | no |

Group summary:

- Tail labels:
  - count: 8
  - mean normalized weight: 1.8004
  - minimum: 1.0371
  - maximum: 2.9922

- Non-tail labels:
  - count: 19
  - mean normalized weight: 0.6630
  - minimum: 0.4086
  - maximum: 0.9324

Comparison with direct inverse-frequency weighting:

- Maximum normalized square-root weight: 2.9922
- Maximum normalized inverse-frequency weight: 6.3003

For the rarest label, grief:

- square-root weight: 2.9922
- inverse-frequency weight: 6.3003

### Observations

- The square-root weighting produces the intended monotonic relationship
  between rarity and structural weight.
- The rarest labels receive substantially stronger structural emphasis, while
  frequent labels receive weights below the unit mean.
- The tail subset receives an average structural weight of approximately 1.8,
  compared with approximately 0.66 for non-tail labels.
- The weighting is continuous rather than binary: labels closer to the
  tail/non-tail boundary receive only modest additional weight.
- desire, the least rare member of the predefined tail set, receives a weight
  only slightly above 1.0, while grief receives approximately 3.0.
- This behavior is desirable because the method depends on actual training
  frequency rather than simply assigning all predefined tail labels the same
  additional weight.
- Direct inverse-frequency weighting is substantially more aggressive, with a
  maximum normalized weight above 6.3.
- The square-root transformation therefore provides a more conservative way
  to increase structural support for rare labels and reduces the risk that the
  rarest classes dominate the structural objective.
- The observed weight range appears sufficiently large to create meaningful
  frequency dependence without introducing extreme coefficients.

### Conclusion

The normalized square-root inverse-frequency weighting behaves as intended on
the full GoEmotions training distribution.

It gives substantially greater structural influence to the rarest emotion
labels while remaining considerably less aggressive than direct
inverse-frequency weighting.

The square-root rarity weighting is therefore retained as the primary
tail-aware weighting function for the proposed structural regularization
method.

The next step is to integrate these weights with the hybrid label-relation
matrix in the structural loss and verify the resulting loss numerically before
full model training.

---

## Experiment 55 – Feedback-driven redesign of the structured label method

**Date:** 2026-10-22

### Goal

Redesign the structured label method in response to the methodological concerns identified in supervisor feedback.

The main goals of this experiment were:

- replace the symmetric Jaccard-based empirical relation with a more tail-friendly directed relation;
- investigate rank-based normalization instead of min-max normalization;
- sparsify the hybrid label graph to reduce the risk of overly dense attractive regularization;
- implement an asymmetric structural loss using stop-gradient;
- verify that the asymmetric formulation produces genuinely directed label-to-label transfer.

### 1. Asymmetric empirical label structure

The previous empirical structure used Jaccard similarity:

`J(i,j) = |A_i ∩ A_j| / |A_i ∪ A_j|`

This is symmetric. It can also systematically weaken relations between rare and frequent labels because the union is dominated by the more frequent label.

The empirical relation was therefore reformulated as the conditional probability:

`P(j | i) = |A_i ∩ A_j| / |A_i|`

where `i` is interpreted as the target label and `j` as a potential source of structural information.

The relation is asymmetric:

`P(j | i) != P(i | j)`

The matrix was constructed over the 27 non-neutral GoEmotions labels.

Matrix checks:

- shape: `(27, 27)`
- minimum: `0.0`
- maximum: `0.285714`
- diagonal: `0`
- matrix construction checks passed

### Directed relation examples

Several previously identified tail/head pairs showed strong directional asymmetry:

| Relation | Conditional probability |
|---|---:|
| P(sadness \| grief) | 0.2857 |
| P(grief \| sadness) | 0.0166 |
| P(fear \| nervousness) | 0.2073 |
| P(nervousness \| fear) | 0.0570 |
| P(sadness \| remorse) | 0.1101 |
| P(remorse \| sadness) | 0.0452 |
| P(admiration \| pride) | 0.2072 |
| P(pride \| admiration) | 0.0056 |
| P(joy \| relief) | 0.0588 |
| P(relief \| joy) | 0.0062 |

The `grief -> sadness` relation is a particularly clear example. Approximately 28.6% of grief-labelled training examples also contain sadness. In the opposite direction, only approximately 1.7% of sadness examples contain grief.

This directional information was not represented by the previous symmetric Jaccard relation.

### 2. Tail-label empirical neighbours

The strongest outgoing empirical conditional relations for the predefined tail labels were:

#### grief

- sadness: 0.2857
- disappointment: 0.0519
- remorse: 0.0390
- caring: 0.0390
- annoyance: 0.0260

#### pride

- admiration: 0.2072
- optimism: 0.0901
- gratitude: 0.0721
- joy: 0.0541
- approval: 0.0450

#### relief

- gratitude: 0.0654
- joy: 0.0588
- optimism: 0.0458
- caring: 0.0392
- disapproval: 0.0261

#### nervousness

- fear: 0.2073
- sadness: 0.0549
- disappointment: 0.0427
- caring: 0.0244
- curiosity: 0.0183

#### embarrassment

- sadness: 0.0396
- curiosity: 0.0330
- disappointment: 0.0330
- realization: 0.0297
- annoyance: 0.0297

#### remorse

- sadness: 0.1101
- optimism: 0.0312
- gratitude: 0.0275
- disappointment: 0.0275
- caring: 0.0257

#### fear

- nervousness: 0.0570
- sadness: 0.0252
- admiration: 0.0235
- curiosity: 0.0185
- annoyance: 0.0185

#### desire

- optimism: 0.0998
- admiration: 0.0515
- love: 0.0421
- approval: 0.0281
- gratitude: 0.0203

### Observation

Several important rare labels have a substantially stronger relation to common labels when conditional probability is used instead of Jaccard.

Examples include:

- grief -> sadness
- pride -> admiration
- nervousness -> fear
- remorse -> sadness
- desire -> optimism

The empirical structure is still noisy for some very rare labels. For example, the top relations of embarrassment are relatively weak and less clearly interpretable. This supports retaining the semantic relation source rather than relying on empirical co-occurrence alone.

### 3. Rank-based normalization

The empirical and semantic relation matrices have very different raw distributions.

The previous method used independent min-max normalization. This is sensitive to extreme values and does not make the relative meaning of values from the two distributions directly comparable.

An exploratory percentile/rank normalization was therefore implemented.

Initial normalized ranges were:

- conditional empirical percentile range: `0.0060 – 1.0000`
- semantic percentile range: `0.0007 – 0.9993`

The non-zero lower bounds are caused by tied values under average-rank percentile normalization. This is particularly relevant for the empirical matrix because it contains many zero relations.

Therefore, the normalization implementation requires an additional constraint: raw zero empirical edges should remain exactly zero rather than receiving a small positive percentile value.

This zero-preserving rank normalization will be used in the next revision of the structure construction.

### 4. Sparse hybrid label graph

An exploratory hybrid structure was constructed by combining the percentile-normalized empirical and semantic relations with:

`alpha = 0.5`

This value remains exploratory and is not treated as an optimized hyperparameter.

To reduce the risk of a dense attractive graph pulling all classifier representations toward similar directions, row-wise top-k sparsification was applied.

Initial setting:

`k = 5`

For 27 structural labels, this produced:

- non-zero directed edges: `135`
- expected maximum: `27 * 5 = 135`

Each label therefore retained exactly five outgoing structural relations.

### Tail-label neighbours in the sparse hybrid graph

#### grief

- sadness: 0.9996
- disappointment: 0.9483
- remorse: 0.8830
- surprise: 0.7946
- desire: 0.7871

#### pride

- admiration: 0.9668
- gratitude: 0.9665
- joy: 0.9654
- optimism: 0.9533
- excitement: 0.8930

#### relief

- joy: 0.8869
- gratitude: 0.8185
- optimism: 0.7532
- love: 0.6384
- excitement: 0.6302

#### nervousness

- fear: 0.9975
- sadness: 0.8149
- anger: 0.7079
- disapproval: 0.6979
- embarrassment: 0.6687

#### embarrassment

- remorse: 0.8017
- fear: 0.7611
- anger: 0.6690
- disgust: 0.6687
- annoyance: 0.6491

#### remorse

- sadness: 0.9276
- disappointment: 0.7960
- gratitude: 0.7846
- annoyance: 0.7200
- embarrassment: 0.7044

#### fear

- nervousness: 0.9733
- surprise: 0.8363
- sadness: 0.7978
- disgust: 0.7657
- optimism: 0.7279

#### desire

- optimism: 0.9568
- love: 0.8991
- disappointment: 0.8634
- curiosity: 0.8163
- gratitude: 0.7800

### Observation

The sparse hybrid graph retained several intuitive and previously observed emotion relationships.

Examples include:

- grief -> sadness
- pride -> admiration
- relief -> joy
- nervousness -> fear
- embarrassment -> remorse
- remorse -> sadness
- fear -> nervousness
- desire -> optimism

The top-k formulation also limits the number of attractive relations per label. This directly reduces the risk identified in supervisor feedback that a dense positive semantic graph could encourage broad representation collapse.

The exact value of `k` is not yet considered final.

### 5. Asymmetric structural loss

The previous structural objective was symmetric. For a relation between labels `i` and `j`, both classifier weight vectors received gradients.

This does not implement the intended interpretation that a rare target label should receive structural support from a better-estimated source label.

An asymmetric formulation was therefore implemented using stop-gradient on the source classifier representation.

Conceptually, for a directed relation:

`tail target -> source label`

the target classifier weight is optimized toward the source representation, while the source representation is detached from this structural term.

The classification loss remains free to update all classifier weights normally.

### Gradient sanity check

A synthetic classifier weight matrix was initialized and a single directed structural edge was created:

`grief -> sadness`

Two structural losses were compared.

#### Symmetric cosine regularization

Both representations participate normally in gradient computation.

Results:

- grief gradient norm: `0.006352`
- sadness gradient norm: `0.006372`

Both endpoints received structural gradients.

#### Asymmetric stop-gradient regularization

The sadness source representation was detached.

Results:

- grief gradient norm: `0.006352`
- sadness gradient norm: `0.000000`

The gradient on grief was preserved, while the structural gradient on sadness was eliminated.

### Observation

The gradient test confirms the distinction between the two formulations.

Under symmetric regularization:

`grief <-> sadness`

Both representations are moved toward each other.

Under the asymmetric formulation:

`grief -> sadness`

The grief representation receives structural guidance from sadness, but sadness is not moved toward grief by this structural edge.

The identical grief gradient magnitude in the two tests also confirms that detaching the source does not remove the intended optimization signal from the target.

### Overall observations

The feedback-driven redesign changes several important properties of the original method.

1. **Empirical structure is now directional.**  
   Conditional probability preserves the asymmetry between rare and frequent labels that symmetric Jaccard similarity hides.

2. **The empirical relation is more compatible with tail-to-head transfer.**  
   Relations such as `grief -> sadness` and `pride -> admiration` become substantially more visible.

3. **The graph is sparse rather than fully connected.**  
   Top-k sparsification limits each label to a small number of structural neighbours and reduces the risk of representation collapse.

4. **The structural objective can be made directional.**  
   Stop-gradient ensures that a directed edge updates the target representation without moving the source representation through that structural term.

5. **Rank-based normalization is promising but requires careful zero handling.**  
   The first implementation exposed a tie-handling issue for zero empirical edges. Raw zero relations should remain zero in the final implementation.

6. **Semantic information remains useful.**  
   Conditional co-occurrence is more tail-friendly than Jaccard but remains noisy for labels with very few training examples. Combining it with semantic relations is therefore still motivated.

### Conclusion

The supervisor feedback identified several weaknesses in the first structured-label formulation. The initial redesign successfully addresses three major concerns.

First, asymmetric conditional probabilities provide a more appropriate empirical relation for rare labels than symmetric Jaccard similarity.

Second, top-k graph sparsification provides a simple mechanism for avoiding an almost fully connected attractive graph.

Third, the stop-gradient implementation successfully converts the structural objective from symmetric attraction into directed transfer. The gradient sanity check confirmed that the target label receives the structural gradient while the source label does not.

The resulting approach is therefore moving toward a sparse, asymmetric, tail-directed structural learning method.

Before full model training, the next steps are:

- finalize zero-preserving percentile normalization;
- integrate the previously defined frequency-dependent rarity weights into the asymmetric loss;
- verify the complete structural loss numerically on the full sparse graph;
- define the revised experimental matrix and fair tuning protocol;
- only then integrate the revised structural objective into DistilBERT training.

---

## Experiment 56 – Integrated tail-aware asymmetric structural loss sanity check

**Date:** 2026-10-22

### Goal

Integrate the revised components of the structured-label method and verify that the resulting structural objective is numerically well-defined before model training.

The experiment checks:

- zero-preserving percentile normalization;
- explicit separation of the 28-label classifier space and 27-label structural space;
- sparse directed hybrid structure;
- asymmetric source-detached structural transfer;
- frequency-dependent rarity weighting;
- numerical stability of the complete structural objective.

### 1. Zero-preserving percentile normalization

The initial percentile-normalization implementation assigned small positive percentile values to raw zero empirical relations because tied zero values received average ranks.

This was corrected by ranking only strictly positive off-diagonal relations. Raw zero empirical edges remain exactly zero.

Results:

- raw empirical zero edges: 86
- normalized empirical zero edges: 86
- all raw empirical zeros preserved: True
- positive empirical percentile range: 0.000813 – 1.000000
- positive semantic percentile range: 0.000713 – 0.999287
- empirical diagonal: 0
- semantic diagonal: 0

The hybrid graph was then reconstructed using the corrected normalized matrices.

With exploratory `alpha = 0.5` and row-wise `k = 5` sparsification:

- directed non-zero hybrid edges: 135
- expected directed edges: 135

### 2. Neutral-label handling

The classification and structural label spaces were explicitly separated.

Results:

- classifier labels: 28
- structural labels: 27
- neutral present in classifier space: True
- neutral present in structural space: False
- sparse hybrid structure shape: `(27, 27)`
- rarity-weight vector length: 27

All neutral-handling checks passed.

The classifier therefore retains the full GoEmotions output space, while structural regularization is applied only to the 27 emotion labels.

### 3. Integrated asymmetric structural objective

The revised structural loss combines:

- the sparse directed hybrid relation matrix;
- cosine-based structural distance;
- stop-gradient on source representations;
- frequency-dependent rarity weights;
- normalization by outgoing structural edge strength.

For a directed relation `i -> j`, classifier representation `i` is treated as the target. Representation `j` is used as a detached source representation.

The structural objective therefore changes the target through the structural term without propagating that structural gradient into the source.

### 4. Numerical sanity check

A random 27-label classifier weight matrix was used to test the complete objective before integration into DistilBERT.

Results:

- structural loss: 0.983116
- finite structural loss: True
- active structural rows: 27
- expected active rows: 27
- all per-label structural losses finite: True

Gradient checks:

- all gradients finite: True
- minimum classifier-weight gradient norm: 0.000753
- maximum classifier-weight gradient norm: 0.007225

No inactive structural rows, NaN values, infinite values, or zero-gradient failures were observed.

### 5. Tail-aware weighting behaviour

The average contributions of tail and non-tail labels were inspected separately.

| Group | Mean rarity weight | Mean raw structural loss | Mean weighted contribution |
|---|---:|---:|---:|
| Non-tail | 0.6630 | 0.9955 | 0.6598 |
| Tail | 1.8004 | 0.9661 | 1.7509 |

The raw structural mismatch was similar for the two groups under random initialization.

After rarity weighting, tail labels received substantially greater influence on the structural objective.

The average weighted structural contribution was approximately 2.65 times larger for tail labels than for non-tail labels in this sanity check.

This difference is produced by the intended frequency-dependent weighting rather than by a systematically larger raw structural mismatch for tail labels.

### 6. Tail-aware versus non-tail-aware objective

Using the same random classifier geometry and structural graph:

- non-tail-aware structural loss: 0.986762
- tail-aware structural loss: 0.983116

These scalar values are not interpreted as a performance comparison because the classifier weights are randomly initialized.

The purpose of this comparison is only to confirm that enabling rarity weighting changes the aggregation of otherwise identical per-label structural losses while remaining numerically stable.

### Observations

- Zero empirical relations remain zero after the revised percentile normalization.
- Neutral is cleanly excluded from the structural objective while remaining a classifier output.
- Every structural label has five outgoing relations in the current sparse graph.
- The complete structural loss is finite and differentiable.
- All 27 target representations receive finite gradients.
- The previously verified source-side stop-gradient prevents structural gradients from propagating back through source representations.
- Frequency-dependent weighting substantially increases the relative contribution of rare labels without modifying their raw structural mismatch.
- Mean-normalized rarity weights preserve an approximately comparable overall regularization scale while redistributing emphasis toward rare labels.

### Conclusion

The revised structural objective passes the implementation-level sanity checks required before integration into the classifier training pipeline.

The method now combines a sparse directed hybrid label graph with asymmetric source-detached transfer and frequency-dependent tail weighting. Neutral is explicitly excluded from structural regularization, and zero empirical relations remain zero after normalization.

The implementation is numerically stable on the complete 27-label structural graph.

The next methodology tasks are to define collapse diagnostics and control structures, establish a fair hyperparameter-tuning protocol, redesign the experimental matrix, and specify the statistical evaluation procedure before full model training.

---

## Experiment 57 – Structural control matrices

**Date:** 2026-10-22

### Goal

Construct control label-relation matrices for later training experiments.

The purpose of these controls is to determine whether improvements from structural regularization are caused by meaningful emotion-label structure rather than by generic regularization or graph connectivity alone.

Two primary controls were constructed:

1. a randomly permuted version of the real sparse hybrid structure;
2. a uniform-weight version using the same edge locations as the real sparse hybrid structure.

A dense uniform graph was also constructed as an optional diagnostic control.

### 1. Real sparse hybrid structure

The reference structure was the revised sparse hybrid graph produced by the previous experiments.

Properties:

- structural labels: 27
- outgoing edges per label: 5
- directed non-zero edges: 135
- density: 0.192308
- mean positive edge weight: 0.808651
- standard deviation of positive weights: 0.116486
- minimum positive weight: 0.384669
- maximum positive weight: 0.999643

### 2. Permuted-structure control

A fixed random label permutation was generated using seed 42.

The same permutation was applied to both axes of the structural matrix. This preserves the graph topology and complete edge-weight distribution while assigning the relations to different emotion labels.

Checks:

- real non-zero edges: 135
- permuted non-zero edges: 135
- real total edge-weight sum: 109.167836
- permuted total edge-weight sum: 109.167836
- sorted positive edge weights identical: True

The complete positive-weight distribution therefore remains unchanged.

Graph summary:

| Graph | Edges | Density | Mean weight | Std. weight | Min weight | Max weight |
|---|---:|---:|---:|---:|---:|---:|
| Real hybrid top-5 | 135 | 0.192308 | 0.808651 | 0.116486 | 0.384669 | 0.999643 |
| Permuted | 135 | 0.192308 | 0.808651 | 0.116486 | 0.384669 | 0.999643 |

Example permuted neighbours illustrate that the original semantic interpretation is no longer preserved.

Examples:

- grief -> surprise, confusion, realization, desire, fear
- nervousness -> amusement, curiosity, optimism, love, grief
- fear -> confusion, approval, sadness, desire, grief
- desire -> approval, confusion, grief, realization, caring

The permuted structure therefore provides a control with the same numerical edge distribution but intentionally incorrect label associations.

### 3. Uniform same-edge control

A second control retained exactly the same sparsity pattern as the real hybrid graph but replaced every positive relation weight with 1.

Checks:

- non-zero edges: 135
- unique positive weight: 1.0
- sparsity pattern identical to real graph: True

Graph summary:

| Graph | Edges | Density | Mean weight | Std. weight | Min weight | Max weight |
|---|---:|---:|---:|---:|---:|---:|
| Uniform same-edge | 135 | 0.192308 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |

This control preserves which labels are connected while removing information encoded in the relative edge strengths.

It can therefore test whether weighted structural information provides additional value beyond graph connectivity alone.

### 4. Optional dense uniform control

A fully connected directed uniform graph was also constructed.

Properties:

- labels: 27
- diagonal: 0
- directed non-zero edges: 702
- expected edges: `27 * 26 = 702`

This graph is retained as an optional diagnostic control.

Unlike the primary controls, it does not preserve the sparsity of the proposed method. Results obtained with this graph would therefore combine the effects of uniform relations and substantially increased graph density.

### 5. Structural-loss sanity check

The real and control structures were evaluated using the same randomly initialized classifier geometry and the same tail-aware asymmetric structural loss.

Results:

| Graph | Structural loss | Active rows | Finite loss | Finite gradients | Mean gradient norm | Max gradient norm |
|---|---:|---:|---|---|---:|---:|
| Real hybrid top-5 | 0.983116 | 27 | True | True | 0.002099 | 0.007225 |
| Permuted | 0.974110 | 27 | True | True | 0.002134 | 0.007105 |
| Uniform same-edge | 0.981982 | 27 | True | True | 0.002097 | 0.007153 |

All three structures are numerically valid for later training experiments.

The absolute loss values are not interpreted as a comparison of structure quality because the classifier weights are randomly initialized.

### Observations

- The permuted control exactly preserves edge count, density, total edge weight, and the complete positive edge-weight distribution of the real graph.
- Label semantics are intentionally destroyed by the permutation.
- The uniform same-edge control exactly preserves graph connectivity while removing relative edge-strength information.
- Both controls are compatible with the proposed tail-aware asymmetric structural loss.
- All 27 structural rows remain active under all primary control conditions.
- Gradient magnitudes are of similar scale across the three graph variants.
- The dense uniform graph is available as an additional diagnostic but should not be treated as an equally controlled comparison because its sparsity differs substantially.

### Experimental interpretation

Future training comparisons can separate several explanations for a possible performance improvement.

If:

`real structure > permuted structure`

this supports the hypothesis that meaningful label associations matter.

If:

`real weighted structure > uniform same-edge structure`

this supports the hypothesis that relation strengths provide useful information beyond connectivity alone.

If the controls perform similarly to the real structure, then improvements may be caused primarily by generic structural regularization rather than by the proposed empirical-semantic label representation.

### Conclusion

The primary structural controls required for later ablation experiments have been successfully constructed and validated.

The real, permuted, and uniform same-edge graphs provide controlled comparisons for testing whether meaningful emotion-label structure and weighted relation strengths contribute to performance.

These controls will be included in the revised experimental design.

---

## Methodology decision – Fair tuning and revised experimental design

**Date:** 2026-10-22

### Motivation

Supervisor feedback identified two related weaknesses in the original experimental plan:

1. the proposed method had a larger validation-tuning budget than the baseline models;
2. the experiment matrix did not cleanly isolate the effects of structural learning, tail-aware weighting, and imbalance-aware classification.

The experimental protocol was redesigned to address both concerns.

### Fair tuning protocol

The main methodology choices will be specified before final model comparison rather than exhaustively selected using validation Tail Macro-F1.

The following choices are fixed across the main experiments:

- base encoder: DistilBERT;
- common optimization and training settings;
- structural graph sparsity: `k = 5`;
- empirical relation: directed conditional probability `P(j | i)`;
- zero-preserving percentile normalization;
- square-root inverse-frequency rarity weighting;
- source-detached asymmetric structural loss;
- primary hybrid coefficient: `alpha = 0.5`.

The values `alpha = 0` and `alpha = 1` will be used as source-ablation conditions rather than as validation-selected alternatives.

The structural-loss coefficient will use a small predefined development grid:

`beta ∈ {0.01, 0.1, 1.0}`

Beta will be selected once on a designated structured development configuration and then fixed for the remaining structured experiments.

Decision-threshold tuning will use the same predefined global threshold grid and the same validation-selection rule for every model.

This design reduces the number of validation-selected degrees of freedom available only to the proposed method.

### Revised main experiment matrix

The main experiments follow a two-factor design.

Classification objective:

- BCE
- Distribution-Balanced Loss

Structural treatment:

- no structural regularization;
- hybrid structural regularization without rarity weighting;
- hybrid structural regularization with tail-aware rarity weighting.

This produces six main conditions:

| ID | Classification objective | Structure | Tail-aware weighting |
|---|---|---|---|
| E0 | BCE | none | – |
| E1 | BCE | hybrid | no |
| E2 | BCE | hybrid | yes |
| E3 | DB Loss | none | – |
| E4 | DB Loss | hybrid | no |
| E5 | DB Loss | hybrid | yes |

Key comparisons:

- E1 vs E0: effect of structural learning under BCE;
- E2 vs E1: incremental effect of tail-aware weighting under BCE;
- E3 vs E0: effect of imbalance-aware classification;
- E4 vs E3: effect of structural learning under DB Loss;
- E5 vs E4: incremental effect of tail-aware weighting under DB Loss;
- E5 vs E2: interaction between imbalance-aware classification and the proposed structural method.

### Structure-source ablation

The contribution of empirical and semantic relations will be tested separately under a fixed structured configuration.

| ID | Structural source |
|---|---|
| A0 | empirical only (`alpha = 1`) |
| A1 | semantic only (`alpha = 0`) |
| A2 | hybrid (`alpha = 0.5`) |

The hybrid condition can reuse the corresponding main experiment result and therefore does not require a duplicate training run.

### Structural controls

The following graph controls will be compared under an otherwise identical structured configuration:

| ID | Structural graph |
|---|---|
| C0 | real sparse hybrid graph |
| C1 | randomly permuted graph |
| C2 | uniform-weight graph with identical edge locations |

The real graph can reuse the corresponding main experiment result.

The controls test whether performance improvements depend on meaningful label structure rather than generic sparse regularization.

### Training-budget strategy

Development experiments may initially use a single seed for implementation validation and configuration screening.

Repeated-seed evaluation will be reserved for the final principal comparisons rather than applied to every exploratory ablation.

This avoids multiplying every diagnostic configuration by the full repeated-run budget while still allowing robust final statistical evaluation.

### Conclusion

The revised design removes redundancies from the original experiment matrix and provides direct comparisons for the individual contributions of imbalance-aware classification, structural learning, and tail-aware weighting.

The tuning protocol also reduces the proposed method's validation advantage by fixing most methodological choices in advance and limiting validation-based structural hyperparameter selection.

---

## Methodology note – Tail-label split robustness and statistical evaluation

**Date:** 2026-10-22

### Motivation

Supervisor feedback highlighted that several rare GoEmotions labels contain very few positive validation and test examples. This can make Tail Macro-F1 highly sensitive to individual predictions.

The official GoEmotions split was therefore inspected before finalizing the statistical evaluation protocol.

### Official split

The official simplified GoEmotions splits contain:

- train: 43,410 examples
- validation: 5,426 examples
- test: 5,427 examples

This corresponds approximately to an 80/10/10 split.

Tail-label positive counts are:

| Label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| grief | 77 | 13 | 6 | 96 |
| pride | 111 | 15 | 16 | 142 |
| relief | 153 | 18 | 11 | 182 |
| nervousness | 164 | 21 | 23 | 208 |
| embarrassment | 303 | 35 | 37 | 375 |
| remorse | 545 | 68 | 56 | 669 |
| fear | 596 | 90 | 78 | 764 |
| desire | 641 | 77 | 83 | 801 |

The official split approximately preserves the global split proportions. However, the smallest classes still contain very few test positives.

The clearest examples are:

- grief: 6 test positives
- relief: 11 test positives
- pride: 16 test positives

For grief, only 6.25% of all positive examples occur in the test split instead of the approximately 10% expected from the global split ratio.

### Interpretation

No large part of the dataset is unused. The official train, validation, and test sets together cover the complete simplified dataset used in the experiments.

The problem is instead statistical instability.

For labels such as grief, changing the prediction outcome of only one or two test examples can substantially alter the label-level F1 score. Tail Macro-F1 can therefore vary noticeably because of the particular allocation of rare positive examples.

### Revised evaluation strategy

The official GoEmotions split will remain the primary benchmark. This preserves comparability with previous work.

Final principal models will be evaluated using five fixed random seeds. Results will be reported as mean and standard deviation.

The main reported metrics will include:

- Tail Macro-F1
- Micro-F1
- Macro-F1
- Tail Macro Average Precision
- Macro-MCC

Paired bootstrap resampling will be used to quantify uncertainty in the differences between principal models.

Planned bootstrap protocol:

- 10,000 bootstrap resamples
- resampling unit: complete test example
- identical sampled indices for both models in each paired comparison
- 95% percentile confidence intervals

Bootstrap samples in which a required tail label has no positive example will not be treated as having zero F1. Such samples will be rejected and resampled.

Primary paired comparisons will include:

- proposed BCE model vs BCE baseline
- proposed DB-Loss model vs DB-Loss baseline
- proposed DB-Loss model vs proposed BCE model

### Iterative-stratified robustness evaluation

The official split alone may not provide sufficient evidence that rare-label improvements are stable across different allocations of the small tail classes.

A secondary robustness analysis will therefore use repeated multi-label iterative-stratified splits.

The complete simplified dataset will be repartitioned several times using approximately:

- 80% training
- 10% validation
- 10% test

The splitting procedure will attempt to preserve the frequency of every label across the three subsets simultaneously.

This is particularly important for the rare emotion labels.

Five independently seeded iterative-stratified splits are planned for the robustness analysis.

These splits will not replace the official benchmark and will not be used for additional hyperparameter tuning. Hyperparameters selected using the main development protocol will remain fixed.

The purpose of the repeated splits is only to test whether the conclusions remain stable when rare examples are distributed more evenly across training, validation, and test sets.

### Cross-validation decision

Full nested or 10-fold cross-validation was considered but is not included in the primary AI Lab protocol.

Transformer fine-tuning combined with multiple models, seeds, and hyperparameter choices would make full cross-validation computationally expensive.

Repeated iterative-stratified splits provide a more practical robustness test for the specific concern in this project: instability caused by the allocation of rare multi-label examples.

More extensive cross-validation can be considered later for the thesis or a publication-oriented extension.

### Conclusion

The official GoEmotions split is suitable as the primary benchmark but contains very small test counts for some rare labels.

The final evaluation will therefore combine:

1. the official benchmark split;
2. repeated-seed evaluation;
3. paired bootstrap confidence intervals;
4. repeated iterative-stratified robustness splits.

This provides stronger evidence than relying on a single Tail Macro-F1 value from the official test set.

---

## Experiment 58 – Per-label threshold calibration baseline

**Date:** 2026-10-26

### Goal

Test whether the poor performance of rare emotion labels can be substantially improved through label-specific decision calibration without changing or retraining the underlying BCE classifier.

This experiment addresses the possibility that part of the long-tail performance gap is caused by label-specific prediction bias rather than inadequate learned representations.

### Setup

The existing full-data DistilBERT + BCE baseline was reused without additional training.

Validation and test probabilities were generated from the same trained classifier.

A separate decision threshold was optimized for every label using validation predictions.

Instead of using a coarse fixed threshold grid, the candidate thresholds were taken from the actual validation precision--recall curve for each label.

For each label \(i\), the threshold maximizing validation F1 was selected.

The selected thresholds were then applied unchanged to the test set.

No test information was used during threshold selection.

### Selected tail-label thresholds

| Label | Threshold | Validation F1 |
|---|---:|---:|
| grief | 0.0100 | 0.0541 |
| pride | 0.0322 | 0.7200 |
| relief | 0.0184 | 0.1639 |
| nervousness | 0.1191 | 0.4242 |
| embarrassment | 0.1243 | 0.5672 |
| remorse | 0.2530 | 0.7547 |
| fear | 0.1156 | 0.6596 |
| desire | 0.4747 | 0.5378 |

The large variation in selected thresholds indicates substantial label-specific calibration differences.

### Test results

The per-label calibrated BCE model achieved:

- Micro-F1: **0.5949**
- Macro-F1: **0.4972**
- Tail Macro-F1: **0.4089**
- Tail Macro-Precision: **0.4677**
- Tail Macro-Recall: **0.4482**
- Tail Macro-AP: **0.3777**

For comparison, the same BCE model with the previously selected global threshold of 0.15 achieved:

- Micro-F1: 0.5739
- Macro-F1: 0.4626
- Tail Macro-F1: 0.3212

Thus, label-specific calibration improved Tail Macro-F1 by approximately:

\[
0.4089 - 0.3212 = 0.0877.
\]

This corresponds to an approximate 27\% relative increase in Tail Macro-F1 without retraining the classifier.

### Tail-label test results

| Label | Threshold | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| grief | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 0.1018 |
| pride | 0.0322 | 0.7500 | 0.3750 | 0.5000 | 0.3602 |
| relief | 0.0184 | 0.2222 | 0.7273 | 0.3404 | 0.3096 |
| nervousness | 0.1191 | 0.4615 | 0.2609 | 0.3333 | 0.2424 |
| embarrassment | 0.1243 | 0.6000 | 0.4054 | 0.4839 | 0.3517 |
| remorse | 0.2530 | 0.5287 | 0.8214 | 0.6434 | 0.5952 |
| fear | 0.1156 | 0.4914 | 0.7308 | 0.5876 | 0.6930 |
| desire | 0.4747 | 0.6875 | 0.2651 | 0.3826 | 0.4594 |

### Observations

Per-label threshold calibration substantially improves both overall and rare-label performance.

The result shows that a significant part of the apparent long-tail problem in the plain BCE classifier is related to label-specific decision calibration.

The largest conceptual change is that rare labels often require substantially lower decision thresholds than frequent labels.

Relief is a particularly clear example. A coarse minimum threshold of 0.05 produced zero test F1, while validation-optimal thresholding selected approximately 0.018 and produced a test F1 of 0.3404.

However, calibration alone does not solve all rare-label failures.

Grief remains at zero test F1 despite a very low selected threshold. Its Tail AP is also low, indicating that its problem cannot be explained purely by the final decision threshold.

Tail Macro-AP remains unchanged by threshold optimization because the underlying probability ranking is unchanged. It therefore provides an important threshold-independent metric for evaluating the proposed structural method.

### Conclusion

Per-label calibration is a strong and necessary baseline.

The proposed structured model should therefore not be considered successful merely because it outperforms the original global-threshold BCE Tail Macro-F1 of 0.3212.

A stronger comparison is the calibrated BCE result of approximately:

\[
F1_{\mathrm{tail}} = 0.4089.
\]

Threshold-independent Tail Macro Average Precision should also be monitored to determine whether structural learning improves the underlying ranking of rare labels rather than only changing decision calibration.

The very small validation counts of the rarest labels may make individually optimized thresholds unstable. This further motivates the planned repeated iterative-stratified robustness evaluation.

---

## Experiment 59 – Distribution-Balanced Loss baseline

**Date:** 2026-10-26

### Goal

Evaluate Distribution-Balanced Loss as a strong imbalance-aware baseline for rare-emotion recognition in the full 28-label multi-label GoEmotions setting.

The purpose of this experiment is to determine whether an imbalance-aware objective can improve rare-label recognition beyond the plain BCE baseline while preserving overall classification performance.

### Setup

The same DistilBERT architecture and training configuration as the plain BCE baseline were used.

The only main change was the classification objective.

Distribution-Balanced Loss was implemented as a standalone PyTorch module following the rebalanced, negative-tolerant formulation and the COCO-style configuration:

- reweighting: rebalance
- focal balance parameter: 2.0
- focal gamma: 2.0
- negative scale: 2.0
- initial bias factor: 0.05
- mapping parameters:
  - alpha = 0.1
  - beta = 10.0
  - gamma = 0.2

No oversampling was used, so the training data and sampling procedure remained identical to the BCE baseline.

Training configuration:

- DistilBERT
- learning rate: \(2\times10^{-5}\)
- batch size: 16
- evaluation batch size: 32
- epochs: 3
- weight decay: 0.01
- best checkpoint selected by validation loss

The best checkpoint was obtained after epoch 2.

### Implementation sanity checks

Before full training, the DB Loss implementation was checked on a real GoEmotions batch.

Results:

- logits shape: \(16\times28\)
- labels shape: \(16\times28\)
- loss was finite
- backward propagation completed successfully
- all existing gradients were finite
- 104 model parameters received gradients

The integrated Trainer sanity check also produced a finite initial DB loss.

### Training

Validation losses:

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.028815 | 0.026872 |
| 2 | 0.025357 | **0.026085** |
| 3 | 0.022608 | 0.026306 |

The epoch-2 checkpoint was selected.

### Global-threshold evaluation

A global threshold was selected on the validation set using Macro-F1, following the same procedure as the BCE baseline.

The selected threshold was:

\[
t = 0.45
\]

Test performance:

- Micro-F1: **0.5926**
- Macro-F1: **0.5294**
- Tail Macro-F1: **0.5005**
- Tail Macro-Precision: **0.6047**
- Tail Macro-Recall: **0.5005**
- Macro-MCC: **0.5178**
- Tail Macro-AP: **0.4490**

For comparison, the plain BCE model with its validation-selected global threshold achieved:

- Micro-F1: 0.5739
- Macro-F1: 0.4626
- Tail Macro-F1: 0.3212
- Macro-MCC: 0.4504
- Tail Macro-AP: 0.3777

DB Loss therefore improved both overall and rare-label performance.

### Per-label calibration

The same exact validation-based per-label threshold calibration used for the BCE baseline was also applied to DB Loss.

Results:

- Micro-F1: **0.6049**
- Macro-F1: **0.5128**
- Tail Macro-F1: **0.4519**
- Tail Macro-Precision: **0.5889**
- Tail Macro-Recall: **0.4355**
- Macro-MCC: **0.5066**
- Tail Macro-AP: **0.4490**

Unlike the BCE baseline, per-label thresholding did not improve Tail Macro-F1 or Macro-MCC relative to the global-threshold DB configuration.

This suggests that individually optimized thresholds may overfit the validation set for very rare labels, where only a small number of positive examples are available.

### Tail-label calibrated results

| Label | Test F1 | Test AP |
|---|---:|---:|
| grief | 0.2857 | 0.3568 |
| pride | 0.4800 | 0.3399 |
| relief | 0.2500 | 0.3522 |
| nervousness | 0.2424 | 0.2761 |
| embarrassment | 0.5000 | 0.4223 |
| remorse | 0.6950 | 0.6063 |
| fear | 0.6489 | 0.7291 |
| desire | 0.5132 | 0.5093 |

The improvement in grief is particularly notable. The BCE baseline produced zero test F1 for grief even after calibration, whereas DB Loss produced non-zero F1 and substantially higher AP.

### Comparison with calibration-only improvement

The calibrated BCE baseline achieved a Tail Macro-F1 of 0.4089.

DB Loss with a single global threshold achieved:

\[
F1_{\text{tail}} = 0.5005
\]

This indicates that the DB Loss improvement cannot be explained solely by threshold calibration.

Tail Macro-AP also increased from:

\[
0.3777 \rightarrow 0.4490
\]

Because average precision is threshold-independent, this provides evidence that DB Loss improves the underlying ranking of rare-label predictions rather than merely changing the final decision boundary.

### Conclusion

Distribution-Balanced Loss is a substantially stronger imbalance-aware baseline than both the plain BCE model and the earlier oversampling + weighting baseline.

It improves rare-label performance without producing the large Micro-F1 degradation observed in the earlier OW experiment.

The strongest current baseline is therefore:

\[
\text{DistilBERT + DB Loss + global threshold } 0.45
\]

with:

- Micro-F1 = 0.5926
- Macro-F1 = 0.5294
- Tail Macro-F1 = 0.5005
- Tail Macro-AP = 0.4490
- Macro-MCC = 0.5178

This establishes a significantly stronger target for the proposed structured label-learning method.

The next step is to integrate the revised hybrid structural regularization into training and evaluate:

- E1: BCE + structure
- E2: BCE + structure + tail weighting

followed by the corresponding DB Loss variants E4 and E5.

---

