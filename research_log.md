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