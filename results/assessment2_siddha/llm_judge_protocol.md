# Manual LLM-as-Judge Protocol

## Purpose
The LLM-as-judge evaluation complements the standard ground-truth metrics by assessing the semantic severity of TinyHAR classification errors on the held-out SIDDHA test set. It does not replace accuracy or Macro-F1.

## Sample selection
The existing SIDDHA test analysis was used to export all held-out predictions. Only misclassified windows were eligible. A deterministic sample of 30 errors was selected with `pandas.DataFrame.sample(n=30, random_state=2026)`.

The judge received only the true activity label and the predicted activity label. Raw sensor values, model confidence scores, and subject metadata were not shown to the judge.

## Judge
- Interface: ChatGPT
- Model: GPT-5.6 Sol
- Evaluation mode: manual, in a single conversation
- Date: 2026-11-08

## Fixed rubric
Because all selected samples are errors, the rubric uses three severity levels:

1. **Minor error** — The activities are closely related in movement or posture, so the confusion is relatively understandable.
2. **Moderate error** — The activities are meaningfully different, but share some movement, posture, or contextual characteristics.
3. **Severe error** — The activities are clearly different in movement or posture, and the prediction substantially misrepresents the true activity.

## Fixed prompt
> You are evaluating classification errors from a Human Activity Recognition model.
>
> For each example, you will receive:
> - the true human activity
> - the activity predicted by the model
>
> Your task is to assess the semantic severity of the classification error using this fixed rubric:
>
> 1 = Minor error: the activities are closely related in terms of movement or posture, so the confusion is relatively understandable.
>
> 2 = Moderate error: the activities are meaningfully different, but share some movement, posture, or contextual characteristics.
>
> 3 = Severe error: the activities are clearly different in movement or posture, and the prediction substantially misrepresents the true activity.
>
> Evaluate only the relationship between the true and predicted activities. Do not speculate about information that is not provided.
>
> Return:
> Severity: [1, 2, or 3]
> Reason: [one short sentence]
>
> True activity: {TRUE_ACTIVITY}
> Predicted activity: {PREDICTED_ACTIVITY}

## Summary of results
- Samples evaluated: 30
- Minor errors (1): 16 / 30 (53.3%)
- Moderate errors (2): 14 / 30 (46.7%)
- Severe errors (3): 0 / 30 (0.0%)
- Mean severity: 1.47 / 3

These results suggest that, within this deterministic sample, TinyHAR's errors were usually between activities sharing some movement, posture, or context rather than between completely unrelated activities. This should be interpreted as a qualitative complement to the quantitative test metrics, not as an alternative measure of predictive correctness.

## Limitations
The judgement is model- and prompt-dependent, and the rubric necessarily involves subjective interpretation. The sample contains only 30 misclassified windows. Repeated evaluation with multiple judges or multiple LLMs would provide a stronger estimate of agreement and robustness.
