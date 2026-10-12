# Experimental Band-Gap Prediction and Model Reliability

This notebook starter is a guide for the same workflow implemented in `src/train_model.py`.
Run the script first so the dataset and environment are checked consistently.

## Research questions
- How well can composition-derived descriptors predict experimental band gaps?
- Does the error change when reduced chemical formulae are held out as groups?
- What limitations remain when using composition alone?

## Suggested analysis
1. Inspect dataset size, target distribution, missing values, and formula counts.
2. Explain Magpie descriptors in your own words.
3. Compare the mean baseline against Random Forest using MAE, RMSE and R².
4. Compare random and formula-grouped evaluation without assuming either is perfect.
5. Inspect the actual-versus-predicted plots and feature importance.
6. Read the dataset source and write a limitations section.

**Important:** Do not claim a scientific finding until you have run the code, verified the data and understood the evaluation.
