# Project Guide: Python Implementation of Baseline Method


**How to use:**
```bash
# Independent test
python reproduce_baseline.py --balancing "_SMOTED" --regression "True" --use_cv "False" --feature_count 2200 --svm_cost 10

# 10-fold CV
python reproduce_baseline.py --balancing "_SMOTED" --regression "True" --use_cv "True" --n_folds 10 --feature_count 2800 --svm_cost 100

# Jackknife
python reproduce_baseline.py --balancing "_SMOTED" --use_cv "True" --n_folds -1 --regression "True" --feature_count 2800 --svm_cost 100
```

**Best Results:**
- **Independent Test**: 95.31% accuracy (2200 features, C=10) - matches paper's 95.3%
- **10-fold CV**: 98.62% accuracy (2800 features, C=100)
- **Jackknife**: 98.62% accuracy (2800 features, C=100)
