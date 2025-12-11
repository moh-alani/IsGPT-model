"""
Baseline Reproduction Script (Python Version)
This script reproduces the baseline method from the paper exactly.

TO REPRODUCE THE BASELINE:
1. Set the parameters below according to the paper's baseline method
2. Ensure the required data files exist (or convert RDS files to CSV/pickle)
3. Run this script: python reproduce_baseline.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import read_rds, feature_filtering, calculate_metrics, svm_cv

# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

RANDOM_SEED = 10
np.random.seed(RANDOM_SEED)

# Data balancing: "" for no SMOTE (imbalanced baseline), "_SMOTED" for SMOTE
# Can be overridden via command line: python reproduce_baseline.py --balancing "_SMOTED" --use_cv True --n_folds 10
import argparse
parser = argparse.ArgumentParser(description='Reproduce baseline results')
parser.add_argument('--balancing', type=str, default="", help='Data balancing: "" or "_SMOTED"')
parser.add_argument('--use_cv', type=str, default="False", help='Use cross-validation: "True" or "False"')
parser.add_argument('--n_folds', type=int, default=10, help='Number of CV folds (-1 for jackknife)')
parser.add_argument('--regression', type=str, default="False", help='Use regression mode: "True" or "False"')
parser.add_argument('--feature_count', type=int, default=None, help='Fixed feature count (overrides search)')
parser.add_argument('--svm_cost', type=float, default=None, help='Fixed SVM cost C (overrides search)')
args, unknown = parser.parse_known_args()

BALANCING = args.balancing if args.balancing else ""  # BASELINE: Typically no SMOTE

# Feature scheme: "_comb" for combined features, "_comb_pseAAC" for with pseAAC, etc.
F_SCHEME = "_comb"  # BASELINE: Check paper for exact feature set

# Classification mode: False for classification, True for regression
DO_REGRESSION = args.regression.lower() == "true" if hasattr(args, 'regression') else False  # BASELINE: Typically classification mode

# SVM parameters
# For regression with SMOTE, search for optimal C value
# Reduced search: [1, 10, 30, 100] for faster execution
# Full search: [0.3, 1, 3, 10, 30, 100] for complete search
# If fixed C is provided via command line, use that
if args.svm_cost is not None:
    SVM_COST_LIST = [args.svm_cost]
else:
    REDUCED_SEARCH = True  # Set to False for full parameter search
    if DO_REGRESSION and BALANCING == "_SMOTED":
        if REDUCED_SEARCH:
            SVM_COST_LIST = [1, 10, 30, 100]  # 4 values - faster
        else:
            SVM_COST_LIST = [0.3, 1, 3, 10, 30, 100]  # 6 values - complete
    else:
        SVM_COST_LIST = [1.0]
SVM_KERNEL = "linear"  # BASELINE: linear kernel

# Feature selection
# For regression with SMOTE, search for optimal feature count
# Paper uses "selected features" - we'll search from 2800 down to find best
# Reduced search space: [2800, 2500, 2250, 2000, 1750, 1500] for faster execution
# Full search: list(range(2800, 1499, -50)) for complete search (slower)
# If fixed feature count is provided via command line, use that
if args.feature_count is not None:
    FEATURE_COUNT_LIST = [args.feature_count]
else:
    REDUCED_SEARCH = True  # Set to False for full parameter search
    if DO_REGRESSION and BALANCING == "_SMOTED":
        if REDUCED_SEARCH:
            FEATURE_COUNT_LIST = [2800, 2500, 2250, 2000, 1750, 1500]  # 6 values - faster
        else:
            FEATURE_COUNT_LIST = list(range(2800, 1499, -50))  # 27 values - complete
    else:
        FEATURE_COUNT_LIST = [2800]

# Cross-validation settings
# For independent test: set USE_CV = False
# For cross-validation: set USE_CV = True and N_FOLDS = number of folds
# For jackknife: set USE_CV = True and N_FOLDS = -1
USE_CV = args.use_cv.lower() == "true" if hasattr(args, 'use_cv') else False
N_FOLDS = args.n_folds if hasattr(args, 'n_folds') else 10

# Feature ranking method: "rf" for Random Forest, "manual" if ranking file exists
FEATURE_RANKING_METHOD = "rf"

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def print_config():
    """Print configuration summary."""
    print("=" * 50)
    print("BASELINE REPRODUCTION")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Balancing: {BALANCING}")
    print(f"  Feature Scheme: {F_SCHEME}")
    print(f"  Regression Mode: {DO_REGRESSION}")
    print(f"  SVM Cost: {', '.join(map(str, SVM_COST_LIST))}")
    print(f"  SVM Kernel: {SVM_KERNEL}")
    print(f"  Feature Count: {', '.join(map(str, FEATURE_COUNT_LIST))}")
    print(f"  Use CV: {USE_CV}")
    if USE_CV:
        print(f"  CV Folds: {N_FOLDS}")
    print("=" * 50)


def load_data():
    """Load feature files."""
    # File names
    file_name_suffix = f"{F_SCHEME}{BALANCING}.rds"
    
    ranked_features_file = "rankedFeatures.rds"
    feature_file = f"featurized{file_name_suffix}"
    test_feature_file = f"testFeaturized{F_SCHEME}.rds"
    
    # Check if files exist
    if not os.path.exists(feature_file):
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}\n"
            "Please run featurization first or check the file path.\n"
            "You may need to convert RDS files from the R directory."
        )
    
    print(f"{datetime.now()} >> Reading training set features from {feature_file}...")
    try:
        features = read_rds(feature_file)
    except Exception as e:
        print(f"Error reading RDS file. Trying alternative method...")
        print(f"Error: {e}")
        print("\nYou may need to convert RDS files to CSV/pickle first.")
        print("See convert_rds_to_csv.py for conversion script.")
        raise
    
    print(f"{datetime.now()} >> Done")
    
    test_features = None
    if not USE_CV:
        if not os.path.exists(test_feature_file):
            raise FileNotFoundError(
                f"Test feature file not found: {test_feature_file}\n"
                "Please run featurization first or check the file path."
            )
        print(f"{datetime.now()} >> Reading test set features from {test_feature_file}...")
        test_features = read_rds(test_feature_file)
        print(f"{datetime.now()} >> Done")
    
    return features, test_features, ranked_features_file


def load_or_create_feature_ranking(features, ranked_features_file):
    """Load feature ranking or create it using Random Forest."""
    print(f"{datetime.now()} >> Reading feature ranking from {ranked_features_file}...")
    
    if os.path.exists(ranked_features_file):
        try:
            ranked_features = read_rds(ranked_features_file)
            # If it's a DataFrame, convert to list
            if isinstance(ranked_features, pd.DataFrame):
                ranked_features = ranked_features.iloc[:, 0].tolist()
            elif isinstance(ranked_features, pd.Series):
                ranked_features = ranked_features.tolist()
            print(f"{datetime.now()} >> Done")
            return ranked_features
        except Exception as e:
            print(f"Error reading ranking file: {e}")
            print("Will generate new ranking...")
    
    if FEATURE_RANKING_METHOD == "rf":
        print(f"{datetime.now()} >> Computing feature ranking using Random Forest...")
        
        # Use all features for ranking if FEATURE_COUNT_LIST contains inf
        max_rank_features = (
            features.shape[1] - 1 
            if any(np.isinf(FEATURE_COUNT_LIST)) 
            else int(max(FEATURE_COUNT_LIST))
        )
        
        # Prepare data for ranking
        feature_cols = [col for col in features.columns if col != 'protection']
        ranking_set = features[feature_cols[:max_rank_features] + ['protection']].copy()
        
        # Convert protection to numeric if needed
        if ranking_set['protection'].dtype == 'object' or ranking_set['protection'].dtype.name == 'category':
            ranking_set['protection'] = pd.Categorical(ranking_set['protection']).codes
        
        X_rank = ranking_set.drop('protection', axis=1)
        y_rank = ranking_set['protection']
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        rf_model.fit(X_rank, y_rank)
        
        # Get feature importance and rank
        importances = rf_model.feature_importances_
        feature_names = X_rank.columns.tolist()
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        ranked_features = [name for name, _ in feature_importance]
        
        # Save ranking (as pickle for now, can convert to RDS if needed)
        ranking_pickle = ranked_features_file.replace('.rds', '.pkl')
        import pickle
        with open(ranking_pickle, 'wb') as f:
            pickle.dump(ranked_features, f)
        print(f"{datetime.now()} >> Done. Saved to {ranking_pickle}")
        
        return ranked_features
    else:
        raise ValueError(
            f"Feature ranking file not found: {ranked_features_file}\n"
            "Please generate it first or set FEATURE_RANKING_METHOD = 'rf'."
        )


def preprocess_data(features, test_features=None):
    """Preprocess data for training."""
    # Random shuffle to match R's behavior (R code does: features <- features[sample(nrow(features)),])
    # This affects CV fold assignments
    features = features.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Convert to regression mode if needed
    if DO_REGRESSION:
        # Cis-Golgi becomes 1 and Trans-Golgi becomes 2.
        # But we want Cis-Golgi (positive class) to be 1 and Trans-Golgi to be 0
        if features['protection'].dtype == 'object' or features['protection'].dtype.name == 'category':
            # Convert factor to numeric: Cis-Golgi=1, Trans-Golgi=2
            features['protection'] = pd.Categorical(features['protection']).codes + 1
        features['protection'] = 2 - features['protection'].astype(float)
        
        if test_features is not None:
            if test_features['protection'].dtype == 'object' or test_features['protection'].dtype.name == 'category':
                test_features['protection'] = pd.Categorical(test_features['protection']).codes + 1
            test_features['protection'] = 2 - test_features['protection'].astype(float)
    else:
        # For classification, ensure protection is numeric
        if features['protection'].dtype == 'object' or features['protection'].dtype.name == 'category':
            features['protection'] = pd.Categorical(features['protection']).codes
        
        if test_features is not None:
            if test_features['protection'].dtype == 'object' or test_features['protection'].dtype.name == 'category':
                test_features['protection'] = pd.Categorical(test_features['protection']).codes
    
    return features, test_features


def run_cross_validation(features, ranked_features):
    """Run cross-validation evaluation."""
    print(f"{datetime.now()} >> Entering cross validation. Folds = {N_FOLDS} ...")
    
    # Jackknife
    n_folds = len(features) if N_FOLDS < 0 else N_FOLDS
    
    # Reduce feature vectors to max size for efficiency
    if not any(np.isinf(FEATURE_COUNT_LIST)):
        features = feature_filtering(features, ranked_features, max(FEATURE_COUNT_LIST))
    
    acc_data = []
    best_perf = None
    best_params = None
    
    for max_feature_count in FEATURE_COUNT_LIST:
        if np.isinf(max_feature_count):
            training_set = features.copy()
        else:
            training_set = feature_filtering(features, ranked_features, max_feature_count)
        
        X_train = training_set.drop('protection', axis=1)
        y_train = training_set['protection']
        
        for svm_c in SVM_COST_LIST:
            perf = svm_cv(X_train, y_train, svm_cost=svm_c, n_folds=n_folds, 
                         kernel=SVM_KERNEL, regression_mode=DO_REGRESSION)
            
            if DO_REGRESSION:
                print(f"{max_feature_count}, {svm_c}, {perf['auc']:.4f}, {perf['threshold']:.4f}, "
                      f"{perf['acc']:.4f}, {perf['spec']:.4f}, {perf['sens']:.4f}, {perf['mcc']:.4f}", end="")
                acc_data.append([max_feature_count, svm_c, perf['auc'], perf['threshold'],
                                perf['acc'], perf['spec'], perf['sens'], perf['mcc']])
            else:
                print(f"{max_feature_count}, {svm_c}, {perf['acc']:.4f}, {perf['sens']:.4f}, "
                      f"{perf['spec']:.4f}, {perf['mcc']:.4f}", end="")
                acc_data.append([max_feature_count, svm_c, perf['acc'], perf['sens'],
                                perf['spec'], perf['mcc']])
            
            # Save results
            out_file = f"baseline_out{F_SCHEME}{BALANCING}.csv"
            if DO_REGRESSION:
                df_results = pd.DataFrame(acc_data, columns=['FeatureCount', 'SVMCost', 'AUC', 
                                                              'Threshold', 'Accuracy', 'Specificity', 
                                                              'Sensitivity', 'MCC'])
            else:
                df_results = pd.DataFrame(acc_data, columns=['FeatureCount', 'SVMCost', 'Accuracy',
                                                             'Sensitivity', 'Specificity', 'MCC'])
            df_results.to_csv(out_file, index=False)
            
            if best_perf is None or best_perf['acc'] < perf['acc']:
                best_perf = perf
                best_params = {'maxFeatureCount': max_feature_count, 'svmC': svm_c}
                print(" <-- BEST", end="")
            
            print()
    
    # Print best results
    print("\n" + "=" * 50)
    print("BASELINE RESULTS (Cross-Validation)")
    print("=" * 50)
    print(f"Best Result for <nF, C> = {best_params['maxFeatureCount']}, {best_params['svmC']}")
    if DO_REGRESSION:
        print(f"AUCROC                : {best_perf['auc']:.4f}")
        print(f"Threshold             : {best_perf['threshold']:.4f}")
        # Swap sens and spec for interpretation
        best_perf['sens'], best_perf['spec'] = best_perf['spec'], best_perf['sens']
    print(f"Accuracy (Overall)    : {best_perf['acc']:.4f}")
    print(f"Accuracy (Trans-Golgi): {best_perf['sens']:.4f}")
    print(f"Accuracy (Cis-Golgi)  : {best_perf['spec']:.4f}")
    print(f"MCC                   : {best_perf['mcc']:.4f}")
    print("=" * 50)


def run_independent_test(features, test_features, ranked_features):
    """Run independent test set evaluation."""
    print(f"{datetime.now()} >> Entering independent test evaluation ...")
    
    # Reduce feature vectors to max size for efficiency
    if not any(np.isinf(FEATURE_COUNT_LIST)):
        features = feature_filtering(features, ranked_features, max(FEATURE_COUNT_LIST))
        test_features = feature_filtering(test_features, ranked_features, max(FEATURE_COUNT_LIST))
    
    acc_data = []
    best_perf = None
    best_params = None
    
    for max_feature_count in FEATURE_COUNT_LIST:
        if np.isinf(max_feature_count):
            training_set = features.copy()
            test_set = test_features.copy()
        else:
            training_set = feature_filtering(features, ranked_features, max_feature_count)
            test_set = feature_filtering(test_features, ranked_features, max_feature_count)
        
        X_train = training_set.drop('protection', axis=1)
        y_train = training_set['protection']
        X_test = test_set.drop('protection', axis=1)
        y_test = test_set['protection']
        
        for svm_c in SVM_COST_LIST:
            # Train model
            if DO_REGRESSION:
                from sklearn.preprocessing import StandardScaler
                # Scale features to match R's scale=TRUE behavior
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = SVR(kernel=SVM_KERNEL, C=svm_c)
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            else:
                model = SVC(kernel=SVM_KERNEL, C=svm_c)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            # Calculate metrics
            if DO_REGRESSION:
                # For regression, use predictions as probabilities for threshold finding
                metrics = calculate_metrics(y_test, pred, y_pred_proba=pred, regression_mode=True)
                print(f"{max_feature_count}, {svm_c}, {metrics['auc']:.4f}, {metrics['threshold']:.4f}, "
                      f"{metrics['acc']:.4f}, {metrics['spec']:.4f}, {metrics['sens']:.4f}, {metrics['mcc']:.4f}")
                acc_data.append([max_feature_count, svm_c, metrics['auc'], metrics['threshold'],
                                metrics['acc'], metrics['spec'], metrics['sens'], metrics['mcc']])
            else:
                metrics = calculate_metrics(y_test, pred, regression_mode=False)
                print(f"{max_feature_count}, {svm_c}, {metrics['acc']:.4f}, {metrics['sens']:.4f}, "
                      f"{metrics['spec']:.4f}, {metrics['mcc']:.4f}")
                acc_data.append([max_feature_count, svm_c, metrics['acc'], metrics['sens'],
                                metrics['spec'], metrics['mcc']])
            
            # Save results
            out_file = f"baseline_out{F_SCHEME}{BALANCING}.csv"
            if DO_REGRESSION:
                df_results = pd.DataFrame(acc_data, columns=['FeatureCount', 'SVMCost', 'AUC',
                                                              'Threshold', 'Accuracy', 'Specificity',
                                                              'Sensitivity', 'MCC'])
            else:
                df_results = pd.DataFrame(acc_data, columns=['FeatureCount', 'SVMCost', 'Accuracy',
                                                             'Sensitivity', 'Specificity', 'MCC'])
            df_results.to_csv(out_file, index=False)
            
            # Store results for final summary
            if best_perf is None:
                best_perf = metrics
                best_params = {'maxFeatureCount': max_feature_count, 'svmC': svm_c}
    
    # Print results
    print("\n" + "=" * 50)
    print("BASELINE RESULTS (Independent Test)")
    print("=" * 50)
    print(f"Parameters: <nF, C> = {best_params['maxFeatureCount']}, {best_params['svmC']}")
    if DO_REGRESSION:
        print(f"AUCROC                : {best_perf['auc']:.4f}")
        print(f"Threshold             : {best_perf['threshold']:.4f}")
        # Swap sens and spec for interpretation
        best_perf['sens'], best_perf['spec'] = best_perf['spec'], best_perf['sens']
    print(f"Accuracy (Overall)    : {best_perf['acc']:.4f}")
    print(f"Accuracy (Trans-Golgi): {best_perf['sens']:.4f}")
    print(f"Accuracy (Cis-Golgi)  : {best_perf['spec']:.4f}")
    print(f"MCC                   : {best_perf['mcc']:.4f}")
    print("=" * 50)


def main():
    """Main function."""
    print_config()
    
    # Load data
    features, test_features, ranked_features_file = load_data()
    
    # Load or create feature ranking
    ranked_features = load_or_create_feature_ranking(features, ranked_features_file)
    
    # Preprocess data
    features, test_features = preprocess_data(features, test_features)
    
    # Run evaluation
    if USE_CV:
        run_cross_validation(features, ranked_features)
    else:
        run_independent_test(features, test_features, ranked_features)
    
    out_file = f"baseline_out{F_SCHEME}{BALANCING}.csv"
    print(f"\n{datetime.now()} >> Results saved to {out_file}")
    print(f"{datetime.now()} >> Baseline reproduction complete.")


if __name__ == "__main__":
    # Change to script directory to find data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_dir = os.path.join(script_dir, 'R')
    
    # Try to use R directory for data files
    if os.path.exists(r_dir):
        os.chdir(r_dir)
        print(f"Changed to directory: {r_dir}")
    
    main()

