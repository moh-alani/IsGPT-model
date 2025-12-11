"""
Utility functions for baseline reproduction in Python
Includes functions to read RDS files and helper functions
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


def read_rds(filepath):
    """
    Read R RDS file. 
    Note: This is a simplified version. For full RDS support, 
    you may need to use pyreadr or rpy2, or convert RDS to pickle/CSV first.
    
    If RDS reading fails, try converting in R first:
    R: saveRDS(data, "file.rds") -> readRDS("file.rds") -> write.csv(data, "file.csv")
    """
    # First try pickle (in case file was already converted)
    try:
        pkl_file = filepath.replace('.rds', '.pkl')
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    
    try:
        # Try using pyreadr if available
        import pyreadr
        result = pyreadr.read_r(filepath)
        # pyreadr returns a dict, get the first (and usually only) dataframe
        if result and len(result) > 0:
            return list(result.values())[0]
        else:
            raise ValueError(f"No data found in {filepath}")
    except ImportError:
        # Fallback: try pickle (won't work for RDS but might for some files)
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except:
            raise ImportError(
                f"Could not read {filepath}. "
                "Please install pyreadr: pip install pyreadr, "
                "or convert RDS files to CSV/pickle format first."
            )
    except Exception as e:
        raise ValueError(f"Error reading {filepath}: {e}")


def feature_filtering(features_df, ranked_features, max_feature_count=np.inf):
    """
    Filter features based on ranking.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame with features and 'protection' column
    ranked_features : list
        List of feature names in order of importance
    max_feature_count : int or float
        Maximum number of features to keep (np.inf for all)
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with selected features
    """
    max_feature_count = min(max_feature_count, len(ranked_features))
    
    # Get columns to keep
    feature_filter = ranked_features[:int(max_feature_count)].copy()
    feature_filter.append('protection')
    
    # Get all columns in features_df
    columns = list(features_df.columns)
    
    # Remove columns not in filter
    if len(columns) - 1 > max_feature_count:
        columns_to_remove = [col for col in columns if col not in feature_filter]
        features_df = features_df.drop(columns=columns_to_remove, errors='ignore')
    
    # Ensure all selected features exist (add with 0 if missing)
    for feature in feature_filter:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # Reorder columns to match feature_filter order, then add protection
    feature_cols = [f for f in feature_filter if f != 'protection']
    other_cols = [c for c in features_df.columns if c not in feature_filter]
    features_df = features_df[feature_cols + other_cols + ['protection']]
    
    return features_df


def calculate_metrics(y_true, y_pred, y_pred_proba=None, regression_mode=False):
    """
    Calculate performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (or scores for regression)
    y_pred_proba : array-like, optional
        Predicted probabilities (for regression threshold finding)
    regression_mode : bool
        If True, treat as regression and find optimal threshold
    
    Returns:
    --------
    dict
        Dictionary with metrics: acc, sens, spec, mcc, auc (if regression), threshold (if regression)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if regression_mode:
        # For regression, find optimal threshold
        if y_pred_proba is None:
            y_pred_proba = y_pred
        
        # Find optimal threshold based on accuracy (matching ROCR's method)
        # Use a fine-grained search similar to ROCR
        min_pred = np.min(y_pred_proba)
        max_pred = np.max(y_pred_proba)
        # Create a range of thresholds (more granular than just unique values)
        if len(np.unique(y_pred_proba)) > 100:
            # Use percentiles for efficiency
            thresholds = np.percentile(y_pred_proba, np.linspace(0, 100, 201))
        else:
            # Use all unique values plus interpolated values
            thresholds = np.unique(y_pred_proba)
            # Add interpolated values between unique thresholds
            if len(thresholds) > 1:
                interp_thresholds = []
                for i in range(len(thresholds) - 1):
                    interp_thresholds.extend(np.linspace(thresholds[i], thresholds[i+1], 10))
                thresholds = np.unique(np.concatenate([thresholds, interp_thresholds]))
        
        best_threshold = 0.5
        best_acc = 0
        
        for thresh in thresholds:
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5
        
        # Use optimal threshold for final predictions
        y_pred_binary = (y_pred_proba >= best_threshold).astype(int)
        y_pred = y_pred_binary
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'acc': acc,
            'sens': sens,
            'spec': spec,
            'mcc': mcc,
            'auc': auc,
            'threshold': best_threshold
        }
    else:
        # Classification mode
        # Convert predictions to integers if they're not already
        y_pred = y_pred.astype(int)
        y_true = y_true.astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'acc': acc,
            'sens': sens,
            'spec': spec,
            'mcc': mcc
        }


def svm_cv(X, y, svm_cost=1.0, n_folds=10, kernel='linear', regression_mode=False):
    """
    Perform cross-validation with SVM.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix
    y : pandas.Series or numpy.array
        Target labels
    svm_cost : float
        SVM C parameter
    n_folds : int
        Number of CV folds (-1 for leave-one-out)
    kernel : str
        SVM kernel type
    regression_mode : bool
        If True, use SVR instead of SVC
    
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    from sklearn.model_selection import KFold, LeaveOneOut
    
    X = np.array(X) if isinstance(X, pd.DataFrame) else X
    y = np.array(y) if isinstance(y, pd.Series) else y
    
    # Handle jackknife (leave-one-out)
    if n_folds < 0:
        cv = LeaveOneOut()
        n_folds = len(y)
    else:
        # Match R's sequential CV split: folds = seq(from=1,to=N, by=round(N/cross))
        # R doesn't shuffle, so we use shuffle=False
        cv = KFold(n_splits=n_folds, shuffle=False)
    
    pred_vector = []
    y_true_vector = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if regression_mode:
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            # Scale features to match R's scale=TRUE behavior
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = SVR(kernel=kernel, C=svm_cost)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model = SVC(kernel=kernel, C=svm_cost, probability=False)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        
        pred_vector.extend(pred)
        y_true_vector.extend(y_test)
    
    pred_vector = np.array(pred_vector)
    y_true_vector = np.array(y_true_vector)
    
    # Calculate metrics
    if regression_mode:
        # For regression, we need probabilities for threshold finding
        # Re-fit on full data to get probability predictions
        from sklearn.svm import SVR
        model_full = SVR(kernel=kernel, C=svm_cost)
        model_full.fit(X, y)
        # For SVR, predictions are continuous, use them directly
        y_pred_proba = model_full.predict(X)
        metrics = calculate_metrics(y_true_vector, pred_vector, y_pred_proba, regression_mode=True)
    else:
        metrics = calculate_metrics(y_true_vector, pred_vector, regression_mode=False)
    
    return metrics

