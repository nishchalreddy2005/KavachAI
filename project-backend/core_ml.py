import pandas as pd
import numpy as np

# Placeholder imports for models - assuming models are pre-trained and saved elsewhere in project
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix

# Load pre-trained models and preprocessors (you should retrain and save these yourself)
# Here, simplified for demonstration - load from pickle or reinitialize models as needed

# Example loading placeholders - replace with actual model loading
# gnb_model = joblib.load('gnb_model.pkl')
# dt_model = joblib.load('dt_model.pkl')
# xgb_model = joblib.load('xgb_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca.pkl')

# For demonstration, mock a predict output class
class PredictionResult:
    def __init__(self, status, confidence, details):
        self.status = status
        self.confidence = confidence
        self.details = details


def preprocess_logs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the CSV DataFrame from the frontend CSV upload.
    This should map the raw CSV columns and convert or engineer features as required
    to be compatible with the trained models.
    """
    # Example: Map CSV column names to the features expected by the model
    # For demonstration, assume numeric encoding or extraction here
    # This must be adapted to your model's expected input features.
    
    # Dummy preprocessing - select numeric columns or transform appropriately
    # In real use, do feature extraction, encoding, scaling, PCA etc.

    # Example: Just select numeric columns for illustration
    processed_df = df.select_dtypes(include=[np.number]).copy()
    
    # Fill NA or preprocess as needed for your model
    processed_df.fillna(0, inplace=True)
    
    return processed_df


def predict_csv(df: pd.DataFrame) -> PredictionResult:
    # Lazily build models and preprocessors once
    global _MODELS_READY
    try:
        if not _MODELS_READY:
            _build_models()
    except NameError:
        _MODELS_READY = False
        _build_models()

    # Normalize columns
    df = df.copy()
    df.columns = df.columns.str.strip()

    total_rows = len(df)
    if total_rows == 0:
        return PredictionResult("warning", 60.0, "Empty CSV provided.")

    # Prepare inputs for each model using the SAME preprocessing learned from training data
    # GNB: numeric -> scaler_gnb -> pca_gnb
    Xnum_infer = df.select_dtypes(include=[np.number]).copy()
    Xnum_infer.fillna(0, inplace=True)
    # Align numeric columns to training numeric columns
    missing_num_cols = [c for c in _NUMERIC_COLUMNS if c not in Xnum_infer.columns]
    for c in missing_num_cols:
        Xnum_infer[c] = 0
    Xnum_infer = Xnum_infer[_NUMERIC_COLUMNS]
    X_gnb = _pca_gnb.transform(_scaler_gnb.transform(Xnum_infer))

    # DT: numeric + one-hot of protocol/service/flag learned categories
    X_dt_num = Xnum_infer.to_numpy()
    X_dt_cat = _ohe_dt.transform(_extract_cat(df))
    if hasattr(X_dt_cat, 'toarray'):
        X_dt_cat = X_dt_cat.toarray()
    X_dt = np.hstack((np.asarray(X_dt_num, dtype=float), np.asarray(X_dt_cat, dtype=float)))

    # XGB: numeric (scaled+pca) + one-hot of protocol/service/flag
    X_xgb_num = _pca_xgb.transform(_scaler_xgb.transform(Xnum_infer))
    X_xgb_cat = _ohe_xgb.transform(_extract_cat(df))
    # Build dense horizontally then convert to CSR to avoid type issues
    X_xgb_num_dense = X_xgb_num if not hasattr(X_xgb_num, 'toarray') else X_xgb_num.toarray()
    X_xgb_cat_dense = X_xgb_cat if not hasattr(X_xgb_cat, 'toarray') else X_xgb_cat.toarray()
    X_xgb_combined = np.hstack((np.asarray(X_xgb_num_dense, dtype=float), np.asarray(X_xgb_cat_dense, dtype=float)))
    X_xgb = csr_matrix(X_xgb_combined)

    # Predict with three models
    preds_gnb = _gnb_clf.predict(X_gnb)
    preds_dt = _dt_clf.predict(X_dt)
    preds_xgb = _xgb_clf.predict(X_xgb)

    # Majority voting per row; in our convention: 1=normal, 0=anomalous
    votes = np.vstack([preds_gnb, preds_dt, preds_xgb]).T
    # bincount expects non-negative ints
    ensemble = np.apply_along_axis(lambda r: np.bincount(r.astype(int), minlength=2).argmax(), 1, votes)

    # Compute rates
    anomalies = int((ensemble == 0).sum())
    anomaly_ratio = anomalies / float(total_rows)

    if anomaly_ratio == 0.0:
        status = "safe"
        confidence = 95.0
        details = "No suspicious activity detected by ensemble."
    elif anomaly_ratio < 0.2:
        status = "warning"
        confidence = float(65.0 + anomaly_ratio * 100.0)
        details = f"Low-level anomalies by ensemble: {int(anomaly_ratio * 100)}% suspicious."
    else:
        status = "danger"
        confidence = float(min(99.0, 85.0 + (anomaly_ratio - 0.2) * 70.0))
        details = f"High anomaly rate by ensemble: {int(anomaly_ratio * 100)}% suspicious."

    # Add brief per-model signals
    gnb_anom = int((preds_gnb == 0).sum())
    dt_anom = int((preds_dt == 0).sum())
    xgb_anom = int((preds_xgb == 0).sum())
    details = f"{details} (GNB: {gnb_anom}, DT: {dt_anom}, XGB: {xgb_anom} anomalies out of {total_rows})."

    return PredictionResult(status, confidence, details)


# ---- Internal training on cleaned dataset to build reproducible preprocessing and models ----
def _extract_cat(df):
    n = len(df)
    protocol = df["protocol_type"] if "protocol_type" in df.columns else pd.Series([None] * n)
    service = df["service"] if "service" in df.columns else pd.Series([None] * n)
    flag = df["flag"] if "flag" in df.columns else pd.Series([None] * n)
    protocol = protocol.astype(str).fillna("").to_numpy().reshape(-1, 1)
    service = service.astype(str).fillna("").to_numpy().reshape(-1, 1)
    flag = flag.astype(str).fillna("").to_numpy().reshape(-1, 1)
    return np.hstack([protocol, service, flag])


def _build_models() -> None:
    """Train GNB, DT, XGB once using `clean_dataset.pkl` and cache preprocessors."""
    global _MODELS_READY
    global _scaler_gnb, _pca_gnb, _gnb_clf
    global _scaler_xgb, _pca_xgb, _ohe_xgb, _xgb_clf
    global _ohe_dt, _dt_clf
    global _NUMERIC_COLUMNS

    # Load cleaned dataset prepared in IDS pipeline
    clean = pd.read_pickle('clean_dataset.pkl')
    clean.columns = clean.columns.str.strip()

    # Define binary target: 1 for normal, 0 for others
    if 'intrusion_type' in clean.columns:
        col = clean['intrusion_type']
        labels = pd.Series(col, index=clean.index, dtype='string').astype('string').fillna('').str.strip()
    else:
        labels = pd.Series(['normal'] * len(clean), index=clean.index, dtype='string')
    y = labels.apply(lambda v: 1 if v == 'normal.' or v == 'normal' else 0).to_numpy()

    # Numeric columns for models
    X_num = clean.select_dtypes(include=[np.number]).copy()
    X_num.fillna(0, inplace=True)
    _NUMERIC_COLUMNS = list(X_num.columns)

    # Fit GNB pipeline (scaler + PCA by variance threshold)
    _scaler_gnb = StandardScaler()
    X_num_scaled = _scaler_gnb.fit_transform(X_num)
    _pca_gnb = PCA(n_components=0.98)
    X_gnb_train = _pca_gnb.fit_transform(X_num_scaled)
    _gnb_clf = GaussianNB()
    _gnb_clf.fit(X_gnb_train, y)

    # DT pipeline: numeric + OHE for 3 categoricals
    _ohe_dt = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_fit = _extract_cat(clean)
    X_dt_cat = _ohe_dt.fit_transform(X_cat_fit)
    X_dt_train = np.hstack([X_num.values, X_dt_cat])
    _dt_clf = DecisionTreeClassifier(random_state=42, max_depth=None)
    _dt_clf.fit(X_dt_train, y)

    # XGB pipeline: numeric scaled + PCA + OHE
    _scaler_xgb = StandardScaler()
    X_num_scaled_xgb = _scaler_xgb.fit_transform(X_num)
    _pca_xgb = PCA(n_components=0.98)
    X_num_pca_xgb = _pca_xgb.fit_transform(X_num_scaled_xgb)
    _ohe_xgb = OneHotEncoder(handle_unknown='ignore')
    X_xgb_cat = _ohe_xgb.fit_transform(X_cat_fit)
    # Ensure dense combination for training matrix
    X_xgb_cat_dense = X_xgb_cat if not hasattr(X_xgb_cat, 'toarray') else X_xgb_cat.toarray()
    X_xgb_train = np.hstack((np.asarray(X_num_pca_xgb, dtype=float), np.asarray(X_xgb_cat_dense, dtype=float)))
    _xgb_clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    _xgb_clf.fit(X_xgb_train, y)

    _MODELS_READY = True


def predict(log_data: dict) -> dict:
    """
    Predict on a single log entry using the same ensemble as predict_csv.
    Accepts a dict of features and returns status/confidence/details.
    """
    try:
        if isinstance(log_data, dict):
            df = pd.DataFrame([log_data])
            result = predict_csv(df)
            return {"status": result.status, "confidence": result.confidence, "details": result.details}
        return {"status": "error", "confidence": 0, "details": "Unsupported log_data type; expected JSON object."}
    except Exception as e:
        return {"status": "error", "confidence": 0, "details": f"Prediction failed: {str(e)}"}
