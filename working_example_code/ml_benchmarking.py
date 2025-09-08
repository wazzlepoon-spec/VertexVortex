import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from tensorflow.keras import layers, models



# fingerprints_raw: list of dicts, 1D/2D arrays, lists per sample

g = joblib.load('data/fingerprints_the_model.joblib')

def somerep(f):
    lst = []
    lst.append(f['symbolic_trace'])
    lst.append([x[0]*x[1] for x in f['cm'][0]])
    return lst

#agent data in data folder. original csv raw data files are too big too upload.
agents = ['alf3_1024_C', 'alf3_1024_D', 'alf3_256_C', 'alf3_256_D', 'alf3_512_C'
          , 'alf3_512_D', 'hurt0_128_C', 'hurt0_128_D', 'hurt0_256_C']

fingerprints_rawraw = [g[x] for x in agents]
fingerprints_raw = [somerep(x) for x in fingerprints_rawraw]


# agents with names starting with 'a' are angr/ai agents and
# agents with names starting with 'h' are atomicRedTeam/human agents
# labels: humans 1; ai 0

labels = [1 if x.startswith('h') else 0 for x in agents]

# ----- 1. Flatten fingerprint input -----

def flatten_fingerprints(raw_fingerprints, input_len=256):
    flat = []
    for fp in raw_fingerprints:
        if isinstance(fp, dict):
            vec = DictVectorizer(sparse=False)
            flat = vec.fit_transform(raw_fingerprints)
            return StandardScaler().fit_transform(flat)
        elif isinstance(fp, list) or isinstance(fp, np.ndarray):
            fp = np.asarray(fp).flatten()
            padded = np.zeros(input_len)
            padded[:min(input_len, len(fp))] = fp[:input_len]
            flat.append(padded)
        else:
            1/0
    flat = np.array(flat)
    return StandardScaler().fit_transform(flat)

# ----- 2. build cnn and transformer -----

def build_1dcnn(input_len):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(64, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_transformer(input_len, d_model=64, num_heads=4, ff_dim=128):
    inputs = layers.Input(shape=(input_len, 1))
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----- 3. cross-validation -----

def run_models_cv(fingerprints_raw, labels, input_len=256, folds=5):
    X = flatten_fingerprints(fingerprints_raw, input_len)
    y = np.array(labels)

    print(f"Running {folds}-fold stratified CV on {len(y)} samples")

    # --- Random Forest ---
    print("\n Random Forest")
    rf_preds = cross_val_predict(RandomForestClassifier(n_estimators=100), X, y, cv=folds)
    print(classification_report(y, rf_preds))

    # --- XGBoost ---
    print("\n XGBoost")
    xgb_preds = cross_val_predict(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X, y, cv=folds)
    print(classification_report(y, xgb_preds))

    # --- 1D CNN ---
    print("\n 1D CNN")
    cnn_preds, cnn_true = neural_cv(X, y, build_1dcnn, input_len, folds)
    print(classification_report(cnn_true, cnn_preds))

    # --- Transformer ---
    print("\n Transformer")
    tr_preds, tr_true = neural_cv(X, y, build_transformer, input_len, folds)
    print(classification_report(tr_true, tr_preds))

def neural_cv(X, y, model_builder, input_len, folds):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    preds_all = []
    true_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/{folds}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        model = model_builder(input_len)
        model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=0)

        preds = (model.predict(X_test) > 0.5).astype(int).flatten()
        preds_all.extend(preds)
        true_all.extend(y_test)

    return np.array(preds_all), np.array(true_all)


run_models_cv(fingerprints_raw, labels, input_len=256, folds=5)

if 0:
    # EAM performance from runs described in paper.
    y_true=[0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred=[0,0,0,0,0,0,1,1,1,1,1,0,1,0]


