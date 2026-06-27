"""
Oil Spill Binary Classification
Dataset: Kaggle Oil Spill Detection (feature-extracted from satellite imagery)
Target: 1 = Oil Spill, 0 = No Oil Spill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load & Split Data
# ─────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/oil_spill.csv')
X = df.drop(columns=['target'])
y = df['target']

print("=" * 60)
print("OIL SPILL CLASSIFICATION REPORT")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Class Distribution:")
print(f"  Non-Oil Spill (0): {(y==0).sum()} samples ({(y==0).mean()*100:.1f}%)")
print(f"  Oil Spill     (1): {(y==1).sum()} samples ({(y==1).mean()*100:.1f}%)")

# Train / Validation / Test split  (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 2. Class Weights (handle imbalance)
# ─────────────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
print(f"\nClass Weights: {class_weight}")

# ─────────────────────────────────────────────
# 3. Define Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            max_depth=10, random_state=42, n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ))
    ]),
    "MLP Neural Network": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam',
            max_iter=500, early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ]),
}

# ─────────────────────────────────────────────
# 4. Train & Evaluate All Models
# ─────────────────────────────────────────────
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy':  accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall':    recall_score(y_val, y_pred, zero_division=0),
        'f1':        f1_score(y_val, y_pred, zero_division=0),
        'cm':        confusion_matrix(y_val, y_pred),
    }
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, proba)
        results[name]['roc'] = (fpr, tpr, auc(fpr, tpr))

# ─────────────────────────────────────────────
# 5. Pick Best Model (by F1 — good for imbalance)
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best = results[best_name]
print(f"\nBest model (by F1 on validation): {best_name}")

# Final evaluation on test set
best_model = best['model']
y_test_pred = best_model.predict(X_test)
print("\n" + "─" * 60)
print(f"FINAL TEST SET RESULTS — {best_name}")
print("─" * 60)
print(f"  Accuracy : {accuracy_score(y_test, y_test_pred)*100:.2f}%")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=['Non-Oil Spill', 'Oil Spill']))

# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
palette = {"oil": "#e63946", "no_oil": "#457b9d", "bg": "#f8f9fa",
           "grid": "#dee2e6", "text": "#212529"}

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(palette["bg"])
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (a) Class distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = y.value_counts().sort_index()
bars = ax0.bar(['Non-Oil\nSpill (0)', 'Oil\nSpill (1)'],
               counts.values,
               color=[palette["no_oil"], palette["oil"]],
               edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontsize=12, fontweight='bold')
ax0.set_title('Class Distribution', fontsize=13, fontweight='bold', pad=10)
ax0.set_ylabel('Count'); ax0.set_facecolor(palette["bg"])
ax0.grid(axis='y', color=palette["grid"], linewidth=0.7)
ax0.spines[['top','right']].set_visible(False)

# ── (b) Model Comparison Bar Chart
ax1 = fig.add_subplot(gs[0, 1:])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.2
colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']
for i, (met, col) in enumerate(zip(['accuracy','precision','recall','f1'], colors)):
    vals = [results[m][met] for m in model_names]
    bars = ax1.bar(x + i*width, vals, width, label=metrics_names[i],
                   color=col, alpha=0.85, edgecolor='white')
ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels(model_names, fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.set_title('Model Comparison (Validation Set)', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_facecolor(palette["bg"])
ax1.grid(axis='y', color=palette["grid"], linewidth=0.7)
ax1.spines[['top','right']].set_visible(False)
ax1.set_ylabel('Score')

# ── (c) Confusion matrices for all models
for idx, (name, res) in enumerate(results.items()):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[1, col])
    cm = res['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Spill', 'Spill'],
                yticklabels=['No Spill', 'Spill'],
                ax=ax, linewidths=0.5, linecolor='white',
                cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'{name}\nF1={res["f1"]:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.tick_params(labelsize=9)

# ── (d) ROC Curves
ax_roc = fig.add_subplot(gs[2, :2])
roc_colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']
for (name, res), col in zip(results.items(), roc_colors):
    if 'roc' in res:
        fpr, tpr, roc_auc = res['roc']
        ax_roc.plot(fpr, tpr, color=col, lw=2,
                    label=f'{name} (AUC={roc_auc:.3f})')
ax_roc.plot([0,1],[0,1],'--', color='gray', linewidth=1)
ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.02])
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('ROC Curves (Validation Set)', fontsize=13, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=9)
ax_roc.set_facecolor(palette["bg"])
ax_roc.grid(color=palette["grid"], linewidth=0.7)
ax_roc.spines[['top','right']].set_visible(False)

# ── (e) Feature Importance (Random Forest)
ax_fi = fig.add_subplot(gs[2, 2])
rf = results['Random Forest']['model'].named_steps['clf']
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
colors_fi = [palette["oil"] if v == top10.max() else palette["no_oil"] for v in top10.values]
top10.plot(kind='barh', ax=ax_fi, color=colors_fi[::-1], edgecolor='white')
ax_fi.set_title('Top 10 Feature Importances\n(Random Forest)', fontsize=11, fontweight='bold')
ax_fi.set_xlabel('Importance', fontsize=9)
ax_fi.invert_yaxis()
ax_fi.set_facecolor(palette["bg"])
ax_fi.grid(axis='x', color=palette["grid"], linewidth=0.7)
ax_fi.spines[['top','right']].set_visible(False)

plt.suptitle('Oil Spill Detection — ML Classification Analysis',
             fontsize=16, fontweight='bold', y=1.01, color=palette["text"])

plt.savefig('/mnt/user-data/outputs/oil_spill_results.png',
            dpi=150, bbox_inches='tight', facecolor=palette["bg"])
plt.close()
print("\n✅ Plot saved to oil_spill_results.png")
