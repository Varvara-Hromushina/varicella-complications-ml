import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
df = pd.read_excel("осложнения.xlsx")
df.columns = columns

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, "
      f"осложнений в тесте: {y_test.sum()}")

smote_k_values = [3, 5, 7]
rf_params = [
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1},
]

best_f1_cv, best_config, best_pipeline = 0, None, None

for k in smote_k_values:
    for params in rf_params:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=k)),
            ('classifier', RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        mean_f1 = cv_scores.mean()
        print(f"k={k}, n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}: F1={mean_f1:.3f}")
        if mean_f1 > best_f1_cv:
            best_f1_cv, best_config, best_pipeline = mean_f1, (k, params), pipeline

print(f"\nЛучшая конфигурация: k={best_config[0]}, "
      f"n_estimators={best_config[1]['n_estimators']}, "
      f"max_depth={best_config[1]['max_depth']}")
print(f"Лучший CV F1: {best_f1_cv:.3f}")


oof_proba = cross_val_predict(
    best_pipeline, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    method='predict_proba'
)[:, 1]

fpr_oof, tpr_oof, thr_oof = roc_curve(y_train, oof_proba)
THRESHOLD = float(thr_oof[np.argmax(tpr_oof - fpr_oof)])
print(f"\nПорог по Юдену (OOF): {THRESHOLD:.3f}")


best_pipeline.fit(X_train, y_train)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

y_pred_optimal = (y_pred_proba >= THRESHOLD).astype(int)

print("\n МЕТРИКИ ПРИ ПОРОГЕ {:.2f} ".format(THRESHOLD))
print(f"Accuracy:      {accuracy_score(y_test, y_pred_optimal):.3f}")
print(f"Precision:     {precision_score(y_test, y_pred_optimal, zero_division=0):.3f}")
print(f"Recall:        {recall_score(y_test, y_pred_optimal):.3f}")
print(f"F1:            {f1_score(y_test, y_pred_optimal):.3f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal,
                            target_names=['Нет осложнений', 'Есть осложнения'],
                            zero_division=0))

cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm.ravel()

print("Матрица ошибок:")
print(f"              Прогноз: Нет    Прогноз: Есть")
print(f"Факт: Нет      {tn:>3}          {fp:>3}")
print(f"Факт: Есть     {fn:>3}          {tp:>3}")

spec = tn / (tn + fp)
lo_r, hi_r = proportion_confint(tp, tp + fn, alpha=0.05, method='beta')
lo_s, hi_s = proportion_confint(tn, tn + fp, alpha=0.05, method='beta')
print(f"Специфичность: {spec:.3f} [{lo_s:.3f}; {hi_s:.3f}]")
print(f"Recall:        {tp/(tp+fn):.3f} [{lo_r:.3f}; {hi_r:.3f}]")

# --- рисунок ---
cm_plot = np.array([[tn, fp], [fn, tp]])
annot_labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                         [f"FN\n{fn}", f"TP\n{tp}"]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Нет осложнений', 'Есть осложнения'],
            yticklabels=['Нет осложнений', 'Есть осложнения'])
plt.title(f'Матрица ошибок')
plt.ylabel('Фактическое значение')
plt.xlabel('Предсказанное значение')
plt.tight_layout()
plt.savefig('confusion_matrix_blue.png', dpi=300, bbox_inches='tight')



thresholds = np.arange(0.05, 0.95, 0.05)
rows = []
for t in thresholds:
    yp = (y_pred_proba >= t).astype(int)
    rows.append({'Порог': t,
                 'Recall': recall_score(y_test, yp, zero_division=0),
                 'Precision': precision_score(y_test, yp, zero_division=0),
                 'F1': f1_score(y_test, yp, zero_division=0)})
thr_df = pd.DataFrame(rows)
print("\nПорог | Recall | Precision | F1")
print(thr_df.round(3).to_string(index=False))


def bootstrap_ci(y_true, y_proba, threshold, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        y_pred = (yp >= threshold).astype(int)
        metrics['accuracy'].append(accuracy_score(yt, y_pred))
        metrics['precision'].append(precision_score(yt, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(yt, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(yt, y_pred, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(yt, yp))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (np.mean(v), np.percentile(v, lo), np.percentile(v, hi))
            for k, v in metrics.items()}

res = bootstrap_ci(y_test, y_pred_proba, threshold=THRESHOLD, B=2000)
print(f"\n{'Метрика':<12}{'Точка':>8}{'95% ДИ':>20}")
for k, (m, lo, hi) in res.items():
    print(f"{k:<12}{m:>8.3f}   [{lo:.3f}; {hi:.3f}]")


feature_names = list(X.columns)
feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
importance_df = (pd.DataFrame({'Признак': feature_names, 'Важность': feature_importance})
                 .sort_values('Важность', ascending=False).reset_index(drop=True))

print("\nРанг | Признак | Важность")
print("-" * 40)
for i, row in importance_df.iterrows():
    print(f"{i+1:>3}   | {row['Признак']:>8} | {row['Важность']:.3f}")

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Признак'], importance_df['Важность'], color='skyblue')
plt.xlabel('Важность')
plt.title('Важность признаков в модели Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_blue.png', dpi=300)


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# точка на ROC при рабочем пороге
idx = np.argmin(np.abs(np.array(_) - THRESHOLD))  # ближайший порог из roc_curve
yp_t = (y_pred_proba >= THRESHOLD).astype(int)
from sklearn.metrics import confusion_matrix as _cm
tn_t, fp_t, fn_t, tp_t = _cm(y_test, yp_t).ravel()
fpr_point = fp_t / (fp_t + tn_t)
tpr_point = tp_t / (tp_t + fn_t)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=5,
         label=f'ROC кривая (AUC = {auc:.3f})')
plt.scatter([fpr_point], [tpr_point], color='red', s=120, zorder=5,
            label=f'Порог {THRESHOLD:.2f}: TPR={tpr_point:.2f}, FPR={fpr_point:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC кривая')
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_blue.png', dpi=300)


perm = permutation_importance(best_pipeline, X_test, y_test,
                              n_repeats=30, random_state=42, scoring='roc_auc')
perm_df = (pd.DataFrame({'Признак': X.columns,
                         'PermImp': perm.importances_mean,
                         'std': perm.importances_std})
           .sort_values('PermImp', ascending=False))
print("\nPermutation importance (ROC-AUC):")
print(perm_df.round(3).to_string(index=False))



import shap
from scipy.stats import spearmanr

rf = best_pipeline.named_steps['classifier']
scaler = best_pipeline.named_steps['scaler']

X_test_scaled = scaler.transform(X_test)

explainer = shap.TreeExplainer(rf)
sv_raw = explainer.shap_values(X_test_scaled)

if isinstance(sv_raw, list):
    sv = sv_raw[1]
else:
    sv = np.asarray(sv_raw)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

print("sv shape:", sv.shape, "X shape:", X_test_scaled.shape)

shap.summary_plot(sv, X_test_scaled, feature_names=list(X.columns), show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')


shap.summary_plot(sv, X_test_scaled, feature_names=list(X.columns),
                  plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')

print(f"\n{'Признак':<6}{'mean|SHAP|':>12}{'rho(SHAP,x)':>14}{'p':>8}{'направление':>16}")
shap_rows = []
for j, nm in enumerate(X.columns):
    rho, p = spearmanr(X_test_scaled[:, j], sv[:, j])
    direction = 'повышает риск' if rho > 0 else 'снижает риск'
    shap_rows.append({'Признак': nm,
                      'mean|SHAP|': np.abs(sv[:, j]).mean(),
                      'rho': rho, 'p': p, 'направление': direction})
    print(f"{nm:<6}{np.abs(sv[:, j]).mean():>12.3f}{rho:>14.3f}{p:>8.3f}{direction:>16}")

shap_df = (pd.DataFrame(shap_rows)
           .sort_values('mean|SHAP|', ascending=False)
           .reset_index(drop=True))
print("\n", shap_df.round(3).to_string(index=False))

mdi = (pd.DataFrame({'Признак': X.columns,
                     'MDI': best_pipeline.named_steps['classifier'].feature_importances_}))
summary = (mdi.merge(perm_df, on='Признак')
              .merge(shap_df, on='Признак')
              .sort_values('mean|SHAP|', ascending=False))
print("\nСводная таблица важности:")
print(summary.round(3).to_string(index=False))