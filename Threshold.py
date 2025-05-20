import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
import pandas as pd

df = pd.read_csv('predictions_with_probs.csv')

# 提取真实标签与预测概率
y_true = df['true_click']
probs  = df['predicted_prob']

# Precision–Recall 曲线 + F1 最大化
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
# 只看有效阈值对应的 F1（跳过第一个无阈值点）
best_pr_idx = np.nanargmax(f1_scores[1:])
best_pr_threshold = thresholds_pr[best_pr_idx]
print(f"Best F1={f1_scores[best_pr_idx+1]:.4f} at threshold={best_pr_threshold:.4f}")

# ROC曲线+Youden’s J最大化
fpr, tpr, thresholds_roc = roc_curve(y_true, probs)
j_scores = tpr - fpr
best_roc_idx = np.argmax(j_scores)
best_roc_threshold = thresholds_roc[best_roc_idx]
print(f"Best J={j_scores[best_roc_idx]:.4f} at threshold={best_roc_threshold:.4f}")

# 用这两个阈值去二元化
preds_pr  = (probs >= best_pr_threshold).astype(int)
preds_roc = (probs >= best_roc_threshold).astype(int)

# 评估对比
print("Using F1-opt threshold:")
print("  F1 score:", f1_score(y_true, preds_pr))
print("Using Youden’s J threshold:")
print("  Youden’s J TPR–FPR:", j_scores[best_roc_idx])
