# %% [markdown]
# # Diabetes Prediction using Machine Learning and Deep Learning
# ## CS634 Data Mining - Final Project
# 
# **Date:** November 2025  
# 
# This notebook implements three binary classification algorithms to predict diabetes:
# 1. **Random Forest** (Mandatory)
# 2. **LSTM** (Deep Learning)
# 3. **SVM** (Traditional ML)
# 
# We use **10-fold cross-validation** and calculate all metrics manually.
# 
# ---

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, brier_score_loss

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("All libraries imported successfully!")

# %% [markdown]
# ## 2. Load and Explore Data

# %%
data = pd.read_csv('diabetes.csv')

print("First 5 rows of data:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# %% [markdown]
# ## 3. Data Preprocessing

# %%
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_to_fix:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

print("Missing values handled successfully!")
print("\nChecking for any remaining missing values:")
print(data.isnull().sum())

# %% [markdown]
# ## 4. Beautiful Visualizations for Report
# 
# 

# %% [markdown]
# ### 📊 Graph 1: Class Distribution (Pie Chart)

# %%
plt.figure(figsize=(10, 7))

counts = data['Outcome'].value_counts()
colors = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)

plt.pie(counts, labels=['No Diabetes', 'Has Diabetes'], 
        autopct='%1.1f%%', startangle=90, colors=colors,
        explode=explode, shadow=True, 
        textprops={'fontsize': 14, 'weight': 'bold'})

plt.title('Diabetes Distribution in Dataset', 
          fontsize=18, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('graph1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

no_diabetes = data[data['Outcome']==0].shape[0]
has_diabetes = data[data['Outcome']==1].shape[0]
total = data.shape[0]

print(f"\n📊 Class Distribution:")
print(f"No Diabetes: {no_diabetes} ({no_diabetes/total*100:.1f}%)")
print(f"Has Diabetes: {has_diabetes} ({has_diabetes/total*100:.1f}%)")
print("\n✅ Graph 1 saved as 'graph1_class_distribution.png'")

# %% [markdown]
# ### 📊 Graph 2: Age Distribution by Diabetes Status

# %%
# Age distribution histogram
plt.figure(figsize=(12, 7))

no_diabetes = data[data['Outcome'] == 0]['Age']
has_diabetes = data[data['Outcome'] == 1]['Age']

plt.hist(no_diabetes, bins=20, alpha=0.6, label='No Diabetes', 
         color='#3498db', edgecolor='black')
plt.hist(has_diabetes, bins=20, alpha=0.6, label='Has Diabetes', 
         color='#e74c3c', edgecolor='black')

plt.xlabel('Age (years)', fontsize=14, weight='bold')
plt.ylabel('Number of Patients', fontsize=14, weight='bold')
plt.title('Age Distribution by Diabetes Status', 
          fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graph2_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 2 saved as 'graph2_age_distribution.png'")

# %% [markdown]
# ### 📊 Graph 3: Key Features Comparison (Box Plots)

# %%
# Multiple box plots for feature comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Key Medical Features by Diabetes Status', 
             fontsize=20, weight='bold', y=1.02)

features_to_plot = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies']
colors_box = ['#3498db', '#e74c3c']

for idx, feature in enumerate(features_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    no_diab = data[data['Outcome'] == 0][feature]
    has_diab = data[data['Outcome'] == 1][feature]
    
    box_data = [no_diab, has_diab]
    bp = ax.boxplot(box_data, labels=['No Diabetes', 'Has Diabetes'],
                     patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(feature, fontsize=12, weight='bold')
    ax.set_title(f'{feature} Levels', fontsize=13, weight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graph3_feature_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 3 saved as 'graph3_feature_comparison.png'")

# %% [markdown]
# ### 📊 Graph 4: Glucose vs BMI Scatter Plot

# %%
plt.figure(figsize=(12, 8))

no_diab = data[data['Outcome'] == 0]
has_diab = data[data['Outcome'] == 1]

plt.scatter(no_diab['Glucose'], no_diab['BMI'], 
           alpha=0.6, s=100, c='#3498db', label='No Diabetes', 
           edgecolors='black', linewidth=0.5)
plt.scatter(has_diab['Glucose'], has_diab['BMI'], 
           alpha=0.6, s=100, c='#e74c3c', label='Has Diabetes', 
           edgecolors='black', linewidth=0.5)

plt.xlabel('Glucose Level (mg/dL)', fontsize=14, weight='bold')
plt.ylabel('BMI (Body Mass Index)', fontsize=14, weight='bold')
plt.title('Glucose vs BMI - Diabetes Correlation', 
          fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graph4_glucose_bmi_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 4 saved as 'graph4_glucose_bmi_scatter.png'")

# %% [markdown]
# ### 📊 Graph 5: Correlation Heatmap (Clean Version)

# %%
# Beautiful correlation heatmap
plt.figure(figsize=(12, 10))
correlation = data.corr()

mask = np.triu(np.ones_like(correlation, dtype=bool))

sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True,
            linewidths=1, linecolor='white',
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'})

plt.title('Feature Correlation Heatmap', fontsize=18, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('graph5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 5 saved as 'graph5_correlation_heatmap.png'")

# %% [markdown]
# ## 5. Prepare Data for Modeling

# %%
x = data.drop('Outcome', axis=1)
y = data['Outcome']

print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(x.columns)}")

# %% [markdown]
# ## 6. Define Metrics Calculation Functions
# 
# We calculate all metrics **manually** (except confusion matrix generation)

# %%
def calculate_all_metrics(actual, predicted, predicted_proba=None):

    cm = confusion_matrix(actual, predicted, labels=[1, 0])
    tp = cm[0][0]  # True Positive
    fn = cm[0][1]  # False Negative
    fp = cm[1][0]  # False Positive
    tn = cm[1][1]  # True Negative
    
    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity/Recall)
    tnr = tn / (tn + fp)  # True Negative Rate (Specificity)
    fpr = fp / (tn + fp)  # False Positive Rate
    fnr = fn / (tp + fn)  # False Negative Rate
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    
    balanced_acc = (tpr + tnr) / 2
    
    # TSS (True Skill Statistic)
    tss = tpr - fpr
    
    # HSS (Heidke Skill Score)
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = numerator / denominator if denominator > 0 else 0
    
    # Calculate P and N
    p = tp + fn  # Total actual positives
    n = tn + fp  # Total actual negatives
    
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'P': p,
        'N': n,
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'Error_Rate': error_rate,
        'Balanced_Acc': balanced_acc,
        'TSS': tss,
        'HSS': hss
    }
    
    if predicted_proba is not None:
        fpr_curve, tpr_curve, _ = roc_curve(actual, predicted_proba)
        roc_auc = auc(fpr_curve, tpr_curve)
        bs = brier_score_loss(actual, predicted_proba)
        
        climatology = np.mean(actual)
        bs_ref = np.mean((actual - climatology) ** 2)
        bss = 1 - (bs / bs_ref) if bs_ref > 0 else 0
        
        metrics['AUC'] = roc_auc
        metrics['Brier_Score'] = bs
        metrics['BSS'] = bss
    
    return metrics

print("Metrics calculation function defined successfully!")

# %% [markdown]
# ## 7. Set Up 10-Fold Cross-Validation

# %%
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

rf_results = []
lstm_results = []
svm_results = []

print(f"10-Fold Cross-Validation initialized with {num_folds} folds")

# %% [markdown]
# ## 8. Train and Evaluate Models with 10-Fold CV
# 
# We'll train all three models on each fold and calculate metrics.
# 
# **Note:** This cell will take 3-5 minutes to run. Be patient! ⏰

# %%
fold_num = 0

print("Starting 10-Fold Cross-Validation...\n")


for train_idx, test_idx in kfold.split(x):
    fold_num += 1
    print(f"{'='*60}")
    print(f"Processing Fold {fold_num}/{num_folds}")
    print(f"{'='*60}")
    
    # Split data for this fold
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    
    # ============================================
    # 1. RANDOM FOREST
    # ============================================
    print("\n[1/3] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train_scaled, y_train)
    
    # Predictions
    rf_pred = rf_model.predict(x_test_scaled)
    rf_pred_proba = rf_model.predict_proba(x_test_scaled)[:, 1]
    
    # Calculate metrics
    rf_metrics = calculate_all_metrics(y_test, rf_pred, rf_pred_proba)
    rf_metrics['Fold'] = fold_num
    rf_results.append(rf_metrics)
    print(f"Random Forest Accuracy: {rf_metrics['Accuracy']:.4f}")
    
    # ============================================
    # 2. SVM
    # ============================================
    print("\n[2/3] Training SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(x_train_scaled, y_train)
    
    # Predictions
    svm_pred = svm_model.predict(x_test_scaled)
    svm_pred_proba = svm_model.predict_proba(x_test_scaled)[:, 1]
    
    # Calculate metrics
    svm_metrics = calculate_all_metrics(y_test, svm_pred, svm_pred_proba)
    svm_metrics['Fold'] = fold_num
    svm_results.append(svm_metrics)
    print(f"SVM Accuracy: {svm_metrics['Accuracy']:.4f}")
    
    # ============================================
    # 3. LSTM
    # ============================================
    print("\n[3/3] Training LSTM...")
    
    x_train_lstm = x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1))
    x_test_lstm = x_test_scaled.reshape((x_test_scaled.shape[0], x_test_scaled.shape[1], 1))
    
    # Build LSTM model
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(x_train_lstm.shape[1], 1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    lstm_model.fit(x_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Predictions
    lstm_pred_proba = lstm_model.predict(x_test_lstm, verbose=0).flatten()
    lstm_pred = (lstm_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    lstm_metrics = calculate_all_metrics(y_test, lstm_pred, lstm_pred_proba)
    lstm_metrics['Fold'] = fold_num
    lstm_results.append(lstm_metrics)
    print(f"LSTM Accuracy: {lstm_metrics['Accuracy']:.4f}")
    print()

print(f"\n{'='*60}")
print("All folds completed successfully!")
print(f"{'='*60}")

# %% [markdown]
# ## 9. Display Results for Each Fold
# 
# ### 9.1 Random Forest Results

# %%
rf_df = pd.DataFrame(rf_results)

print("\n" + "="*80)
print("RANDOM FOREST - Per-Fold Results")
print("="*80)
print(rf_df.round(4).to_string(index=False))

rf_avg = rf_df.drop('Fold', axis=1).mean()
print("\n" + "="*80)
print("RANDOM FOREST - Average Results Across All Folds")
print("="*80)
for metric, value in rf_avg.items():
    print(f"{metric:20s}: {value:.4f}")

# %% [markdown]
# ### 9.2 SVM Results

# %%
svm_df = pd.DataFrame(svm_results)

# Display results
print("\n" + "="*80)
print("SVM - Per-Fold Results")
print("="*80)
print(svm_df.round(4).to_string(index=False))

# Calculate and display averages
svm_avg = svm_df.drop('Fold', axis=1).mean()
print("\n" + "="*80)
print("SVM - Average Results Across All Folds")
print("="*80)
for metric, value in svm_avg.items():
    print(f"{metric:20s}: {value:.4f}")

# %% [markdown]
# ### 9.3 LSTM Results

# %%
lstm_df = pd.DataFrame(lstm_results)

# Display results
print("\n" + "="*80)
print("LSTM - Per-Fold Results")
print("="*80)
print(lstm_df.round(4).to_string(index=False))

lstm_avg = lstm_df.drop('Fold', axis=1).mean()
print("\n" + "="*80)
print("LSTM - Average Results Across All Folds")
print("="*80)
for metric, value in lstm_avg.items():
    print(f"{metric:20s}: {value:.4f}")

# %% [markdown]
# ## 10. Compare All Models - Summary Table

# %%
comparison = pd.DataFrame({
    'Random_Forest': rf_avg,
    'SVM': svm_avg,
    'LSTM': lstm_avg
})

print("\n" + "="*90)
print("MODEL COMPARISON - Average Performance Across 10 Folds")
print("="*90)
print(comparison.round(4))

print("\n" + "="*90)
print("Best Model for Each Metric:")
print("="*90)
for metric in comparison.index:
    best_model = comparison.loc[metric].idxmax()
    best_value = comparison.loc[metric].max()
    print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")

# %% [markdown]
# ## 11. Additional Model Performance Visualizations

# %% [markdown]
# ### 📊 Graph 6: Feature Importance (Random Forest)

# %%
# Train final RF model for feature importance
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_importance.fit(x, y)

importance = rf_importance.feature_importances_
features = x.columns

indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 7))
colors_bars = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(features))]
bars = plt.barh(range(len(features)), importance[indices], 
                color=colors_bars, edgecolor='black')

plt.yticks(range(len(features)), [features[i] for i in indices], fontsize=12)
plt.xlabel('Importance Score', fontsize=14, weight='bold')
plt.title('Feature Importance for Diabetes Prediction', 
          fontsize=18, weight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', 
             fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('graph6_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 6 saved as 'graph6_feature_importance.png'")

# %% [markdown]
# ### 📊 Graph 7: Model Accuracy Comparison Bar Chart

# %%
rf_accuracy = rf_avg['Accuracy']
svm_accuracy = svm_avg['Accuracy']
lstm_accuracy = lstm_avg['Accuracy']

plt.figure(figsize=(10, 7))
models = ['Random Forest', 'SVM', 'LSTM']
scores = [rf_accuracy, svm_accuracy, lstm_accuracy]
colors_models = ['#2ecc71', '#3498db', '#e74c3c']

bars = plt.bar(models, scores, color=colors_models, 
               edgecolor='black', linewidth=2, width=0.6)

plt.ylabel('Accuracy Score', fontsize=14, weight='bold')
plt.title('Model Performance Comparison', fontsize=18, weight='bold', pad=20)
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', 
             fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('graph7_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 7 saved as 'graph7_model_comparison.png'")

# %% [markdown]
# ### 📊 Graph 8: ROC Curves for All Models

# %%
x_train_viz, x_test_viz, y_train_viz, y_test_viz = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler_final = StandardScaler()
x_train_scaled_viz = scaler_final.fit_transform(x_train_viz)
x_test_scaled_viz = scaler_final.transform(x_test_viz)

# Random Forest
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(x_train_scaled_viz, y_train_viz)
rf_proba_viz = rf_final.predict_proba(x_test_scaled_viz)[:, 1]

# SVM
svm_final = SVC(kernel='rbf', probability=True, random_state=42)
svm_final.fit(x_train_scaled_viz, y_train_viz)
svm_proba_viz = svm_final.predict_proba(x_test_scaled_viz)[:, 1]

# LSTM
x_train_lstm_viz = x_train_scaled_viz.reshape((x_train_scaled_viz.shape[0], x_train_scaled_viz.shape[1], 1))
x_test_lstm_viz = x_test_scaled_viz.reshape((x_test_scaled_viz.shape[0], x_test_scaled_viz.shape[1], 1))

lstm_final = Sequential([
    LSTM(64, activation='relu', input_shape=(x_train_lstm_viz.shape[1], 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_final.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_final.fit(x_train_lstm_viz, y_train_viz, epochs=50, batch_size=32, verbose=0)
lstm_proba_viz = lstm_final.predict(x_test_lstm_viz, verbose=0).flatten()

fpr_rf, tpr_rf, _ = roc_curve(y_test_viz, rf_proba_viz)
fpr_svm, tpr_svm, _ = roc_curve(y_test_viz, svm_proba_viz)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test_viz, lstm_proba_viz)

auc_rf = auc(fpr_rf, tpr_rf)
auc_svm = auc(fpr_svm, tpr_svm)
auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', 
         linewidth=3, color='#2ecc71')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})', 
         linewidth=3, color='#3498db')
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC = {auc_lstm:.3f})', 
         linewidth=3, color='#e74c3c')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('ROC Curves - Model Comparison', fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graph8_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 ROC-AUC Scores:")
print(f"Random Forest: {auc_rf:.4f}")
print(f"SVM: {auc_svm:.4f}")
print(f"LSTM: {auc_lstm:.4f}")
print("\nGraph 8 saved as 'graph8_roc_curves.png'")

# %% [markdown]
# ### 📊 Graph 9: Confusion Matrix Heatmap (Random Forest)

# %%
rf_pred_viz = rf_final.predict(x_test_scaled_viz)
cm = confusion_matrix(y_test_viz, rf_pred_viz)

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            square=True, linewidths=2, linecolor='black',
            annot_kws={'size': 18, 'weight': 'bold'})

plt.xlabel('Predicted Label', fontsize=14, weight='bold')
plt.ylabel('True Label', fontsize=14, weight='bold')
plt.title('Confusion Matrix - Random Forest', fontsize=18, weight='bold', pad=20)
plt.xticks([0.5, 1.5], ['No Diabetes', 'Has Diabetes'], fontsize=12)
plt.yticks([0.5, 1.5], ['No Diabetes', 'Has Diabetes'], fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig('graph9_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 9 saved as 'graph9_confusion_matrix.png'")

# %% [markdown]
# ### 📊 Graph 10: Accuracy Across Folds (Line Chart)

# %%
plt.figure(figsize=(12, 7))
folds = list(range(1, num_folds + 1))

plt.plot(folds, rf_df['Accuracy'], marker='o', label='Random Forest', 
         linewidth=3, markersize=8, color='#2ecc71')
plt.plot(folds, svm_df['Accuracy'], marker='s', label='SVM', 
         linewidth=3, markersize=8, color='#3498db')
plt.plot(folds, lstm_df['Accuracy'], marker='^', label='LSTM', 
         linewidth=3, markersize=8, color='#e74c3c')

plt.xlabel('Fold Number', fontsize=14, weight='bold')
plt.ylabel('Accuracy', fontsize=14, weight='bold')
plt.title('Model Accuracy Across 10 Folds', fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xticks(folds)
plt.tight_layout()
plt.savefig('graph10_accuracy_folds.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 10 saved as 'graph10_accuracy_folds.png'")

# %% [markdown]
# ### 📊 Graph 11: Key Metrics Comparison (5 Subplots)

# %%
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    models = ['Random Forest', 'SVM', 'LSTM']
    values = [rf_avg[metric], svm_avg[metric], lstm_avg[metric]]
    
    bars = ax.bar(models, values, color=['#2ecc71', '#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric, fontsize=12, weight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=11, weight='bold')

fig.delaxes(axes[5])

plt.suptitle('Key Performance Metrics - Model Comparison', 
             fontsize=16, weight='bold', y=1.00)
plt.tight_layout()
plt.savefig('graph11_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph 11 saved as 'graph11_metrics_comparison.png'")

# %% [markdown]
# ### 📊 Graph 12: Individual ROC Curve - Random Forest

# %%
# Individual ROC curve for Random Forest
plt.figure(figsize=(10, 8))

plt.plot(fpr_rf, tpr_rf, color='#2ecc71', linewidth=4, 
         label=f'Random Forest (AUC = {auc_rf:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

plt.fill_between(fpr_rf, tpr_rf, alpha=0.2, color='#2ecc71')

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('ROC Curve - Random Forest', fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('graph12_roc_random_forest.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graph 12 saved as 'graph12_roc_random_forest.png'")
print(f"   Random Forest AUC: {auc_rf:.4f}")

# %% [markdown]
# ### 📊 Graph 13: Individual ROC Curve - SVM

# %%
# Individual ROC curve for SVM
plt.figure(figsize=(10, 8))

plt.plot(fpr_svm, tpr_svm, color='#3498db', linewidth=4, 
         label=f'SVM (AUC = {auc_svm:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

plt.fill_between(fpr_svm, tpr_svm, alpha=0.2, color='#3498db')

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('ROC Curve - Support Vector Machine', fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('graph13_roc_svm.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graph 13 saved as 'graph13_roc_svm.png'")
print(f"   SVM AUC: {auc_svm:.4f}")

# %% [markdown]
# ### 📊 Graph 14: Individual ROC Curve - LSTM

# %%
# Individual ROC curve for LSTM
plt.figure(figsize=(10, 8))

plt.plot(fpr_lstm, tpr_lstm, color='#e74c3c', linewidth=4, 
         label=f'LSTM (AUC = {auc_lstm:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

plt.fill_between(fpr_lstm, tpr_lstm, alpha=0.2, color='#e74c3c')

plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('ROC Curve - LSTM Neural Network', fontsize=18, weight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('graph14_roc_lstm.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graph 14 saved as 'graph14_roc_lstm.png'")
print(f"   LSTM AUC: {auc_lstm:.4f}")

# %%
# Save individual fold results
rf_df.to_csv('random_forest_results.csv', index=False)
svm_df.to_csv('svm_results.csv', index=False)
lstm_df.to_csv('lstm_results.csv', index=False)

comparison.to_csv('model_comparison_summary.csv')

print("Results exported successfully!")
print("\nFiles created:")
print("random_forest_results.csv")
print("svm_results.csv")
print("lstm_results.csv")
print("model_comparison_summary.csv")


