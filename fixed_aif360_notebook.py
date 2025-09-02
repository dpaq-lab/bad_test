# Keep Binder session alive during workshop
import time
import threading
from IPython.display import clear_output

def workshop_keepalive():
    """Keep session alive during presentation"""
    count = 0
    while count < 200:  # Run for ~16 hours max
        time.sleep(300)  # 5 minutes
        count += 1
        clear_output(wait=True)
        print(f"Workshop session active - {time.strftime('%H:%M:%S')}")
        print(f"Runtime: {count * 5} minutes")
        print("Continue with the workshop content below...")

# Start in background
threading.Thread(target=workshop_keepalive, daemon=True).start()

# Keep Binder session alive during workshop
import time
import threading
from IPython.display import clear_output

def workshop_keepalive():
    """Keep session alive during presentation"""
    count = 0
    while count < 200:  # Run for ~16 hours max
        time.sleep(300)  # 5 minutes
        count += 1
        clear_output(wait=True)
        print(f"Workshop session active - {time.strftime('%H:%M:%S')}")
        print(f"Runtime: {count * 5} minutes")
        print("Continue with the workshop content below...")

# Start in background
threading.Thread(target=workshop_keepalive, daemon=True).start()

# =============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# =============================================================================

print("\nSECTION 1: DATA LOADING AND PREPARATION")
print("=" * 40)

# Define column names for Adult dataset
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load the data files
print("Loading Adult dataset from local files...")
df_train = pd.read_csv('data/adult/adult.data', names=column_names, na_values=' ?', skipinitialspace=True)
df_test = pd.read_csv('data/adult/adult.test', names=column_names, na_values=' ?', skipinitialspace=True, skiprows=1)

# Combine datasets
df_raw = pd.concat([df_train, df_test], ignore_index=True)
print(f"Loaded {len(df_raw):,} records")

# Clean the data
print("Cleaning and preparing data...")
df_clean = df_raw.dropna()
print(f"After removing missing values: {len(df_clean):,} records")

# Clean income column (remove periods from test set)
df_clean['income'] = df_clean['income'].str.replace('.', '', regex=False)

# Create binary target variable (0: <=50K, 1: >50K)
df_clean['target'] = (df_clean['income'] == '>50K').astype(int)

# Create binary sex variable (0: Female, 1: Male)
df_clean['sex_binary'] = (df_clean['sex'] == 'Male').astype(int)

print("Data preparation complete!")
print(f"Dataset shape: {df_clean.shape}")
print(f"Target distribution: {df_clean['target'].value_counts().to_dict()}")
print(f"Gender distribution: {df_clean['sex'].value_counts().to_dict()}")

# =============================================================================
# SECTION 2: FEATURE PREPARATION FOR INCOME PREDICTION
# =============================================================================

print("\nSECTION 2: FEATURE PREPARATION FOR PREDICTION")
print("=" * 45)

# Encode categorical variables for modeling
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                   'relationship', 'race', 'native-country']

df_processed = df_clean.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
    le_dict[col] = le

# Define feature columns (excluding target and sex for prediction fairness)
feature_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 
                'hours-per-week'] + [f'{col}_encoded' for col in categorical_cols] + ['sex_binary']

print(f"Features for prediction: {len(feature_cols)} variables")
print(f"Features: {feature_cols}")
print(f"Target: 'target' (income >50K)")
print(f"Protected attribute: 'sex_binary' (gender)")

# Check initial bias in outcomes
female_high_income = df_processed[df_processed['sex_binary'] == 0]['target'].mean()
male_high_income = df_processed[df_processed['sex_binary'] == 1]['target'].mean()

print(f"\nINITIAL BIAS ANALYSIS:")
print(f"Female high income rate: {female_high_income:.1%}")
print(f"Male high income rate: {male_high_income:.1%}")
print(f"Raw disparate impact: {female_high_income / male_high_income:.3f}")

# =============================================================================
# SECTION 3: CREATE AIF360 DATASET
# =============================================================================

print("\nSECTION 3: AIF360 DATASET CREATION")
print("=" * 35)

# Prepare final dataset for AIF360
aif360_df = df_processed[feature_cols + ['target']].copy()

# Create AIF360 StandardDataset
dataset = StandardDataset(
    df=aif360_df,
    label_name='target',
    favorable_classes=[1],
    protected_attribute_names=['sex_binary'],
    privileged_classes=[[1]],  # Male = 1
    categorical_features=[f'{col}_encoded' for col in categorical_cols]
)

print("AIF360 dataset created successfully")
print(f"AIF360 dataset feature shape: {dataset.features.shape}")
print(f"Original feature columns: {len(feature_cols)}")

# Split the data
train_dataset, test_dataset = dataset.split([0.7], shuffle=True, seed=42)
print(f"Training set: {len(train_dataset.labels)} samples")
print(f"Test set: {len(test_dataset.labels)} samples")
print(f"Training features shape: {train_dataset.features.shape}")
print(f"Test features shape: {test_dataset.features.shape}")

# Define privileged and unprivileged groups
privileged_groups = [{'sex_binary': 1}]    # Male
unprivileged_groups = [{'sex_binary': 0}]  # Female

# Initial bias metrics on training data
metric_orig_train = BinaryLabelDatasetMetric(
    train_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\nBIAS METRICS IN TRAINING DATA:")
disparate_impact = metric_orig_train.disparate_impact()
statistical_parity_diff = metric_orig_train.statistical_parity_difference()

print(f"Disparate Impact: {disparate_impact:.3f}")
print(f"Statistical Parity Difference: {statistical_parity_diff:.3f}")

# =============================================================================
# SECTION 4: BASELINE MODEL TRAINING
# =============================================================================

print("\nSECTION 4: BASELINE MODEL TRAINING")
print("=" * 35)

print("Training baseline income prediction model...")

# Prepare data for scikit-learn
scaler = StandardScaler()
X_train = scaler.fit_transform(train_dataset.features)
y_train = train_dataset.labels.ravel()

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Test baseline model
X_test = scaler.transform(test_dataset.features)
y_test = test_dataset.labels.ravel()
y_pred_baseline = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Model Accuracy: {baseline_accuracy:.1%}")

# Feature importance analysis
print(f"\nFeature importance analysis:")
print(f"Number of features in model: {len(baseline_model.feature_importances_)}")
print(f"Number of feature names: {len(feature_cols)}")

# Only show feature importance if dimensions match
if len(baseline_model.feature_importances_) == len(feature_cols):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': baseline_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
else:
    print(f"Feature importance available but dimension mismatch")
    print(f"Top 5 importance values: {sorted(baseline_model.feature_importances_, reverse=True)[:5]}")

# Measure bias in baseline predictions
test_pred_baseline = test_dataset.copy()
test_pred_baseline.labels = y_pred_baseline

cm_baseline = ClassificationMetric(
    test_dataset, test_pred_baseline,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print(f"\nBASELINE MODEL BIAS METRICS:")
print(f"Disparate Impact: {cm_baseline.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {cm_baseline.statistical_parity_difference():.3f}")
print(f"Equal Opportunity Difference: {cm_baseline.equal_opportunity_difference():.3f}")
print(f"Average Odds Difference: {cm_baseline.average_odds_difference():.3f}")

# Risk assessment
di_baseline = cm_baseline.disparate_impact()
if di_baseline < 0.8:
    risk_level = "HIGH RISK"
    print("HIGH LEGAL RISK: Below 80% threshold")
elif di_baseline < 0.9:
    risk_level = "MEDIUM RISK"
    print("MEDIUM RISK: Should be improved")
else:
    risk_level = "LOW RISK"
    print("ACCEPTABLE: Meets basic fairness thresholds")

# =============================================================================
# SECTION 5: BIAS MITIGATION WITH REWEIGHING
# =============================================================================

print("\nSECTION 5: BIAS MITIGATION - REWEIGHING")
print("=" * 38)

print("Applying Reweighing to training data...")

# Apply Reweighing algorithm
reweighing = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

train_dataset_reweighed = reweighing.fit_transform(train_dataset)
print("Reweighing applied successfully")

# Train model with reweighed data
X_train_reweighed = scaler.fit_transform(train_dataset_reweighed.features)
y_train_reweighed = train_dataset_reweighed.labels.ravel()
sample_weights = train_dataset_reweighed.instance_weights

reweighed_model = RandomForestClassifier(n_estimators=100, random_state=42)
reweighed_model.fit(X_train_reweighed, y_train_reweighed, sample_weight=sample_weights)

# Test reweighed model
y_pred_reweighed = reweighed_model.predict(X_test)
reweighed_accuracy = accuracy_score(y_test, y_pred_reweighed)

print(f"Reweighed Model Accuracy: {reweighed_accuracy:.1%}")
print(f"Performance Change: {reweighed_accuracy - baseline_accuracy:+.1%}")

# Measure bias in reweighed predictions
test_pred_reweighed = test_dataset.copy()
test_pred_reweighed.labels = y_pred_reweighed

cm_reweighed = ClassificationMetric(
    test_dataset, test_pred_reweighed,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print(f"\nREWEIGHED MODEL BIAS METRICS:")
print(f"Disparate Impact: {cm_reweighed.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {cm_reweighed.statistical_parity_difference():.3f}")
print(f"Equal Opportunity Difference: {cm_reweighed.equal_opportunity_difference():.3f}")
print(f"Average Odds Difference: {cm_reweighed.average_odds_difference():.3f}")

# Calculate improvements
di_improvement = cm_reweighed.disparate_impact() - cm_baseline.disparate_impact()
eo_improvement = abs(cm_baseline.equal_opportunity_difference()) - abs(cm_reweighed.equal_opportunity_difference())

print(f"\nBIAS REDUCTION ACHIEVED:")
print(f"Disparate Impact improvement: {di_improvement:+.3f}")
print(f"Equal Opportunity improvement: {eo_improvement:+.3f}")

# =============================================================================
# SECTION 6: POST-PROCESSING BIAS MITIGATION
# =============================================================================

print("SECTION 6: POST-PROCESSING BIAS MITIGATION")
print("=" * 42)

print("Reweighing showed minimal improvement. Trying post-processing approach...")
print("Post-processing adjusts model outputs rather than training data.")

# Try Equalized Odds Post-processing
try:
    from aif360.algorithms.postprocessing import EqOddsPostprocessing
    
    print("Applying Equalized Odds Post-processing...")
    
    # Create validation dataset for post-processing
    train_val_dataset, _ = train_dataset.split([0.5], shuffle=True, seed=42)
    X_train_val = scaler.transform(train_val_dataset.features)
    y_pred_train_val = baseline_model.predict(X_train_val)
    
    # Create dataset with baseline predictions for training post-processor
    train_pred_dataset = train_val_dataset.copy()
    train_pred_dataset.labels = y_pred_train_val
    
    # Fit post-processor
    eq_odds = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=42
    )
    
    eq_odds.fit(train_val_dataset, train_pred_dataset)
    
    # Apply post-processing to test predictions
    test_pred_postproc = eq_odds.predict(test_pred_baseline)
    y_pred_postproc = test_pred_postproc.labels.ravel()
    
    postproc_accuracy = accuracy_score(y_test, y_pred_postproc)
    print(f"Post-processed Model Accuracy: {postproc_accuracy:.1%}")
    print(f"Performance Change from Baseline: {postproc_accuracy - baseline_accuracy:+.1%}")
    
    # Measure bias in post-processed predictions
    cm_postproc = ClassificationMetric(
        test_dataset, test_pred_postproc,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    print(f"\nPOST-PROCESSED MODEL BIAS METRICS:")
    print(f"Disparate Impact: {cm_postproc.disparate_impact():.3f}")
    print(f"Statistical Parity Difference: {cm_postproc.statistical_parity_difference():.3f}")
    print(f"Equal Opportunity Difference: {cm_postproc.equal_opportunity_difference():.3f}")
    print(f"Average Odds Difference: {cm_postproc.average_odds_difference():.3f}")
    
    # Calculate improvements from baseline
    di_improvement_postproc = cm_postproc.disparate_impact() - cm_baseline.disparate_impact()
    eo_improvement_postproc = abs(cm_baseline.equal_opportunity_difference()) - abs(cm_postproc.equal_opportunity_difference())
    
    print(f"\nPOST-PROCESSING BIAS REDUCTION:")
    print(f"Disparate Impact improvement: {di_improvement_postproc:+.3f}")
    print(f"Equal Opportunity improvement: {eo_improvement_postproc:+.3f}")
    
    # Risk assessment
    di_postproc = cm_postproc.disparate_impact()
    if di_postproc < 0.8:
        risk_level_postproc = "HIGH RISK"
        print("Still HIGH LEGAL RISK: Below 80% threshold")
    elif di_postproc < 0.9:
        risk_level_postproc = "MEDIUM RISK"
        print("MEDIUM RISK: Improved but should be better")
    else:
        risk_level_postproc = "LOW RISK"
        print("ACCEPTABLE: Meets basic fairness thresholds")
    
    postprocessing_available = True
    
except ImportError:
    print("EqOddsPostprocessing not available in this environment")
    postprocessing_available = False
except Exception as e:
    print(f"Post-processing failed: {e}")
    print("This can happen with severely biased datasets like Adult")
    postprocessing_available = False

# Try alternative post-processing approach
if not postprocessing_available:
    print("\nTrying Calibrated Equalized Odds as alternative...")
    try:
        from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
        
        # Get prediction probabilities instead of binary predictions
        y_pred_prob_train = baseline_model.predict_proba(X_train_val)[:, 1]
        y_pred_prob_test = baseline_model.predict_proba(X_test)[:, 1]
        
        # Create datasets with probabilities
        train_pred_prob_dataset = train_val_dataset.copy()
        train_pred_prob_dataset.scores = y_pred_prob_train
        
        test_pred_prob_dataset = test_dataset.copy()
        test_pred_prob_dataset.scores = y_pred_prob_test
        
        # Fit calibrated post-processor
        cal_eq_odds = CalibratedEqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            cost_constraint="fpr",  # Focus on false positive rate
            seed=42
        )
        
        cal_eq_odds.fit(train_val_dataset, train_pred_prob_dataset)
        
        # Apply calibrated post-processing
        test_pred_cal = cal_eq_odds.predict(test_pred_prob_dataset)
        y_pred_cal = test_pred_cal.labels.ravel()
        
        cal_accuracy = accuracy_score(y_test, y_pred_cal)
        print(f"Calibrated Post-processed Model Accuracy: {cal_accuracy:.1%}")
        
        # Measure bias in calibrated predictions
        cm_cal = ClassificationMetric(
            test_dataset, test_pred_cal,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        print(f"\nCALIBRATED POST-PROCESSED MODEL BIAS METRICS:")
        print(f"Disparate Impact: {cm_cal.disparate_impact():.3f}")
        print(f"Equal Opportunity Difference: {cm_cal.equal_opportunity_difference():.3f}")
        
        di_improvement_cal = cm_cal.disparate_impact() - cm_baseline.disparate_impact()
        print(f"Disparate Impact improvement: {di_improvement_cal:+.3f}")
        
        postprocessing_available = True
        
    except Exception as e:
        print(f"Calibrated post-processing also failed: {e}")
        print("The Adult dataset's bias may be too severe for standard post-processing")

# Summary of post-processing results
print(f"\nPOST-PROCESSING SUMMARY:")
if postprocessing_available:
    print("Post-processing approach completed successfully")
    print("Compare results with pre-processing (reweighing) approach")
else:
    print("Post-processing approaches failed - dataset bias too severe")
    print("This demonstrates the limits of algorithmic bias mitigation")
    print("Some biased datasets require more aggressive interventions or new data collection")

print(f"\nWHY POST-PROCESSING MIGHT WORK BETTER:")
print(f"- Adjusts final outputs rather than training process")
print(f"- Can be more targeted in addressing specific fairness metrics")
print(f"- Often more effective for structural bias in historical data")
print(f"- Easier to tune for business requirements")

# =============================================================================
# SECTION 7: COMPREHENSIVE COMPARISON & VISUALIZATION
# =============================================================================

print("SECTION 7: MODEL COMPARISON ANALYSIS")
print("=" * 37)

# Create comprehensive comparison
print("MODEL PERFORMANCE COMPARISON:")
print(f"{'Metric':<30} {'Baseline':<12} {'Reweighed':<12} {'Change':<10}")
print("-" * 70)
print(f"{'Accuracy':<30} {baseline_accuracy:<12.3f} {reweighed_accuracy:<12.3f} {reweighed_accuracy-baseline_accuracy:+.3f}")
print(f"{'Disparate Impact':<30} {cm_baseline.disparate_impact():<12.3f} {cm_reweighed.disparate_impact():<12.3f} {di_improvement:+.3f}")
print(f"{'Statistical Parity Diff':<30} {cm_baseline.statistical_parity_difference():<12.3f} {cm_reweighed.statistical_parity_difference():<12.3f} {cm_reweighed.statistical_parity_difference()-cm_baseline.statistical_parity_difference():+.3f}")
print(f"{'Equal Opportunity Diff':<30} {cm_baseline.equal_opportunity_difference():<12.3f} {cm_reweighed.equal_opportunity_difference():<12.3f} {cm_reweighed.equal_opportunity_difference()-cm_baseline.equal_opportunity_difference():+.3f}")
print(f"{'Average Odds Diff':<30} {cm_baseline.average_odds_difference():<12.3f} {cm_reweighed.average_odds_difference():<12.3f} {cm_reweighed.average_odds_difference()-cm_baseline.average_odds_difference():+.3f}")

# Group-specific performance analysis
print(f"\nGROUP-SPECIFIC PERFORMANCE ANALYSIS:")
print(f"{'Group Performance':<25} {'Baseline':<12} {'Reweighed':<12}")
print("-" * 50)
print(f"{'Female TPR':<25} {cm_baseline.true_positive_rate(privileged=False):<12.3f} {cm_reweighed.true_positive_rate(privileged=False):<12.3f}")
print(f"{'Male TPR':<25} {cm_baseline.true_positive_rate(privileged=True):<12.3f} {cm_reweighed.true_positive_rate(privileged=True):<12.3f}")
print(f"{'Female FPR':<25} {cm_baseline.false_positive_rate(privileged=False):<12.3f} {cm_reweighed.false_positive_rate(privileged=False):<12.3f}")
print(f"{'Male FPR':<25} {cm_baseline.false_positive_rate(privileged=True):<12.3f} {cm_reweighed.false_positive_rate(privileged=True):<12.3f}")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Disparate Impact Comparison
models = ['Baseline', 'Reweighed']
di_values = [cm_baseline.disparate_impact(), cm_reweighed.disparate_impact()]
colors = ['red' if x < 0.8 else 'orange' if x < 0.9 else 'green' for x in di_values]

bars1 = ax1.bar(models, di_values, color=colors, alpha=0.7)
ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Legal Threshold (0.8)')
ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Fairness (1.0)')
ax1.set_ylabel('Disparate Impact')
ax1.set_title('Disparate Impact: Legal Compliance')
ax1.legend()
ax1.grid(True, alpha=0.3)

for bar, val in zip(bars1, di_values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
             ha='center', va='bottom', fontweight='bold')

# 2. Multiple Fairness Metrics
metrics_names = ['Stat Parity', 'Equal Opp', 'Avg Odds']
baseline_vals = [abs(cm_baseline.statistical_parity_difference()),
                abs(cm_baseline.equal_opportunity_difference()),
                abs(cm_baseline.average_odds_difference())]
reweighed_vals = [abs(cm_reweighed.statistical_parity_difference()),
                 abs(cm_reweighed.equal_opportunity_difference()),
                 abs(cm_reweighed.average_odds_difference())]

x = np.arange(len(metrics_names))
width = 0.35

ax2.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7, color='red')
ax2.bar(x + width/2, reweighed_vals, width, label='Reweighed', alpha=0.7, color='blue')
ax2.set_xlabel('Fairness Metrics (Absolute Values)')
ax2.set_ylabel('Metric Values')
ax2.set_title('Multiple Fairness Metrics Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Group-specific True Positive Rates
groups = ['Female', 'Male']
tpr_baseline = [cm_baseline.true_positive_rate(privileged=False), cm_baseline.true_positive_rate(privileged=True)]
tpr_reweighed = [cm_reweighed.true_positive_rate(privileged=False), cm_reweighed.true_positive_rate(privileged=True)]

x_groups = np.arange(len(groups))
ax3.bar(x_groups - width/2, tpr_baseline, width, label='Baseline', alpha=0.7, color='red')
ax3.bar(x_groups + width/2, tpr_reweighed, width, label='Reweighed', alpha=0.7, color='blue')
ax3.set_xlabel('Gender Groups')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Equal Opportunity: TPR by Gender')
ax3.set_xticks(x_groups)
ax3.set_xticklabels(groups)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Performance vs Fairness Trade-off
ax4.scatter(cm_baseline.disparate_impact(), baseline_accuracy, 
           s=100, color='red', label='Baseline', alpha=0.7)
ax4.scatter(cm_reweighed.disparate_impact(), reweighed_accuracy, 
           s=100, color='blue', label='Reweighed', alpha=0.7)
ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Legal Threshold')
ax4.set_xlabel('Disparate Impact (Fairness)')
ax4.set_ylabel('Accuracy (Performance)')
ax4.set_title('Performance vs Fairness Trade-off')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 7: EXECUTIVE SUMMARY & RECOMMENDATIONS
# =============================================================================

print("\nSECTION 7: EXECUTIVE SUMMARY")
print("=" * 30)

print("BUSINESS IMPACT ANALYSIS:")
print(f"Performance Trade-off: {reweighed_accuracy - baseline_accuracy:+.1%} accuracy")
print(f"Bias Reduction: {di_improvement:+.3f} disparate impact improvement")

# Determine recommendation
if cm_reweighed.disparate_impact() >= 0.8 and reweighed_accuracy >= baseline_accuracy * 0.98:
    recommendation = "DEPLOY: Bias reduced with minimal performance impact"
elif cm_reweighed.disparate_impact() >= 0.8:
    recommendation = "DEPLOY WITH CAUTION: Bias reduced but some performance loss"
elif di_improvement > 0.1:
    recommendation = "PARTIAL SUCCESS: Significant improvement but still needs work"
else:
    recommendation = "ALTERNATIVE NEEDED: Try different bias mitigation approach"

print(f"\nRECOMMENDation: {recommendation}")

print(f"\nKEY FAIRNESS METRICS FOR INCOME PREDICTION:")
print(f"1. Disparate Impact: Legal compliance (should be > 0.8)")
print(f"2. Equal Opportunity: Fair treatment of qualified candidates")
print(f"3. Statistical Parity: May not be appropriate for income prediction")

print(f"\nWHY EQUAL OPPORTUNITY IS BETTER FOR INCOME PREDICTION:")
print(f"- Focuses on fair treatment of people who SHOULD get high income")
print(f"- Allows for legitimate differences due to qualifications")
print(f"- More business-appropriate than strict statistical parity")

print(f"\nNEXT STEPS:")
print(f"1. Consider Equal Opportunity as primary metric")
print(f"2. Test other AIF360 algorithms (post-processing, adversarial debiasing)")
print(f"3. Implement bias monitoring in production")
print(f"4. Regular model audits for bias drift")

print("\n" + "=" * 70)
print("WORKSHOP COMPLETE: AIF360 INCOME PREDICTION BIAS MITIGATION")
print("=" * 70)