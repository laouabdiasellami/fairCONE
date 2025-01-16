import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class FairCONE:
    def __init__(self, max_step=10, tmin=0, tmax=1, beta=1.0, dp_threshold=0.1):
        self.max_step = max_step
        self.tmin = tmin
        self.tmax = tmax
        self.beta = beta
        self.dp_threshold = dp_threshold
        self.base_classifier = RandomForestClassifier(random_state=42)
        
    def demographic_parity_diff(self, y_pred, sensitive_features):
        """Calculate demographic parity difference"""
        group_0_pred = y_pred[sensitive_features == 0]
        group_1_pred = y_pred[sensitive_features == 1]
        
        rate_0 = np.mean(group_0_pred)
        rate_1 = np.mean(group_1_pred)
        
        return abs(rate_0 - rate_1)
    
    def get_slope(self, accuracy, dp_diff, t_val):
        """Calculate CONE slopes for accuracy-fairness trade-off"""
        # Normalize metrics to [0,1] range
        norm_acc = accuracy
        norm_dp = 1 - dp_diff  # Convert DP diff to fairness score
        
        # Calculate current performance measure (weighted combination)
        curr_fm = self.beta * norm_acc + (1 - self.beta) * norm_dp
        
        # Calculate slopes for left and right bounds
        slope_left = (curr_fm - norm_dp) / (t_val + 1e-10)
        slope_right = (norm_acc - curr_fm) / (1 - t_val + 1e-10)
        
        return curr_fm, slope_left, slope_right
    
    def add_cone(self, state, accuracy, dp_diff, t_val):
        """Add CONE constraint to state matrix based on accuracy and DP"""
        x_cone = np.linspace(self.tmin, self.tmax, state.shape[1])
        y_cone = np.linspace(1, 0, state.shape[0])
        
        # Get slopes and current performance
        curr_fm, slope_left, slope_right = self.get_slope(accuracy, dp_diff, t_val)
        
        # Create constraint lines
        grid_x, grid_y = np.meshgrid(x_cone, y_cone)
        line_left = grid_y - slope_left * grid_x - (curr_fm - slope_left * t_val)
        line_right = grid_y - slope_right * grid_x - (curr_fm - slope_right * t_val)
        
        # Update state matrix
        state[np.logical_and(line_left >= 0, line_right >= 0)] = -state.shape[0]
        
        return curr_fm
    
    def get_best_fm_available(self, state, t_vals):
        """Find next best operating point in state space"""
        # Find available performance levels
        available_fm = np.where(state.sum(axis=1) > -state.shape[1]**2)[0]
        
        if len(available_fm) == 0:
            return -1, -1
        
        # Get best available performance level
        best_fm_idx = available_fm[0]
        
        # Find valid t-values at this performance level
        valid_t = np.where(state[best_fm_idx] == 1)[0]
        
        if len(valid_t) == 0:
            t_val = self.tmin
        else:
            # Choose middle point between previous t-values
            t_pos = valid_t[len(valid_t)//2]
            t_val = self.tmin + (self.tmax - self.tmin) * (t_pos / (state.shape[1] - 1))
            
            # Adjust based on previous t-values if available
            if t_vals is not None and len(t_vals) > 0:
                prev_lower = t_vals[t_vals < t_val]
                prev_higher = t_vals[t_vals > t_val]
                
                if len(prev_lower) > 0 and len(prev_higher) > 0:
                    t_val = (np.max(prev_lower) + np.min(prev_higher)) / 2
                elif len(prev_lower) > 0:
                    t_val = (np.max(prev_lower) + self.tmax) / 2
                elif len(prev_higher) > 0:
                    t_val = (self.tmin + np.min(prev_higher)) / 2
        
        max_fm = 1 - (best_fm_idx / (state.shape[0] - 1))
        
        return t_val, max_fm
    
    def compute_weights(self, t_value, sensitive_features):
        """Compute sample weights based on t-value and sensitive features"""
        weights = np.ones(len(sensitive_features))
        weights[sensitive_features == 0] *= t_value
        weights[sensitive_features == 1] *= (1 - t_value)
        return weights
    
    def evaluate_classifier(self, clf, X, y, sensitive_features):
        """Evaluate classifier performance metrics"""
        predictions = clf.predict(X)
        accuracy = accuracy_score(y, predictions)
        dp_diff = self.demographic_parity_diff(predictions, sensitive_features)
        
        return predictions, accuracy, dp_diff
    
    def fit_predict(self, X, y, sensitive_features):
        """Main training loop with fairness constraints"""
        state_size = 100
        state = np.ones((state_size, state_size))
        
        outputs = {
            "accuracies": np.zeros(self.max_step),
            "dp_differences": np.zeros(self.max_step),
            "t_values": np.zeros(self.max_step),
            "predictions": []
        }
        
        best_acc = 0
        best_dp = float('inf')
        best_clf = None
        next_t_val = (self.tmin + self.tmax) / 2
        
        step = 0
        while step < self.max_step:
            # Store t-value
            outputs["t_values"][step] = next_t_val
            
            # Compute sample weights
            weights = self.compute_weights(next_t_val, sensitive_features)
            
            # Train classifier with weights
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X, y, sample_weight=weights)
            
            # Evaluate performance
            preds, acc, dp_diff = self.evaluate_classifier(clf, X, y, sensitive_features)
            
            # Store metrics
            outputs["accuracies"][step] = acc
            outputs["dp_differences"][step] = dp_diff
            outputs["predictions"].append(preds)
            
            # Update best model if improved
            if acc >= best_acc and dp_diff <= best_dp:
                best_acc = acc
                best_dp = dp_diff
                best_clf = clf
            
            # Add constraint to state
            curr_fm = self.add_cone(state, acc, dp_diff, next_t_val)
            
            # Get next operating point
            next_t_val, max_fm = self.get_best_fm_available(state, outputs["t_values"][:step+1])
            
            if max_fm <= curr_fm or next_t_val < 0:
                break
                
            step += 1
        
        return best_clf, outputs

def load_dataset(dataset_path, dataset_type):
    """Load and preprocess different dataset types"""
    try:
        if dataset_type == "adult":
            df = pd.read_csv(dataset_path)
            
            # Encode categorical variables
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in ['income', 'sex']:  # Don't encode target and sensitive
                    df[col] = pd.Categorical(df[col]).codes
            
            X = df.drop(['income', 'sex'], axis=1)
            y = (df['income'] == '>50K').astype(int)
            sensitive_attr = (df['sex'] == 'Male').astype(int)
            return X.values, y.values, sensitive_attr.values
            
        
            
        elif dataset_type == "hourly_wages":
            df = pd.read_csv(dataset_path)
            # Encode categorical variables if any
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in ['wage_per_hour', 'female']:
                    df[col] = pd.Categorical(df[col]).codes
                    
            X = df.drop(['wage_per_hour', 'female'], axis=1)
            median_wage = df['wage_per_hour'].median()
            y = (df['wage_per_hour'] > median_wage).astype(int)
            sensitive_attr = df['female'].values
            return X.values, y.values, sensitive_attr
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def run_experiment(X, y, sensitive_attr):
    # Split data
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_attr, test_size=0.2, random_state=42
    )

    # Train original classifier (without fairness)
    original_clf = RandomForestClassifier(random_state=42)
    original_clf.fit(X_train, y_train)
    y_pred_original = original_clf.predict(X_test)
    
    # Calculate original metrics
    acc_original = accuracy_score(y_test, y_pred_original)
    fair_cone = FairCONE()
    dp_original = fair_cone.demographic_parity_diff(y_pred_original, sensitive_test)

    # Train fair CONE classifier
    fair_clf, outputs = fair_cone.fit_predict(X_train, y_train, sensitive_train)
    y_pred_fair = fair_clf.predict(X_test)
    
    # Calculate fair metrics
    acc_fair = accuracy_score(y_test, y_pred_fair)
    dp_fair = fair_cone.demographic_parity_diff(y_pred_fair, sensitive_test)

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    bars1 = ax1.bar(['Original', 'Fair CONE'], 
                    [acc_original, acc_fair],
                    color=['skyblue', 'lightgreen'])
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{height:.3f}',
                ha='center', va='center',
                color='black', fontweight='bold')

    # Demographic parity plot
    bars2 = ax2.bar(['Original', 'Fair CONE'],
                    [dp_original, dp_fair],
                    color=['skyblue', 'lightgreen'])
    ax2.set_title('Demographic Parity Difference')
    ax2.set_ylabel('|DP Difference|')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{height:.3f}',
                ha='center', va='center',
                color='black', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Original Classifier - Accuracy: {acc_original:.3f}, DP Difference: {dp_original:.3f}")
    print(f"Fair CONE Classifier - Accuracy: {acc_fair:.3f}, DP Difference: {dp_fair:.3f}")
    
    return {
        'original': {'accuracy': acc_original, 'dp_diff': dp_original},
        'fair': {'accuracy': acc_fair, 'dp_diff': dp_fair},
        'training_history': outputs
    }

if __name__ == "__main__":
    # Configure dataset
    dataset_type = "hourly_wages"  # Change this to "abalone" or "hourly_wages" as needed
    dataset_paths = {
        "adult": "datasets/adult.csv",
        
        "hourly_wages": "datasets/hourly_wages.csv"
    }
    
    # Load and process data
    X, y, sensitive_attr = load_dataset(dataset_paths[dataset_type], dataset_type)
    
    # Run experiment
    results = run_experiment(X, y, sensitive_attr)
