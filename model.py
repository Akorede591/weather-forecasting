# model.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Using Regressor for continuous target
from sklearn.feature_selection import SelectKBest, mutual_info_regression 
from sklearn.metrics import r2_score, mean_absolute_error

def prepare_dataset(filepath, target_column):
    """
    Loads, cleans, encodes, and splits the weather dataset.
    """
    df = pd.read_csv(filepath)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Clean missing numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    # Clean missing categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # One-Hot Encoding for categories
    X = pd.get_dummies(X, drop_first=True)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def run_baseline_pipeline(X_train, X_test, y_train, y_test):
    """
    Trains a baseline Random Forest Regressor using ALL available features.
    """
    start_time = time.time()
    
    # Initialize Random Forest Regressor with 100 trees
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model live
    rf_model.fit(X_train, y_train)
    
    # Predict on the test data
    predictions = rf_model.predict(X_test)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    elapsed_time = time.time() - start_time
    
    return r2, mae, elapsed_time


def run_mi_pipeline(X_train, X_test, y_train, y_test, feature_names, k):
    """
    Selects the top K features using Mutual Information and trains a Random Forest Regressor.
    """
    start_time = time.time()
    
    # 1. Initialize and run Mutual Information feature selection
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get the names of the selected features
    mask = selector.get_support()
    selected_features = [feature_names[i] for i, val in enumerate(mask) if val]
    
    # 2. Train Random Forest on ONLY the selected K features
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_selected, y_train)
    
    # 3. Evaluate predictions
    predictions = rf_model.predict(X_test_selected)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    elapsed_time = time.time() - start_time
    return r2, mae, elapsed_time, selected_features


def pso_fitness_function(particle, X_train, X_test, y_train, y_test):
    """
    Evaluates a feature subset (particle position) using a fast Random Forest.
    """
    # Convert continuous position values to a binary mask (threshold at 0.5)
    mask = particle > 0.5
    
    # If a particle selects absolutely zero features, give it a terrible fitness score
    if not any(mask):
        return -9999.0 
        
    # Filter training and testing sets to only include the active features
    X_train_sub = X_train[:, mask]
    X_test_sub = X_test[:, mask]
    
    # Use 10 trees (instead of 100) to keep the swarm's live iterations lightning fast
    rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_sub, y_train)
    
    # Evaluate using R-squared as the fitness metric
    preds = rf.predict(X_test_sub)
    return r2_score(y_test, preds)


def run_pso_pipeline(X_train, X_test, y_train, y_test, feature_names, num_particles=10, max_iter=5):
    """
    Optimizes feature selection using Particle Swarm Optimization (PSO).
    """
    start_time = time.time()
    num_features = X_train.shape[1]
    
    # Initialize particle positions (probabilities of picking a feature) and velocities
    positions = np.random.uniform(0, 1, (num_particles, num_features))
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, num_features))
    
    # Track personal bests for each individual particle
    p_best_pos = np.copy(positions)
    p_best_fit = np.array([pso_fitness_function(p, X_train, X_test, y_train, y_test) for p in positions])
    
    # Track global best across the whole swarm
    g_best_idx = np.argmax(p_best_fit)
    g_best_pos = np.copy(p_best_pos[g_best_idx])
    g_best_fit = p_best_fit[g_best_idx]
    
    convergence_history = [g_best_fit]
    
    # Swarm Optimization Loop
    for it in range(max_iter):
        it_start = time.time()
        for i in range(num_particles):
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
            
            # PSO Velocity Update Equation (w=0.5, c1=1.5, c2=1.5)
            velocities[i] = (0.5 * velocities[i] + 
                             1.5 * r1 * (p_best_pos[i] - positions[i]) + 
                             1.5 * r2 * (g_best_pos - positions[i]))
            
            # Bound velocity to avoid chaotic explosions
            velocities[i] = np.clip(velocities[i], -0.2, 0.2)
            
            # Update position
            positions[i] = np.clip(positions[i] + velocities[i], 0, 1)
            
            # Evaluate fitness of new position
            fitness = pso_fitness_function(positions[i], X_train, X_test, y_train, y_test)
            
            # Update Personal Best
            if fitness > p_best_fit[i]:
                p_best_fit[i] = fitness
                p_best_pos[i] = np.copy(positions[i])
                
                # Update Global Best
                if fitness > g_best_fit:
                    g_best_fit = fitness
                    g_best_pos = np.copy(positions[i])
                    
        convergence_history.append(g_best_fit)
        print(f"   ↳ Iteration {it+1}/{max_iter} complete | Swarm Best R²: {g_best_fit:.4f} | Took {time.time()-it_start:.1f}s")
        
    # Identify final selected features
    best_mask = g_best_pos > 0.5
    selected_features = [feature_names[idx] for idx, val in enumerate(best_mask) if val]
    
    # Train the FINAL high-quality model (100 trees) on the optimized feature subset
    X_train_final = X_train[:, best_mask]
    X_test_final = X_test[:, best_mask]
    
    final_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_rf.fit(X_train_final, y_train)
    
    predictions = final_rf.predict(X_test_final)
    final_r2 = r2_score(y_test, predictions)
    final_mae = mean_absolute_error(y_test, predictions)
    
    elapsed_time = time.time() - start_time
    return final_r2, final_mae, elapsed_time, selected_features, convergence_history


if __name__ == "__main__":
    dataset_path = r"C:\Users\USER\Desktop\Weather forecasting2\weather_prediction_dataset.csv"
    target = "HEATHROW_temp_mean" 
    
    print("🔄 Step 1: Preprocessing dataset...")
    X_train, X_test, y_train, y_test, feat_names = prepare_dataset(dataset_path, target)
    print("✅ Preprocessing complete.")
    
    print("\n🔄 Step 2: Running Baseline Random Forest Model (All 164 features)...")
    base_r2, base_mae, base_duration = run_baseline_pipeline(X_train, X_test, y_train, y_test)
    print("🏁 Baseline Execution Results:")
    print(f"   • R-Squared (R²): {base_r2:.4f}")
    print(f"   • Mean Absolute Error (MAE): {base_mae:.2f}°C")
    print(f"   • Execution Time: {base_duration:.2f} seconds")
    
    # --- RUNNING THE MI PIPELINE ---
    K_FEATURES = 15  
    print(f"\n🔄 Step 3: Running MI + Random Forest (Top {K_FEATURES} features)...")
    mi_r2, mi_mae, mi_duration, mi_feats = run_mi_pipeline(X_train, X_test, y_train, y_test, feat_names, k=K_FEATURES)
    print("🏁 MI + RF Execution Results:")
    print(f"   • R-Squared (R²): {mi_r2:.4f}")
    print(f"   • Mean Absolute Error (MAE): {mi_mae:.2f}°C")
    print(f"   • Execution Time: {mi_duration:.2f} seconds")
    print(f"   • Selected Features: {mi_feats[:5]}... (+ {len(mi_feats)-5} more)")

    # --- RUNNING THE NEW PSO PIPELINE ---
    print("\n🔄 Step 4: Running PSO + Random Forest (Swarm Optimization)...")
    pso_r2, pso_mae, pso_duration, pso_feats, history = run_pso_pipeline(
        X_train, X_test, y_train, y_test, feat_names, num_particles=10, max_iter=5
    )
    print("🏁 PSO + RF Execution Results:")
    print(f"   • R-Squared (R²): {pso_r2:.4f}")
    print(f"   • Mean Absolute Error (MAE): {pso_mae:.2f}°C")
    print(f"   • Execution Time: {pso_duration:.2f} seconds")
    print(f"   • Features Selected: {len(pso_feats)} out of 164")