### 🎯 Student Final Grade Prediction (Elastic Net)  

A predictive model using **Elastic Net** regression to estimate students' final grades.  

#### 🚀 Features  
- **Elastic Net Regression**: Balances L1 & L2 regularization  
- **Hyperparameter Tuning**: Optimized with **GridSearchCV**  
- **Standardized Data**: Preprocessed for better accuracy  

#### 📌 Usage  
```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GridSearchHelper import get_param_grid, perform_grid_search

# Data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ElasticNet model
model = ElasticNet()
param_grid = get_param_grid(model)
best_model = perform_grid_search(model, param_grid, X_train_scaled, y_train)

print(f"Best Parameters: {best_model.best_params_}")
```
 
