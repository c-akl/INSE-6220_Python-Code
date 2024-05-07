
#MULTI-CLASS CLASSIFICATION WITH PYCARET
#Install Pycaret
#!pip install pycaret
#DONE

#Run the below code in your notebook to check the installed version
#from pycaret.utils import version
#version()

import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,5)

# Read CSV file into DataFrame
df = pd.read_csv('/Users/co/Desktop/Obesity Classification.csv')

# Specify the target variable
target_variable = 'Label'

# Setup PyCaret without PCA
clf_no_pca = setup(data=df, target=target_variable, train_size=0.7, session_id=123, normalize=True, pca=False)

# Show the best model and their statistics
best_model_no_pca = compare_models()
print(best_model_no_pca)

# Tune hyperparameters with scikit-learn (default)
tuned_best_model_no_pca = tune_model(best_model_no_pca)
print(tuned_best_model_no_pca)

# Visualize the performance of the best model
plot_model(best_model_no_pca, plot='confusion_matrix')

# Evaluate the performance of the best model
evaluate_model(best_model_no_pca)



# Setup PyCaret with PCA
clf_pca = setup(data=df, target=target_variable, train_size=0.7, session_id=123, normalize=True, pca=True, pca_components=3)

# Show the best model and their statistics
best_model_pca = compare_models()
print(best_model_pca)

# Tune hyperparameters with scikit-learn (default)
tuned_best_model_pca = tune_model(best_model_pca)
print(tuned_best_model_pca)

#Confusion Matrix
from pycaret.classification import create_model

# Create the top three models after applying PCA
top_three_models_pca = compare_models(n_select=3)

# Iterate over the top three models
for i, model in enumerate(top_three_models_pca):
    print(f"Plotting confusion matrix for Model {i + 1}: {model}")
    # Create the model
    model = create_model(model)
    # Plot the confusion matrix
    plot_model(model, plot='confusion_matrix')

#ROC Curves
from pycaret.classification import plot_model

# Iterate over the top three models
for i, model in enumerate(top_three_models_pca):
    print(f"Plotting ROC curve for Model {i + 1}: {model}")
    # Create the model
    model = create_model(model)
    # Plot the ROC curve
    plot_model(model, plot='auc')

# Additional machine learning algorithm (Light Gradient Boosting Machine - LGBM)
lgbm = create_model('lightgbm')

# Tune hyperparameters for LGBM
tuned_lgbm = tune_model(lgbm)

# Evaluate the performance of tuned LGBM
evaluate_model(tuned_lgbm)

# Visualize the performance of tuned LGBM
plot_model(tuned_lgbm, plot='confusion_matrix')

# Plot ROC curve for tuned LGBM
plot_model(tuned_lgbm, plot='auc')

# Summary plot of LGBM
plot_model(tuned_lgbm, plot='feature')


#EXPLAINABLE AI WITH SHAPLEY VALUES
rf_pca = create_model('rf')

tuned_rf_pca = tune_model(rf_pca)

#Shap Summary Plot
tuned_rf_pca = tune_model(rf_pca)


# Summary plot of the three principal components
plot_model(tuned_best_model_pca, plot='feature')

# Force plot
interpret_model(tuned_best_model_pca)

# Combined force plot
interpret_model(tuned_best_model_pca, plot='reason', observation=5)







