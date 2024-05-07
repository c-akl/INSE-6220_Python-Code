
#IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
import pandas as pd
plt.rcParams['figure.figsize'] = (7,5)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import LabelEncoder
from pca import pca

import csv
from matplotlib.ticker import FixedLocator
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from scipy.stats import beta
from scipy.stats import f

from itertools import cycle

#Machine Learning Modules
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#UPLOAD FILE

#Read data file
df = pd.read_csv ('/Users/co/Desktop/Obesity Classification.csv')
df.head(n=108)
df.info()

# Convert non-numerical columns to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

#Labels
y = df['Label']
target = df['Label'].to_numpy()

X = df.drop(columns=['Label', 'ID'])
print(X.head(108))
print('\n')
print(X.describe())

#Standardized Data
X_st = StandardScaler().fit_transform(X)
df = pd.DataFrame(X_st)
df.columns = X.columns
print(df.describe())

#Observations and Variables
observations = list(df.index)
print(observations)
variables = list(df.columns)
print(variables)


#DATA VISUALISATION

#a. Label Distribution
# Sample data representing frequencies
categories = ['Normal weight', 'Underweight', 'Overweight', 'Obese']
frequencies = [29, 47, 20, 12]  # Example frequencies, you should replace this with your actual data

# Create histogram
plt.bar(categories, frequencies, color=['blue', 'orange', 'green', 'red'])

# Labeling axes and title
plt.xlabel('Weight Categories')
plt.ylabel('Frequency')
plt.title('Weight Distribution')

# Show plot
plt.show()

#b. Box & Whisker Plots
ax = plt.figure()
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.title('Box & Whisker Plot')
plt.show()

#c. Use swarmplot() or stripplot to show the datapoints on top of the boxes:
ax = plt.figure()
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax = sns.stripplot(data=df, color=".25")
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('Swarm Plot')
plt.show()
print(df.describe())

#d. Pair Plot
sns.pairplot(df)
plt.show()


#COVARIANCE
dfc = df - df.mean() #centered data
ax = sns.heatmap(dfc.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True,
            cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0);
plt.title('Covariance Matrix')
plt.show()


#PRINCIPAL COMPONENT ANALYSIS (PCA)
#Note that the data was already standardized

pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)

idx_Normalweight = np.where(y == 0)
idx_Underweight = np.where(y == 1)
idx_Overweight = np.where(y == 2)
idx_Obese = np.where(y == 3)

# Transform the data into principal component space
pc_scores = pca.transform(df)

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

plt. figure()
sc = plt.scatter(pc_scores[:,0], pc_scores[:,1], c=target_encoded, cmap='viridis')
plt.legend(handles=sc.legend_elements()[0], labels=['Normal Weight', 'Underweight', 'Overweight', 'Obese']) #Add Legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.colorbar(label='Label')
plt.show()

#Eigenvalues & Eigenvectors
A = pca.components_.T
# Display eigenvalues
eigenvalues = pca.explained_variance_
print("Eigenvalues:")
print(eigenvalues)

# Print the principal components (eigenvectors)
print("Principal Components (Eigenvectors):")
print(A)

plt.figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Components Visualization')

# Annotate principal components with feature labels with offset and avoiding overlap
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    if label == 'BMI':  # Adjust placement for 'BMI' annotation
        plt.annotate(label, xy=(x, y), xytext=(-10, -15), textcoords='offset points', ha='right', va='top')
    elif label == 'Height':  # Adjust placement for 'Height' annotation
        plt.annotate(label, xy=(x, y), xytext=(5, 10), textcoords='offset points', ha='left', va='bottom')
    elif label == 'Weight':  # Adjust placement for 'weight' annotation
        plt.annotate(label, xy=(x, y), xytext=(15, 5), textcoords='offset points', ha='right', va='bottom')
    else:
        plt.annotate(label, xy=(x, y), xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')

plt.grid(True)
plt.show()

#PC Visualization with Annotation
plt.figure(figsize=(8, 6))
plt.scatter(A[:, 0], A[:, 1], marker='o', c='b', s=100)  # Adjust the color and size of the points
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    if label == 'BMI':  # Adjust placement for 'BMI' annotation
        plt.annotate(label, xy=(x, y), xytext=(-10, -15), textcoords='offset points', ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=5))
    elif label == 'Height':  # Adjust placement for 'Height' annotation
        plt.annotate(label, xy=(x, y), xytext=(5, 10), textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=5))
    elif label == 'Weight':  # Adjust placement for 'Weight' annotation
        plt.annotate(label, xy=(x, y), xytext=(20, 10), textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=5))
    else:
        plt.annotate(label, xy=(x, y), xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=5))

plt.grid(True)
plt.title('Principal Components Visualization with Annotation')
plt.show()

#Scree Plot
#Eigenvalues
Lambda = pca.explained_variance_
#print(f'Eigenvalues:\n{Lambda}')

#Scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/sum(Lambda), 'ro-', lw=3)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.title('Scree Plot')
plt.show()

#Explained Variance
ell = pca.explained_variance_ratio_
plt.figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained Variance')
plt.show()

#Explained Variance per PC
PC_variance = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
print(PC_variance)

# Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[:, 0]
A2 = A[:, 1]
Z1 = Z[:, 0]
Z2 = Z[:, 1]

plt.figure()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i in range(len(A1)):
    # Arrows project features as vectors onto PC axes
    plt.arrow(0, 0, A1[i] * max(Z1), A2[i] * max(Z2), color='k', width=0.01, head_width=0.05, alpha=0.75,
              length_includes_head=True)

    # Adjusting height annotation y-coordinate to -1.8
    if variables[i] == 'Height':
        plt.text(A1[i] * max(Z1) * 1.2, A2[i] * max(Z2) * 1.2 + 0.35, variables[i], color='k', verticalalignment='top')
    else:
        plt.text(A1[i] * max(Z1) * 1.2, A2[i] * max(Z2) * 1.2, variables[i], color='k')

plt.scatter(Z[idx_Normalweight, 0], Z[idx_Normalweight, 1], c='b', label='Normal Weight (0)')
plt.scatter(Z[idx_Underweight, 0], Z[idx_Underweight, 1], c='r', label='Underweight (1)')
plt.scatter(Z[idx_Overweight, 0], Z[idx_Overweight, 1], c='g', label='Overweight (2)')
plt.scatter(Z[idx_Obese, 0], Z[idx_Obese, 1], c='y', label='Obese (3)')

plt.legend(loc='upper left')
plt.title('Biplot')
plt.show()

#Pareto Chart
from pca import pca
# Initialize and keep all PCs
model = pca()
# Fit transform
out = model.fit_transform(df)
# Print the top features. The results show that f1 is best, followed by f2 etc
print(out['topfeat'])
model.plot();
plt.show()

# Biplot with Loadings, Enumerated Points, and Arrow Vectors
#plt.figure(figsize=(10, 8))

# Plot PC scores
#for i, (pc1, pc2) in enumerate(zip(Z[:, 0], Z[:, 1])):
    #plt.scatter(pc1, pc2, c=target_encoded[i], cmap='viridis')
    #plt.text(pc1, pc2, str(i+1), fontsize=8, ha='center', va='center')

# Plot feature loadings with arrow vectors
#for i, txt in enumerate(variables):
   # plt.annotate(txt, (A[i, 0], A[i, 1]), fontsize=10, ha='center', va='center')
    #plt.arrow(0, 0, A[i, 0], A[i, 1], color='k', width=0.005, head_width=0.03)

#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.title('Biplot with Loadings, Enumerated Points, and Arrow Vectors')

#plt.colorbar(label='Label')
#plt.legend()
#plt.grid(True)
#plt.show()

#model.biplot(legend=False, hotellingt2=True) #TypeError: pca.biplot() got an unexpected keyword argument 'hotellingt2'

model.biplot(cmap=None, label=False, legend=False)
plt.show()

#3D plot
ax = model.biplot3d(legend=False)
plt.show()

# Principal Components
comps = pd.DataFrame(A, columns=variables)
sns.heatmap(comps, cmap='RdYlGn_r', linewidths=0.5, annot=True,
            cbar=True, square=True)
ax = plt.gca()  # Get the current axis
ax.tick_params(labelbottom=False, labeltop=True)  # Modify tick parameters
plt.xticks(rotation=90)
plt.title('Covariance Matrix for Principal components')
plt.show()

print(f'PC1:{A1}')
print(f'PC2:{A2}')

#Hotelling's T2 Test
alpha = 0.05
p=Z.shape[1]
n=Z.shape[0]

UCL=((n-1)**2/n )*beta.ppf(1-alpha, p / 2 , (n-p-1)/ 2)
UCL2=p*(n+1)*(n-1)/(n*(n-p) )*f.ppf(1-alpha, p , n-p)
Tsquare=np.array([0]*Z.shape[0])
for i in range(Z.shape[0]):
  Tsquare[i] = np.matmul(np.matmul(np.transpose(Z[i]),np.diag(1/Lambda) ) , Z[i])

fig, ax = plt.subplots()
ax.plot(Tsquare,'-b', marker='o', mec='y',mfc='r' )
ax.plot([UCL for i in range(len(Z1))], "--g", label="UCL")
plt.ylabel('Hotelling')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))
plt.title('Hotelling T2 Test')
plt.show()

print(np.argwhere(Tsquare>UCL)) #this function will print the points with T^2 greater than the upper control limit (points 27, 56, 101, and 104)

#Control Charts for Principal Components
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(Z1,'-b', marker='o', mec='y',mfc='r')
ax1.plot([3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label="UCL")
ax1.plot([-3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--b", label='LCL')
ax1.plot([0 for i in range(len(Z1))], "-", color='black',label='CL')
ax1.set_ylabel('Principal Component 1')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

ax2.plot(Z2,'-b', marker='o', mec='y',mfc='r')
ax2.plot([3*np.sqrt(Lambda[1]) for i in range(len(Z2))], "--g", label="UCL")
ax2.plot([-3*np.sqrt(Lambda[1]) for i in range(len(Z2))], "--b", label='LCL')
ax2.plot([0 for i in range(len(Z2))], "-", color='black',label='CL')
ax2.set_ylabel('Principal Component 2')
ax2.set_xlabel('Sample Number')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))
plt.legend()
plt.title('Control Charts for Principal Components')
plt.show()

#Print out of control points (no point is outside either control limits)
print(np.argwhere(Z1<-3*np.sqrt(Lambda[0])))
print(np.argwhere(Z1>3*np.sqrt(Lambda[0])))
print(np.argwhere(Z2<-3*np.sqrt(Lambda[1])))
print(np.argwhere(Z2>3*np.sqrt(Lambda[1])))


#MULTI-CLASS CLASSIFICATION
# Test-Train Split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(df, target_encoded, test_size=0.3, random_state=0)
print(f'Train Dataset Size: {X_train.shape[0]}')
print(f'Test Dataset Size: {X_test.shape[0]}')

# Train the classifier on the training data
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Evaluate the classifier on the training set
train_predictions = classifier.predict(X_train)
train_report = classification_report(y_train, train_predictions)
print("Training Classification Report:")
print(train_report)

# Evaluate the classifier on the test set
test_predictions = classifier.predict(X_test)
test_report = classification_report(y_test, test_predictions)
print("Testing Classification Report:")
print(test_report)


#a. Gaussian Naive Bayes (GNB)
# Test-Train Split for different datasets
Z_train, Z_test, zy_train, zy_test = train_test_split(pc_scores, target_encoded, test_size=0.3, random_state=0)
Z12_train, Z12_test, z12y_train, z12y_test = train_test_split(pc_scores[:, :2], target_encoded, test_size=0.3, random_state=0)

#Gaussian Naive Bayes (GNB)
gnb = GaussianNB()

datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test),
            ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
    gnb.fit(Xtr, ytr)
    y_pred = gnb.predict(Xtst)
    gnb_score = gnb.score(Xtst, ytst)

    # Classification Report
    print(f'DATASET: {name}')
    print('Classification Report:')
    print(classification_report(ytst, y_pred, digits=3))

    # Confusion Matrix
    cm_gnb = confusion_matrix(y_true=ytst, y_pred=y_pred)
    ax = sns.heatmap(cm_gnb, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
    plt.yticks(rotation=0)
    ax.tick_params(labelbottom=False, labeltop=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()

    # Plotting decision regions
    if name == 'Z12':
        x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
        y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
        plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.legend()
        plt.show()

        print(np.where(ytst != y_pred))


#b. K Nearest Neighbors (KNN)
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Assuming 'df', 'target_encoded', and 'pc_scores' are defined previously

# Hyperparameter grid search for k
param_grid = {'n_neighbors': [2, 4, 8, 16, 32]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Test-Train Split for different datasets
X_train, X_test, y_train, y_test = train_test_split(df, target_encoded, test_size=0.3, random_state=0)
Z_train, Z_test, zy_train, zy_test = train_test_split(pc_scores, target_encoded, test_size=0.3, random_state=0)
Z12_train, Z12_test, z12y_train, z12y_test = train_test_split(pc_scores[:, :2], target_encoded, test_size=0.3, random_state=0)

# Find best k
knn_full_data = grid_search.fit(X_train, y_train)
knn_Z = grid_search.fit(Z_train, zy_train)
knn_Z12 = grid_search.fit(Z12_train, z12y_train)

# Get best k
print('Grid Search Results:')
k_full_data = knn_full_data.best_params_
k_Z = knn_Z.best_params_
k_Z12 = knn_Z12.best_params_
print(f'k_full_data: {k_full_data}\nk_Z: {k_Z}\nk_Z12: {k_Z12}')

# Define the evaluation metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Apply best k
knn = KNeighborsClassifier(n_neighbors=k_full_data.get('n_neighbors'))
scores_knn_full_data = cross_validate(knn, X_train, y_train, cv=5, scoring=scoring)
scores_knn_Z = cross_validate(knn, Z_train, zy_train, cv=5, scoring=scoring)
scores_knn_Z12 = cross_validate(knn, Z12_train, z12y_train, cv=5, scoring=scoring)

# Store the scores in a dictionary
knn_scores_dict = {}
for i in scoring:
    knn_scores_dict[f"knn_full_data {i}"] = scores_knn_full_data[f'test_{i}']
    knn_scores_dict[f"knn_Z {i}"] = scores_knn_Z[f'test_{i}']
    knn_scores_dict[f"knn_Z12 {i}"] = scores_knn_Z12[f'test_{i}']

knn_scores_data = pd.DataFrame(knn_scores_dict)
print(f'{knn_scores_data}\n')

datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test),
            ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
    # Apply on train-test split
    knn.fit(Xtr, ytr)
    y_pred = knn.predict(Xtst)

    # Classification Report
    print(f'DATASET: {name}')
    print('Classification Report:')
    print(classification_report(ytst, y_pred, digits=3))

    # Confusion Matrix
    cm_knn = confusion_matrix(y_true=ytst, y_pred=y_pred)
    ax = sns.heatmap(cm_knn, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
    plt.yticks(rotation=0)
    ax.tick_params(labelbottom=False, labeltop=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()

    # Plotting decision regions
    if name == 'Z12':
        x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
        y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
        plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.legend()
        plt.show()

        print(np.where(ytst != y_pred))


#c. Decision Trees (DT)
# Hyperparameter search for DT depth
param_grid = {'max_depth': [2, 4, 8, 16, 32, 64]}
dt = DecisionTreeClassifier(random_state=0)
grid_search = GridSearchCV(dt, param_grid, cv=5)

# Find best depth
dt_full_data = grid_search.fit(X_train, y_train)
dt_Z = grid_search.fit(Z_train, zy_train)
dt_Z12 = grid_search.fit(Z12_train, z12y_train)

# Get best tree depth
print('Grid Search Results:')
depth_full_data = dt_full_data.best_params_
depth_Z = dt_Z.best_params_
depth_Z12 = dt_Z12.best_params_
print(f'depth_full_data: {depth_full_data}\ndepth_Z: {depth_Z}\ndepth_Z12: {depth_Z12}')

# Apply best k
dt = DecisionTreeClassifier(max_depth=depth_full_data.get('max_depth'))
scores_dt_full_data = cross_validate(dt, X_train, y_train, cv=5, scoring=scoring)
scores_dt_Z = cross_validate(dt, Z_train, zy_train, cv=5, scoring=scoring)
scores_dt_Z12 = cross_validate(dt, Z12_train, z12y_train, cv=5, scoring=scoring)

dt_scores_dict={}
for i in ['fit_time', 'test_f1_macro']:
  dt_scores_dict["dt_full_data " + i ] = scores_dt_full_data[i]
  dt_scores_dict["dt_Z  " + i ] = scores_dt_Z[i]
  dt_scores_dict["dt_Z12 " + i ] = scores_dt_Z12[i]

dt_scores_data = pd.DataFrame(dt_scores_dict).T
#dt_scores_data['avgs'] = dt_scores_data.mean(axis=1)
print(f'{dt_scores_data}\n')

datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test), ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
  # Apply on train-test split
  dt.fit(Xtr, ytr)
  y_pred = dt.predict(Xtst)
  dt_score = dt.score(Xtst, ytst)
  #print(dt_score)

  # Classification Report
  print(f'DATASET: {name}')
  print('Classification Report:')
  print(classification_report(ytst, y_pred, digits=3))

  # Confusion Matrix
  cm_dt = confusion_matrix(y_true=ytst, y_pred=y_pred)
  ax = sns.heatmap(cm_dt, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
  plt.yticks(rotation=0)
  ax.tick_params(labelbottom=False,labeltop=True)
  ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
  #plt.title('Decision Tree Confusion Matrix')
  plt.show()

  #ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
  if name == 'Z12':
    # Plotting decision regions
    x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
    y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
    plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.legend()
    plt.show()


#d. ROC Curves
#ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test), ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
for i, (name, X_tr, y_tr, X_tst, y_tst) in enumerate(datasets):
  # Binarize the labels
  y_train = label_binarize(y_tr, classes=[0, 1, 2])
  y_test = label_binarize(y_tst, classes=[0, 1, 2])
  n_classes = y_train.shape[1]
  print(f'DATASET: {name}')

  list_algos = [gnb, knn, dt]
  algo_name = ['Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree']
  for i, (algo, algo_name) in enumerate(zip(list_algos, algo_name)):
    classifier = OneVsRestClassifier(algo)
    y_pred = classifier.fit(X_tr, y_train).predict(X_tst)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel(), drop_intermediate=False)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig, ax = plt.subplots()

    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', color="deeppink", linestyle=':')
    plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', color="navy", linestyle=':')

    colors = cycle(['c', 'm', 'r'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i],tpr[i], color=color,label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{algo_name}')
    plt.legend()
    plt.show()


#e. Bar Chart Plot
# ADAPTED FROM: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
def autolabel(rects):
    for r in rects:
        height = r.get_height()
        ax.annotate(f'{height}', xy=(r.get_x() + r.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

n_groups = 3
ind = np.arange(n_groups)

# F1 Scores from above
NB = (0.897, 0.955, 0.941)
KNN = (0.906, 0.906, 0.903)
DT = (0.918, 0.898, 0.870)

# create plot
fig, ax = plt.subplots(figsize=(10,7))
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

rects1 = plt.bar(index, NB, bar_width, alpha=opacity, color='b', label='Naive Bayes')
rects2 = plt.bar(index + bar_width, KNN, bar_width, alpha=opacity, color='y', label='K-Nearest Neighbors')
rects3 = plt.bar(index + bar_width*2, DT, bar_width, alpha=opacity, color='k', label='Decision Tree')

ax.set_xlabel('Data Set')
ax.set_ylabel('Macro-F1 Scores')
#plt.title(f'')
plt.xticks(index + bar_width, ('Original Data', 'All PCs', 'Two PCs'))
plt.legend(loc="lower right")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.title('Bar Chart Plot')
plt.show()


































