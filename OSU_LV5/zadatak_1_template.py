import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn . linear_model import LogisticRegression
from sklearn . metrics import accuracy_score, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="rainbow")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x', cmap="rainbow")
plt.show()

#b
LogisticReg_Model = LogisticRegression()
LogisticReg_Model.fit(X_train,y_train)

#c
theta0=LogisticReg_Model.intercept_
theta1,theta2=LogisticReg_Model.coef_[0]
print(LogisticReg_Model.coef_)
print(LogisticReg_Model.intercept_)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap="rainbow")
plt.plot(X_train[:,0],(-theta0-theta1*X_train[:,0])/theta2)
plt.show()


#d
y_predict=LogisticReg_Model.predict(X_test)

cm=confusion_matrix(y_test,y_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="inferno")
plt.show()

print("Accuracy",accuracy_score(y_test,y_predict))
print("Precision",precision_score(y_test,y_predict))
print("Recall",recall_score(y_test,y_predict))

#e
y_correct = y_predict == y_test
plt.scatter(X_test[:,0],X_test[:,1],c=y_correct,cmap=matplotlib.colors.ListedColormap(['black','green']))
plt.show()