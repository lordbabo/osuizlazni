import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn . linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay, accuracy_score


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])
    plt.show()


# ucitaj podatke
df = pd.read_csv("OSU_LV5/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
y = y[:,0]
# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


#a
classes, counts_train=np.unique(y_train,return_counts=True)
classes, counts_test=np.unique(y_test,return_counts=True)
X_axis = np.arange(len(classes))
plt.bar(X_axis - 0.2, counts_train, 0.4, label = 'Train')
plt.bar(X_axis + 0.2, counts_test, 0.4, label = 'Test') 
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.xlabel("Penguins")
plt.ylabel("Counts")
plt.legend()
plt.show()

#b
LogisticReg_Model=LogisticRegression()
LogisticReg_Model.fit(X_train,y_train)

#c
print(LogisticReg_Model.coef_)
print(LogisticReg_Model.intercept_)
#razlika je u tome sto u ovom zadatku imamo vise koeficijenata
#ovdje ima 3x3 matrica dok je u proslom primjeru bila 1x2

#d
plot_decision_regions(X_train,y_train,LogisticReg_Model)
#podatci su dobro podijeljeni, s minimalnim greskama

#e
y_predict=LogisticReg_Model.predict(X_test)
cm=confusion_matrix(y_test,y_predict)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print("Accuracy",accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
