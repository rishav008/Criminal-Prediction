
import pandas as pd

dataset_train = pd.read_csv('criminal_train.csv')
#X = dataset_train.iloc[:, 1:71].values
X = dataset_train.iloc[:, [1, 2, 5, 7, 13, 14, 26, 30, 32, 34, 35, 36, 39, 41, 44, 48, 56, 57, 59, 68, 69]].values
y = dataset_train.iloc[:, 71].values

dataset_train2 = pd.read_csv('criminal_test.csv')
z = dataset_train2.iloc[:, [1, 2, 5, 7, 13, 14, 26, 30, 32, 34, 35, 36, 39, 41, 44, 48, 56, 57, 59, 68, 69]].values
#z = dataset_train.iloc[:, 1:71].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
z = sc.transform(z)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 150, kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 32, epochs = 100)


y_pred2 = classifier.predict(z)
y_pred2 = (y_pred2 > 0.5)



