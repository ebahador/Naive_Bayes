import pandas as pnd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

clf = GaussianNB()  # Classifier
data = pnd.DataFrame()

# Create our target variable (Y)
data['Gender'] = ['male', 'male', 'male', 'male', 'male',
                  'female', 'female', 'female', 'female', 'female']

# Create our feature variables (X)
data['Height'] = [182.88, 180.44, 170.07, 180.44, 185,
                  152.4, 167.64, 165.2, 175.26, 168.43]  # in CM

data['Weight'] = [81.64, 86.18, 77.11, 74.84, 90.5,
                  45.35, 68.03, 58.96, 68.03, 60.61]  # in KG

data['Foot_Size'] = [12, 11, 12, 10, 10, 6, 8, 7, 9, 6]

# print(data):
#    Gender  Height  Weight  Foot_Size
# 0    male  182.88   81.64         12
# 1    male  180.44   86.18         11
# 2    male  170.07   77.11         12
# 3    male  180.44   74.84         10
# 4    male  185.00   90.50         10
# 5  female  152.40   45.35          6
# 6  female  167.64   68.03          8
# 7  female  165.20   58.96          7
# 8  female  175.26   68.03          9
# 9  female  168.43   60.61          6

X = data[['Height', 'Weight', 'Foot_Size']]
Y = data['Gender']

# We can predict Y variable from X variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  # %20
clf.fit(X_train, Y_train)

print(clf.predict(X_test))
print(clf.predict([[170, 50, 12]]))
# our prediction expectation is a matrix, so we use [[something, another thing]] (double[[]])
