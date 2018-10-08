import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import imp
# Load iris data
iris = sns.load_dataset("iris")
#part a
sns.pairplot(iris, hue="species", height=2.5)
plt.show()
#part b
# splitting data into training and test set with  test size of 20 percent
xtrain,xtest, ytrain, ytest=train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
gnb=GaussianNB()# creating an instance of Naive Bayes
gnb.fit(xtrain, ytrain)
expected_outcomes = ytest
prediction = gnb.predict(xtest)
print("Accuracy: ", metrics.accuracy_score(expected_outcomes, prediction))
