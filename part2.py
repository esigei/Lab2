
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris=datasets.load_iris()
xtrain,xtest,ytrain,ytest=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
#part a
my_classification=svm.SVC(kernel='poly',degree=4).fit(xtrain,ytrain)
print("Poly Accuracy: ",my_classification.score(xtest,ytest))
#partb
myrbf=svm.SVC(kernel='rbf',C=1).fit(xtrain,ytrain)
print('RBF Accuracy:',myrbf.score(xtest,ytest))
#part C
change=svm.SVC(kernel='rbf',C=0.8,gamma='auto').fit(xtrain,ytrain)
print('changes: ', change.score(xtest,ytest))

#part D
"""
Poly Kernel has an accuracy of 96.67% ,RBF kernel has 95.0% accuracy while rbf kernel with change in C and gamma 
gave a least accuracy of 91.67%. So the polynomial Kernel gives the best fit and better accuracy.


"""