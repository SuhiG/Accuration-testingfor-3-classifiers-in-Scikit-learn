from sklearn import tree #importing decision tree classifier
from sklearn.neural_network import MLPClassifier #importing MLP classifier
from sklearn.neighbors import KNeighborsClassifier #importing KNeighbors classifier


def treecls(a,b): #defying fuction for tree classifier
	clf=tree.DecisionTreeClassifier() 
	clf=clf.fit(a,b)#fitting the dataset to the classifier
	prediction=clf.predict([[150, 70, 43]]) #giving the dataset to be predicted based on the previous data sets and printing it
	print(prediction)
	print(clf.predict_proba([[150, 70, 43]]))	#printing the prediction probability

def mlp(a,b): #defying mlp classifier

	clf=MLPClassifier()
	clf=clf.fit(a,b) #fitting the dataset to the classifier
	prediction=clf.predict([[150, 70, 43]])#giving the dataset to be predicted based on the previous data sets and printing it
	print(prediction)
	c=clf.predict_proba([[150, 70, 43]])#printing the prediction probability
	print(c)

def Kneigh(a,b): #defying the KNeighbor classifier
	neigh=KNeighborsClassifier()
	neigh.fit(a,b)#fitting the dataset to the classifier
	print(neigh.predict([[150, 70, 43]]))#giving the dataset to be predicted based on the previous data sets and printing it
	print(neigh.predict_proba([[150, 70, 43]]))#printing the prediction probability

#dataset 1
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#dataset 2
Y = ["male", "female", "female", "female", "male", "male", "female", "female",
     "female", "male", "male"]

#calling the functions
treecls(X,Y)
mlp(X,Y)
Kneigh(X,Y)
