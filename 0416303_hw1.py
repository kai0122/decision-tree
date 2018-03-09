import requests
import numpy as np
from sklearn import tree
import os
import pydotplus 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

res = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
data = res.text.split("\n")
data.pop()
data.pop()

for i in range(len(data)):
	data[i] = data[i].split(',')


data = np.array(data)
x, y = data[:,0:4], data[:,4]
x=x.astype(float)

#*******************************************************************************
resub_confusionMatrix = [[0., 0., 0.],
						 [0., 0., 0.],
						 [0., 0., 0.]]
kFold_confusionMatrix = [[0., 0., 0.],
						 [0., 0., 0.],
						 [0., 0., 0.]]
normalized_resubMatrix = [[0., 0., 0.],
						  [0., 0., 0.],
						  [0., 0., 0.]]
normalized_kFoldMatrix = [[0., 0., 0.],
						  [0., 0., 0.],
						  [0., 0., 0.]]

resub_accuracy_score = 0
kFold_accuracy_score = 0

for i in range(10):
	kf = KFold(n_splits=10, random_state=5, shuffle=True)
	KFold(n_splits=10)
	for train, test in kf.split(x,y):
		x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
	clf = RandomForestClassifier(n_estimators=5,max_depth=3)
	clf = clf.fit(x_train,y_train)
	i_tree = 0
	for tree_in_forest in clf.estimators_:
		with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
			my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
		os.unlink('tree_' + str(i_tree) + '.dot')
		dot_data = tree.export_graphviz(tree_in_forest, out_file=None) 
		graph = pydotplus.graph_from_dot_data(dot_data) 
		graph.write_pdf('tree_' + str(i_tree) + '.pdf') 
		i_tree = i_tree + 1
	resub_confusionMatrix += confusion_matrix(y,clf.predict(x))
	kFold_confusionMatrix += confusion_matrix(y_test,clf.predict(x_test))
	resub_accuracy_score += accuracy_score(clf.predict(x), y)
	kFold_accuracy_score += accuracy_score(clf.predict(x_test), y_test)
	normalized_resubMatrix += confusion_matrix(y,clf.predict(x)) / confusion_matrix(y,clf.predict(x)).astype(np.float).sum(axis=1)
	normalized_kFoldMatrix += confusion_matrix(y_test,clf.predict(x_test)) / confusion_matrix(y_test,clf.predict(x_test)).astype(np.float).sum(axis=1)



resub_confusionMatrix /= 10
kFold_confusionMatrix /= 10
resub_accuracy_score /= 10
kFold_accuracy_score /= 10
normalized_resubMatrix /= 10
normalized_kFoldMatrix /= 10

print "***************** Resubstitution Result *****************"
print "Confusion Matrix:"
print resub_confusionMatrix
print "Normalize Confusion Matrix:"
print normalized_resubMatrix
print "Accuracy Score:"
print resub_accuracy_score

print "******************** 10-Fold Result ********************"
print "Confusion Matrix:"
print kFold_confusionMatrix
print "Normalize Confusion Matrix:"
print normalized_kFoldMatrix
print "Accuracy Score:"
print kFold_accuracy_score
#*******************************************************************************

