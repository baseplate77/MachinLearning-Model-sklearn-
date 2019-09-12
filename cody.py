from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pyttsx3
import numpy as np

#this is data set

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

X_test = [[190,90,44],[170,70,32],[150,95,39],[145,56,31],[189,69,45],[120,39,20],[99,69,49],[160,59,29],
          [145,89,46],[120,36,19],[179,69,43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#this part of code is use for classification through perceptron model

clk_perceptron = Perceptron()

clk_perceptron.fit(X,Y)

per_test = clk_perceptron.predict(X)
acc_per = accuracy_score(Y,per_test)
print(acc_per)

#this part of code is use for classification through Support Vector Machine model


clk_svm = SVC()
clk_svm.fit(X,Y)
per_svm = clk_svm.predict(X)

acc_svm = accuracy_score(Y,per_svm)
print(acc_svm)

#this part of code is use for classification through KNN model

clk_knn = KNeighborsClassifier()
clk_knn.fit(X,Y)
per_knn = clk_knn.predict(X)
acc_knn = accuracy_score(Y,per_knn)
print(acc_knn)

#this part of code is use for classification by DecisionTreeClassifier model

clk_tree = tree.DecisionTreeClassifier()
clk_tree.fit(X,Y)
per_tree = clk_tree.predict(X)
acc_tree = accuracy_score(Y,per_tree)
print(acc_tree)

#this part of code is use to genrate audio

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[0].id)
list = [acc_knn, acc_per, acc_svm, acc_tree]

classifier = {0: 'KNeighborsClassifier', 1: ' Perceptron ', 2: 'SupportVectorMachine', 3: 'DecisionTreeClassifier'}


def acc_score_compare():
    index = np.argmax(list)

    return index


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


if __name__ == "__main__":
    i =acc_score_compare()
    speak('best model for classifying this data is ' + classifier[i])
