from django.urls import path
from algorithms.Classifiers.KnnClassification import KnnView
from algorithms.Classifiers.NaiveBayesClassification import NaiveBayes
from algorithms.Classifiers.J48Classification import J48View
from algorithms.Classifiers.SVMClassification import SVM
from algorithms.Classifiers.LogisticRegressionClassification import LogisticRegressionView
urlpatterns = [
    path('Knn', KnnView.as_view()),
    path('NaiveBayes', NaiveBayes.as_view()),
    path('SVM', SVM.as_view()),
    path('J48', J48View.as_view()),
    path('LogisticRegression', LogisticRegressionView.as_view()),
    
]