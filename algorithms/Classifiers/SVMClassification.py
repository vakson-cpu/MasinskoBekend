from rest_framework.response import Response
from rest_framework import generics
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pandas as pd
from algorithms.TextNormalization import preprocess_text

# Global variables to store the trained SVM model and vectorizer
clf = None
vectorizer = None

class SVM(generics.GenericAPIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if the SVM model is already trained
        global clf, vectorizer
        if clf is None or vectorizer is None:
            clf, vectorizer = self.train_model()

    def train_model(self):
        df = pd.read_csv(r"algorithms\UpdatedDisneyLand.csv", encoding='latin-1')

        n_samples = df['Rating'].value_counts().min()
        df_balanced = pd.concat([df[df['Rating'] == rating].sample(n=n_samples, random_state=42) for rating in range(1, 6)])
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        # numberOfMin = df_balanced[df['Rating'] == 5]
        # num_rows = len(numberOfMin)
        RatingsArray = df_balanced["Rating"]
        ReviewTextArray = df_balanced["Review_Text"]

        ReviewTextArray = [preprocess_text(text) for text in ReviewTextArray]

        global vectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(ReviewTextArray)

        X_train, X_test, RatingsArray_train, RatingsArray_test = train_test_split(X, RatingsArray, test_size=0.2, random_state=42)

        oversampler = SMOTE(random_state=42)
        X_train, RatingsArray_train = oversampler.fit_resample(X_train, RatingsArray_train)

        global clf
        clf = SVC()
        clf.fit(X_train, RatingsArray_train)

        predicted_ratings = clf.predict(X_test)
        accuracy = accuracy_score(RatingsArray_test, predicted_ratings)
        report = classification_report(RatingsArray_test, predicted_ratings)
        return clf, vectorizer

    def post(self, request):
        newValue = request.data["text"]
        newValue = preprocess_text(newValue)
        newValues = [newValue]

        global vectorizer, clf
        vectorizedInput = vectorizer.transform(newValues)
        predictedValue = clf.predict(vectorizedInput)
        print(predictedValue)

        return Response({
            "prediction": predictedValue[0]
        })
