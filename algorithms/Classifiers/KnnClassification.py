from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status, generics
import math
from datetime import datetime
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from algorithms.TextNormalization import convert_negations
import pandas as pd

vectorizer = None
clf = None

class KnnView(generics.GenericAPIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global vectorizer, clf
        if vectorizer is None or clf is None:
            vectorizer, clf = self.train_model()

    def train_model(self):
        df = pd.read_csv(r"algorithms\UpdatedDisneyLand.csv", encoding='latin-1')
        n_samples = df['Rating'].value_counts().min()
        df_balanced = pd.concat([df[df['Rating'] == rating].sample(n=n_samples, random_state=42) for rating in range(1, 6)])
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        RatingsArray = df_balanced["Rating"]
        ReviewTextArray = df_balanced["Review_Text"]

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r"\b(?:park|disney|disneyland|rides|land|time|get|day|go|people|one|ride|would|kid|place|how|year|food|2|like|kids|parks|paris|see|is|i'm|me|you|were|was|have|has|disneyworld)\b", "", text)
            return text

        ReviewTextArray = [preprocess_text(text) for text in ReviewTextArray]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(ReviewTextArray)

        X_train, X_test, RatingsArray_train, RatingsArray_test = train_test_split(X, RatingsArray, test_size=0.2, random_state=42)

        oversampler = SMOTE(random_state=42)
        X_train, RatingsArray_train = oversampler.fit_resample(X_train, RatingsArray_train)

        clf = KNeighborsClassifier()
        clf.fit(X_train, RatingsArray_train)

        return vectorizer, clf

    def post(self, request):
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r"\b(?:park|disney|disneyland|rides|land|time|get|day|go|people|one|ride|would|kid|place|how|year|food|2|like|kids|parks|paris|see|is|i'm|me|you|were|was|have|has|disneyworld)\b", "", text)
            return text

        newValue = request.data["text"]
        newValues = []
        newValue = preprocess_text(newValue)
        newValues.append(newValue)
        global vectorizer, clf
        vectorizedInput = vectorizer.transform(newValues)
        predictedValue = clf.predict(vectorizedInput)

        return Response({
            "prediction": predictedValue[0]
        })
