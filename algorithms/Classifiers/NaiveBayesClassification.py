from rest_framework.response import Response
from rest_framework import generics
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from algorithms.TextNormalization import convert_negations, remove_stop_words, remove_special_characters
import pandas as pd
clf = None
vectorizer = None

class NaiveBayes(generics.GenericAPIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global clf, vectorizer
        if clf is None or vectorizer is None:
            clf, vectorizer = self.train_model()

    def train_model(self):
        df = pd.read_csv(r"algorithms\UpdatedDisneyLand.csv", encoding='latin-1')

        n_samples = df['Rating'].value_counts().min()
        df_balanced = pd.concat([df[df['Rating'] == rating].sample(n=n_samples, random_state=42) for rating in range(1, 6)])
        df_balanced = df_balanced.sample(frac=1, random_state=42)

        RatingsArray = df_balanced["Rating"]
        oldReviewTextArray = df_balanced["Review_Text"]
        ReviewTextArray = []

        for element in oldReviewTextArray:
            newWord = element.lower()
            newWord = convert_negations(newWord)
            newWord = remove_stop_words(newWord)
            newWord = remove_special_characters(newWord)
            substrings = ["park", "disney", "disneyland", "rides", "land", "time", "get", "day", "go", "people",
                          "one", "ride", "would", "kid", "place", "how", "year", "food", "2", "like", "kids", "parks",
                          "paris", "see", "is", "i'm", "me", "you", "were", "was", "have", "has", "disneyworld"]
            pattern = r"\b(?:{})\b".format("|".join(map(re.escape, substrings)))
            for substring in substrings:
                if substring in newWord:
                    newWord = re.sub(pattern, "", newWord)

            ReviewTextArray.append(newWord)
        global vectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(ReviewTextArray)

        X_train, X_test, RatingsArray_train, RatingsArray_test = train_test_split(X, RatingsArray, test_size=0.2)
        global clf
        clf = MultinomialNB()
        clf.fit(X_train, RatingsArray_train)
        return clf,vectorizer

    def post(self, request):
        newValue = request.data["text"]
        newValues = []
        newValue = convert_negations(newValue)
        newValues.append(newValue)
        global vectorizer, clf
        vectorizedInput = vectorizer.transform(newValues)
        predictedValue = clf.predict(vectorizedInput)
        return Response({
            "prediction": predictedValue[0]
        })
