from rest_framework.response import Response
from rest_framework import generics
import re
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from algorithms.TextNormalization import remove_stop_words, convert_negations
import pandas as pd

vectorizer = None
clf = None

class LogisticRegressionView(generics.GenericAPIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global vectorizer, clf
        if vectorizer is None or clf is None:
            vectorizer, clf = self.train_model()

    def train_model(self):
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(r"algorithms\UpdatedDisneyLand.csv", encoding='latin-1')

        RatingsArray = df["Rating"]
        oldReviewTextArray = df["Review_Text"]
        ReviewTextArray = []

        for element in oldReviewTextArray:
            newWord = element.lower()
            newWord = convert_negations(newWord)
            newWord = remove_stop_words(newWord)
            substrings = ["park", "disney", "disneyland", "rides", "land", "time", "get", "day", "go", "people",
                          "one", "ride", "would", "kid", "place", "how", "year", "food", "2", "like", "kids", "parks",
                          "paris", "see", "is", "i'm", "me", "you", "were", "was", "have", "has", "disneyworld"]
            pattern = r"\b(?:{})\b".format("|".join(map(re.escape, substrings)))
            for substring in substrings:
                if substring in newWord:
                    newWord = re.sub(pattern, "", newWord)

            ReviewTextArray.append(newWord)

        # Extract the features and labels
        y = RatingsArray

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(ReviewTextArray, y, test_size=0.3, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Create a Logistic Regression classifier
        clf = SklearnLogisticRegression()

        # Train the classifier
        clf.fit(X_train, y_train)

        return vectorizer, clf

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
