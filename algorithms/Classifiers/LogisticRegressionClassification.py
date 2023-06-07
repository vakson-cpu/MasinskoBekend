from rest_framework.response import Response
from rest_framework import generics
import re
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from algorithms.TextNormalization import tokenize,remove_special_characters,remove_stop_words,StemTheWords,convert_negations
from algorithms.InsertText import insertText
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
        df = pd.read_csv(r"algorithms\DisneyLandReviews.csv", encoding='latin-1')
        n_samples = df['Rating'].value_counts().min()
        df_balanced = pd.concat([df[df['Rating'] == rating].sample(n=n_samples, random_state=42) for rating in range(1, 6)])
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        RatingsArray = df_balanced["Rating"]
        ReviewTextArray = df_balanced["Review_Text"]
        RatingsArray = df_balanced["Rating"]
        oldReviewTextArray = df_balanced["Review_Text"]
        ReviewTextArray = []


        for element in oldReviewTextArray:
            newWord=  element.lower()
            newWord= convert_negations(newWord)
            newWord=  remove_stop_words(newWord)
            newWord= remove_special_characters(newWord)
            substrings = ["park", "disney", "disneyland","rides","land","time","get","day","go","people","one","ride","would","kid","place","how","year","food","2","like","kids","parks","paris","see","is","i'm","me","you","were","was","have","has","disneyworld"]
            pattern = r"\b(?:{})\b".format("|".join(map(re.escape, substrings)))
            for substring in substrings:
                if substring in newWord:
                    newWord = re.sub(pattern,"",newWord)
                    
            ReviewTextArray.append(newWord)


        # Extract the features and labels
        X = ReviewTextArray
        y = RatingsArray;

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(ReviewTextArray, y, test_size=0.3, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Create a Logistic Regression classifier
        clf = LogisticRegression()

        # Train the classifier
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print('F1 score:', f1_score(y_test, y_pred, average="macro"))
        print(classification_report(y_test, y_pred))

        return vectorizer, clf

    def post(self, request):
        newValue = request.data["text"]
        newValues= insertText(newValue)
        global vectorizer, clf
        vectorizedInput = vectorizer.transform(newValues)
        predictedValue = clf.predict(vectorizedInput)
        return Response({
            "prediction": predictedValue[0]
        })
