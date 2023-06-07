from rest_framework.response import Response
from rest_framework import generics
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pandas as pd
from algorithms.TextNormalization import remove_special_characters,remove_stop_words,StemTheWords,convert_negations
from algorithms.InsertText  import insertText

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
        df = pd.read_csv(r"algorithms\DisneyLandReviews.csv", encoding='latin-1')

        n_samples = df['Rating'].value_counts().min()

        # Subset the DataFrame with equal number of samples for each rating
        df_balanced = pd.concat([df[df['Rating'] == rating].sample(n=n_samples, random_state=42) for rating in range(1, 6)])

        # Shuffle the rows
        df_balanced = df_balanced.sample(frac=1, random_state=42)
        numberOfMin = df_balanced[df['Rating'] == 5]

        # Get the number of rows in the filtered DataFrame
        num_rows = len(numberOfMin)
        print(numberOfMin)
        RatingsArray = df_balanced["Rating"]
        ReviewTextArray = df_balanced["Review_Text"]
        #==
        # features = df['Review_Text'].values
        #=== 
        # print(tokenize("It was nice they were very nice "))
        RatingsArray = df_balanced["Rating"]
        oldReviewTextArray=df_balanced["Review_Text"]
        ReviewTextArray=[]


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



        # ReviewTextArray = [preprocess_text(text) for text in ReviewTextArray]

        # Vectorize the text using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(ReviewTextArray)

        # Split the data into train and test sets
        X_train, X_test, RatingsArray_train, RatingsArray_test = train_test_split(X, RatingsArray, test_size=0.2, random_state=42)

        # Perform oversampling using SMOTE for class balancing
        oversampler = SMOTE(random_state=42)
        X_train, RatingsArray_train = oversampler.fit_resample(X_train, RatingsArray_train)

        # Train the SVM classifier
        clf = SVC()
        clf.fit(X_train, RatingsArray_train)

        # Predict ratings for the test set
        predicted_ratings = clf.predict(X_test)
        accuracy = accuracy_score(RatingsArray_test, predicted_ratings)
        report = classification_report(RatingsArray_test, predicted_ratings)
        print(report)
        return clf, vectorizer

    def post(self, request):
        newValue = request.data["text"]
        newValues= insertText(newValue)
        global vectorizer, clf
        vectorizedInput = vectorizer.transform(newValues)
        predictedValue = clf.predict(vectorizedInput)
        return Response({
            "prediction": predictedValue[0]
        })
