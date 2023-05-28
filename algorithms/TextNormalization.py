from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# nltk.download('averaged_perceptron_tagger')

# from nltk.tag import pos_tag
import re
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def tokenize(string):
    return re.compile('\w+').findall(string.lower())

def remove_special_characters(text):
    # Define the pattern for special characters and double quotes
    pattern = r'[^a-zA-Z0-9\s"]'
    
    # Remove special characters and double quotes using regular expression
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text
def remove_stop_words(text):
    # Download the stop words corpus if not already present
    # nltk.download('stopwords')

    # Get the list of English stop words
    # stop_words = set(stopwords.words('english'))

    # Tokenize the text into individual words
    words = tokenize(text)

    # Remove stop words from the list of words
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Reconstruct the text from the filtered words
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text


def StemTheWords(sentence):
    stemmer = PorterStemmer()
    # Tokenize the sentence into words  
    tokens = tokenize(sentence)

    # Perform stemming on each word
    stemmed_words = [stemmer.stem(word) for word in tokens]

    # Join the stemmed words back into a sentence
    stemmed_sentence = " ".join(stemmed_words)
    
    return stemmed_sentence
def convert_negations(text):
    # Define the mapping of negation phrases to negative representations
    negation_map = {
    "not good": "bad",
    "not bad": "decent",
    "wasn't the best": "average",
    "not the best": "decent",
    "it could've been better": "decent",
    "not as good": "decent",
    "wasn't good": "bad",
    "wasn't perfect": "good",
    "won't return": "bad",
    "won't come back": "awful",
    "didn't like": "bad",
    "not recommend": "bad",
    "wasn't awful": "decent",
    "wasn't very good": "decent",
    "wasn't bad":"decent",
    "wasn't super bad":"average"
    
    # Add more negation phrases and their corresponding negative representations here
}

    # Iterate over the negation phrases in the mapping and replace them in the text
    for negation, negative_rep in negation_map.items():
        # Escape any special characters in the negation phrase for regex matching
        negation_pattern = re.escape(negation)

        # Define the regular expression pattern to match the negation phrase
        pattern = r"\b" + negation_pattern + r"\b"

        # Replace the negation phrase with the negative representation in the text
        text = re.sub(pattern, negative_rep, text)
        

    return text

# def seperate_adjective(text):
    
#     tokens = tokenize(text)

#     # Perform POS tagging
#     tagged_tokens = pos_tag(tokens)

#     # Extract adjectives
#     adjectives = [token for token, tag in tagged_tokens if tag.startswith('JJ')]

#     # Create a new string with only adjectives
#     adjective_string = ' '.join(adjectives)
#     return adjective_string


# Text normalization
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\b(?:park|disney|disneyland|rides|land|time|get|day|go|people|one|ride|would|kid|place|how|year|food|2|like|kids|parks|paris|see|is|i'm|me|you|were|was|have|has|disneyworld)\b", "", text)
    return text




