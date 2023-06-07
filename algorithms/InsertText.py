from algorithms.TextNormalization import tokenize,remove_special_characters,remove_stop_words,StemTheWords,convert_negations
import re
def insertText(text):
    newValues = []
    newValue=text.lower()    
    newValue=convert_negations(newValue)
    newWord=  remove_stop_words(newValue)
    newWord= remove_special_characters(newValue)
    substrings = ["park", "disney", "disneyland","rides","land","time","get","day","go","people","one","ride","would","kid","place","how","year","food","2","like","kids","parks","paris","see","is","i'm","me","you","were","was","have","has","disneyworld"]
    pattern = r"\b(?:{})\b".format("|".join(map(re.escape, substrings)))
    for substring in substrings:
        if substring in newWord:
            newWord = re.sub(pattern,"",newWord)
    print(newWord)
    newValues.append(newValue)
    return newValues
    