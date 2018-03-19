from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk

text1 = "Natural language processing (NLP) is a field of computer science, artificial intelligence and computational " \
        "linguistics concerned with the interactions between computers and human (natural) languages, and, " \
        "in particular, concerned with programming computers to fruitfully process large natural language corpora. " \
        "Challenges in natural language processing frequently involve natural language understanding, " \
        "natural language generation (frequently from formal, machine-readable logical forms), connecting language " \
        "and machine perception, managing human-computer dialog systems, or some combination thereof. "

text2 = "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences " \
        "into English. The authors claimed that within three or five years, machine translation would be a solved " \
        "problem.[2] However, real progress was much slower, and after the ALPAC report in 1966, which found that " \
        "ten-year-long research had failed to fulfill the expectations, funding for machine translation was " \
        "dramatically reduced. Little further research in machine translation was conducted until the late 1980s, " \
        "when the first statistical machine translation systems were developed. "

text3 = "During the 1970s, many programmers began to write conceptual ontologies, which structured real-world " \
        "information into computer-understandable data. Examples are MARGIE (Schank, 1975), SAM (Cullingford, 1978), " \
        "PAM (Wilensky, 1978), TaleSpin (Meehan, 1976), QUALM (Lehnert, 1977), Politics (Carbonell, 1979), " \
        "and Plot Units (Lehnert 1981). During this time, many chatterbots were written including PARRY, Racter, " \
        "and Jabberwacky。 "


def get_tokens(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def train_SVD():
    # Read documents
    docs2 = [text1, text2]
    docs3 = [text1, text2, text3]

    cv2 = CountVectorizer()
    X2 = cv2.fit_transform(docs2)

    cv3 = CountVectorizer()
    X3 = cv3.fit_transform(docs3)

    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    svd.fit(X2)
    Y2 = svd.transform(X2)

    svd.fit(X3)
    Y3 = svd.transform(X3)
    print(Y2)
    print(Y3)

    # for word in cv.get_feature_names():
    #     print('{},{}: {}'.format(word, cv.vocabulary_[word], Y[cv.vocabulary_[word],:]))

train_SVD()

