import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf= pickle.load(open("C:\\Users\\Lenovo\\Untitled Folder 6\\vectorizer.pkl",'rb'))
model= pickle.load(open("C:\\Users\\Lenovo\\Untitled Folder 6\\modelemail.pkl",'rb'))


st.title("Email/SMS Spam Classifier")

with st.container():
    st.write("""
    This spam email classifier project utilized machine learning techniques and Natural Language Processing (NLP) to develop an efficient system for identifying and classifying spam emails. 
    By employing algorithms such as Naive Bayes, we trained a model to analyze and process text data from email messages.
    Using NLP techniques, we extracted relevant features from the email content, including the presence of certain keywords, email headers, and patterns in the text. These features were then used to train the Naive Bayes classifier, which learned to distinguish between spam and legitimate emails based on the provided labeled data.
    
    """)

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



with st.container():
    with st.sidebar:
        members = [
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},
            
        ]

        # Define the page title and heading
        st.markdown("<h1 style='font-size:28px'>Developer</h1>", unsafe_allow_html=True)

        # Iterate over the list of members and display their details
        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")        