

# import numpy as np
# import pandas as pd
# import matplotlib.pylab as plt
# import seaborn as sns
# import nltk
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, log_loss
# from nltk.corpus import stopwords
# import string
# from nltk.stem.porter import PorterStemmer
# import joblib
# from tkinter import *
# from tkinter import messagebox

# # Load dataset
# df = pd.read_csv('spam or ham data/spamhamdata.csv', sep='\t', names=['target', 'text'])
# print(df)

# # Encode labels
# label_encode = LabelEncoder()
# df['target'] = label_encode.fit_transform(df['target'])

# # Check and remove duplicates
# df.isnull().sum()
# df.duplicated().sum()
# df = df.drop_duplicates(keep='first')
# df.duplicated().sum()
# df.shape
# df['target'].value_counts()

# # Feature engineering
# df['num_character'] = df['text'].apply(len)
# print(df)

# nltk.download('punkt')
# nltk.download('stopwords')

# df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
# print(df)

# df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
# print(df.head())

# df[['num_character', 'num_words', 'num_sentence']].describe()
# df[df['target'] == 0][['num_character', 'num_words', 'num_sentence']].describe()
# df[df['target'] == 1][['num_character', 'num_words', 'num_sentence']].describe()

# # Visualization
# plt.figure(figsize=(12, 6))
# sns.histplot(df[df['target'] == 0]['num_character'])
# sns.histplot(df[df['target'] == 1]['num_character'], color='yellow')
# sns.pairplot(df, hue='target')

# # Text preprocessing function
# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = [i for i in text if i.isalnum()]

#     text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

#     text = [ps.stem(i) for i in text]

#     return " ".join(text)

# transform_text("FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv")

# df['transform_text'] = df['text'].apply(transform_text)
# print(df.head())

# # Create word corpus for spam
# spam_corpus = []
# for mes in df[df['target'] == 1]["transform_text"].tolist():
#     for word in mes.split():
#         spam_corpus.append(word)

# print(len(spam_corpus))

# from collections import Counter
# word_counts = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['Word', 'Count'])

# # Plot word frequencies
# sns.barplot(data=word_counts, x='Word', y='Count')
# plt.xticks(rotation='vertical')
# plt.show()

# # Vectorization and model training
# cv = CountVectorizer()
# x = cv.fit_transform(df['transform_text']).toarray()
# y = df['target']
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1)

# model = RandomForestClassifier()
# model.fit(xtrain, ytrain)

# # Model evaluation
# y_pred_proba = model.predict_proba(xtest)
# ypred = model.predict(xtest)
# print('accuracy_score:', accuracy_score(ytest, ypred))
# print('confusion_matrix:')
# print(confusion_matrix(ytest, ypred))
# print('precision_score:', precision_score(ytest, ypred))
# print('log_loss:', log_loss(ytest, y_pred_proba))

# # Save the model and vectorizer
# joblib.dump(model, 'spam_or_ham_model.pkl')
# joblib.dump(cv, 'count_vectorizer.pkl')



from tkinter import *
from tkinter import messagebox
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib

# Load the trained model
model = joblib.load('spam_or_ham_model.pkl')
cv = joblib.load('count_vectorizer.pkl')  # Assuming you saved CountVectorizer too

# Define prediction function
def predict_spam_ham():
    input_text = entry.get()
    if input_text.strip() == "":
        messagebox.showerror("Error", "Please enter some text.")
    else:
        ps = PorterStemmer()
        input_text_lower = input_text.lower()
        input_text_tokens = nltk.word_tokenize(input_text_lower)
        filtered_tokens = []
        for word in input_text_tokens:
            # Check if the word is alphanumeric and not in stopwords
            if word.isalnum() and word not in stopwords.words('english'):
                # Stem the word
                stemmed_word = ps.stem(word)
                # Add the stemmed word to the filtered tokens list
                filtered_tokens.append(stemmed_word)
        # Join the filtered tokens into a single string
        preprocessed_text = " ".join(filtered_tokens)
        # Transform the preprocessed text into a vector
        input_vector = cv.transform([preprocessed_text])
        # Make prediction
        prediction = model.predict(input_vector)
        print(prediction)
        # Update result label based on prediction
        if prediction[0] == 0:
            result_label.config(text="Ham Email",font="arial 30")
        else:
            result_label.config(text="Spam Email",font="arial 30")

# Create Tkinter GUI
root = Tk()
root.geometry('600x600')
root.configure(background='sky blue')
root.title("Spam or Ham Classifier")


label = Label(root, text="Email Spam Or Ham Classifier",font='bold 30',bg='skyblue')
label.pack()

label = Label(root, text="Enter Text:",font='bold 25',bg='skyblue')
label.pack()

entry = Entry(root, width=50,font='italic 25')
entry.pack()

# login_user_id_frame=LabelFrame(root,text='Enter Details',font='arial 20 italic',bd=20,relief=GROOVE,bg='lightsteelblue3')
# login_user_id_frame.pack()

predict_button = Button(root, text="Check", command=predict_spam_ham,font='bold 30',bg='black',fg='white')
predict_button.pack()

result_label = Label(root,bg='skyblue')
result_label.pack()

root.mainloop()

