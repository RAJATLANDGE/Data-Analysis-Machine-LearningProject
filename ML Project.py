import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\ML project\spam.csv',encoding_errors='ignore')
print(df.sample(5))

# data cleaning
# eda
# text processing
# model building
# evaluation
# improvement
# website
#deployment

print(df.shape)
#data cleaning
print(df.info())
# drop last 3 columns
df.drop(columns= ["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
print(df.sample(5))
print(df.shape)

# renaming the columns
df.rename(columns = {"v1":"target","v2":"text"},inplace= True)
print(df.sample(5))

# label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])
print(df.sample(5))

# check missing values

print("null values",df.isnull().sum())

# check for duplicate values
print("duplicated values",df.duplicated().sum())

# remove duplicates
df = df.drop_duplicates(keep = "first")
print("duplicated values",df.duplicated().sum())
print(df.shape)


#EDA

print(df["target"].value_counts())

import matplotlib.pyplot as plt
plt.pie(df["target"].value_counts(),labels=["ham","spam"],autopct="%0.2f")
# plt.show()

#data is imbalance

import nltk

# nltk.download("punkt")
# give the length of each text message
df["num_characters"] = df["text"].apply(len)
print(df.head())

# no. of words
df["num_words"] = df["text"].apply(lambda x:len(nltk.word_tokenize(x)))
print(df.head())

# no of sentences
df["num_sentences"] = df["text"].apply(lambda x:len(nltk.sent_tokenize(x)))
print(df.sample(5))

print(df[["num_words","num_characters","num_sentences"]].describe())
# ham dscribe
print(df[df["target"]==0][["num_words","num_characters","num_sentences"]].describe())
# spam describe
print(df[df["target"]==1][["num_words","num_characters","num_sentences"]].describe())

import seaborn as sns

sns.pairplot(df,hue="target")
# plt.show()

sns.histplot(df[df["target"]==0]["num_characters"])
sns.histplot(df[df["target"]==1]["num_characters"],color = "red")
# plt.show()
sns.heatmap(df.corr(),annot=True)
# plt.show()

# data preprocessing
# lower case
# tokenaization
# removing special character
# removing stopword and punctuation    (eg is, of, the) (stopward = a word which does not contribution in meaning of word
# but contribution in sentence formation)
# stemming      (eg dance dancing danced etc write as dance ) this is stemming
from nltk.corpus import stopwords
import string
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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# print(transform_text("HI how Are Rajat landge &(*&%^&* you"))
# print(transform_text("Did you like my presentation in ML"))
# print(transform_text("I Love the ML algorithm and its mathematical institution. How about you ?"))
# nltk.download('stopwords')
# print(stopwords.words("english"))  #give the list of stopwards
# print(string.punctuation)   # give the list of punctuation

df["transformed_text"]=df["text"].apply(transform_text)
print(df.sample(5))

from wordcloud import WordCloud
wc = WordCloud(width=50, height= 50, min_font_size=10,background_color="white")

spam_wc = wc.generate(df[df["target"]==1]["transformed_text"].str.cat(sep = " "))
plt.figure(figsize=(15,6))
# plt.imshow(spam_wc)
# plt.show()

ham_wc = wc.generate(df[df["target"]==0]["transformed_text"].str.cat(sep = " "))
plt.figure(figsize=(15,6))
# plt.imshow(ham_wc)
# plt.show()

# extract top words use in ham and spam
spam_corpus = []
for msg in df[df["target"]==1]["transformed_text"].tolist():
    for word in msg.split():
        spam_corpus.append(word)

# from collections import Counter
# x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0]
# y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1]
# sns.barplot(x,y)
# plt.xticks(rotation='vertical')
# plt.show()
#
ham_corpus = []
for msg in df[df["target"]==0]["transformed_text"].tolist():
    for word in msg.split():
        ham_corpus.append(word)
#
# sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[0])
# plt.xticks(rotation = "vertical")
# plt.show()


#### model building
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['transformed_text']).toarray()
print(x)
y = df["target"].values

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# appending the num_character col to X
# x = np.hstack((x,df['num_characters'].values.reshape(-1,1)))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb.fit(x_train,y_train)

y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

# tfidf --> MNB

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}


def train_classifier(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision


train_classifier(svc, x_train, y_train, x_test, y_test)

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, x_train, y_train, x_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

sns.catplot(x = 'Algorithm', y='value',hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
# plt.show()


# model improve
# 1. Change the max_features parameter of TfIdf
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
new_df_scaled.merge(temp_df,on='Algorithm')

# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(x_train,y_train)

y_pred = voting.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))



















