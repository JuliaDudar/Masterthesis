import pandas as pd
import seaborn as sns
sns.set_palette(palette="viridis", n_colors=3)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


#navigate to folder with raw data
data = "C:/Users/yulya/PycharmProjects/usentiment_analysis/drugsComTrain_raw.tsv"

#define classes
class_names = ['negativ','neutral', 'positiv']
#class_names = ['negativ', 'positiv']

#define a number of samples for each class to balance the corpus
# for three classes
x = 21000
#for two classes
#x = 23000

#define tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#define max length of reviews for filtering
review_length = 250
#review_length = 150


#read data and covert data to Data Frame
def read_data(data):
    with open(data, "r", encoding="utf-8") as infile:
        df = pd.read_csv(infile, sep = "\t")
    return df


#delete duplicates
def delete_duplicates(df):
    df = df.sort_values('review', ascending=True)
    df = df.drop_duplicates(subset='review', keep='first')
    return df


#plot review-ratings
def plot_ratings(df):
    sns.countplot(df.rating)
    plt.xlabel('review score')
    plt.ylabel('number of reviews')
    return plt.show()


#divide data in three classes according to ratings
def to_sentiment(rating):
  rating = int(rating)
  if rating <= 3:
    return 0
  elif rating >= 8:
    return 2
  else:
    return 1


#divide data in two classes according to ratings
def to_sentiment_2classes(rating):
  rating = int(rating)
  if rating <= 5:
    return 0
  elif rating >= 6:
    return 1


#plot number of reviews for each class
def plot_sentiment(df):
    ax = sns.countplot(df.sentiment)
    plt.xlabel('review sentiment')
    plt.ylabel('number of reviews')
    ax.set_xticklabels(class_names)
    return plt.show()


#clean reviews
def clean_reviews(df):
    df.replace("&#039;", "'", regex=True, inplace=True)
    df.replace("&quot;", "", regex=True, inplace=True)
    df.replace("&lt;", "<", regex=True, inplace=True)
    df.replace("&gt;", ">", regex=True, inplace=True)
    df.replace("&ndash;", "—", regex=True, inplace=True)
    df.replace("&lsquo;", "—", regex=True, inplace=True)
    df.replace("&rsquo;", "", regex=True, inplace=True)
    df.replace("&sbquo;", "", regex=True, inplace=True)
    df.replace("&ldquo;", "", regex=True, inplace=True)
    df.replace("&rdquo;", "", regex=True, inplace=True)
    df.replace("&bdquo;", "", regex=True, inplace=True)
    df.replace("&euro;", "€", regex=True, inplace=True)
    df.replace("\n", "", regex=True, inplace=True)
    #delete usefulCount column
    df.drop('usefulCount', axis=1, inplace=True)
    #set name for first column
    df.index.set_names("ID", inplace=True)
    return df


#balance a corpus
def sampling_x_elements(group, x = x):
    if len(group) < x:
        return group
    return group.sample(x)


#tokenize reviews and count the number of tokens, add review-length column
def count_token(df, tokenizer):
    token_length = []
    for txt in df.review:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_length.append(len(tokens))
    df['review_length'] = token_length
    return df


def main(data, tokenizer, review_length):
    df = read_data(data)
    df = delete_duplicates(df)
    plot = plot_ratings(df)
    df['sentiment'] = df.rating.apply(to_sentiment)
    print(df.head(20))
    # df['sentiment'] = df.rating.apply(to_sentiment_2classes)
    sentiment_plot = plot_sentiment(df)
    df = clean_reviews(df)
    df = count_token(df, tokenizer)
    #filter out reviews that are longer that threshold
    filtered_df = df[df["review_length"] < review_length]
    #prepear a small version of dataset
    # df_train, df_test = train_test_split(filtered_df, test_size=0.5, random_state=17, stratify = filtered_df.sentiment.values)
    # print(df_train.shape)
    #balance the corpus
    balanced = pd.DataFrame(filtered_df.groupby('sentiment').apply(sampling_x_elements).reset_index(drop =True))
    # sort values
    balanced = balanced.sort_values('review', ascending=True)
    balanced.drop(balanced.columns[[0]], axis=1, inplace=True)
    balanced.index.set_names("ID", inplace=True)
    # print(balanced.sentiment.value_counts())
    #save results as csv
    with open("balanced_test.csv", "w", encoding="utf-8") as outfile:
        balanced.to_csv(outfile, sep=",")



main(data, tokenizer, review_length)