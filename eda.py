import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from wordcloud import WordCloud
import feature as fe

def write_line(f, text=""):

    f.write(text + "\n") #to write in file

def plt_boxplot(data, label, feature):

    sns.boxplot(x=label, y=feature, data=data)
    plt.title("Boxplot " + feature)
    plt.savefig("Output/second_eda/" + feature + ".png", dpi=300, bbox_inches='tight')
    plt.show()

def data_to_plt(data, subject_col, message_col):

    data = data.drop_duplicates()

    data[subject_col] = data[subject_col].fillna('NO_SUBJECT').astype(str)  # replace NA to NO_SUBJECT
    data[message_col] = data[message_col].fillna('NO_MESSAGE').astype(str)  # replace NA to NO_MESSAGE

    return data

def dataset_overview(data, f):
    write_line(f, "=== DATASET OVERVIEW ===")
    write_line(f, str(data.head())) #first 5 rows
    write_line(f)

    write_line(f, "Dataset info:") #info about dataset
    data.info(buf=f)
    write_line(f)


def class_distribution(data, data_col, f):
    write_line(f, "=== CLASS DISTRIBUTION ===")
    write_line(f, str(data[data_col].value_counts())) #count of ham and spam
    write_line(f)
    write_line(f, "Class ratio:")
    write_line(f, str(data[data_col].value_counts(normalize=True))) #ratio of ham and spam
    write_line(f)


def empty_text_stats(data, subject_col, message_col, data_col, f):
    write_line(f, "=== EMPTY TEXT STATS ===")

    empty_subject = data[subject_col].isna() | (data[subject_col].astype(str).str.strip() == "") #mask of subject
    empty_message = data[message_col].isna() | (data[message_col].astype(str).str.strip() == "") #mask of message
    both_empty = empty_subject & empty_message #mask of both

    write_line(f, "Blank Subject by class:")
    write_line(f, str(data.loc[empty_subject, data_col].value_counts())) #check dependence tag with blank subject
    write_line(f)

    write_line(f, "Blank Message by class:")
    write_line(f, str(data.loc[empty_message, data_col].value_counts())) #check dependence tag with blank message
    write_line(f)

    write_line(f, f"Empty Subject: {empty_subject.sum()}")
    write_line(f, f"Empty Message: {empty_message.sum()}")
    write_line(f, f"Empty Subject & Message: {both_empty.sum()}")
    write_line(f)


def duplicate_stats(data, subject_col, message_col, data_col, f):
    write_line(f, "=== DUPLICATES ===")

    conflicts = (
        data
        .groupby([subject_col, message_col])[data_col] #danger duplicates
        .nunique()
    )

    write_line(f, f"Full duplicates: {data.duplicated().sum()}")
    write_line(f, f"Text duplicates: {data.duplicated(subset=[subject_col, message_col]).sum()}")
    write_line(f, f"Conflicting texts: {(conflicts > 1).sum()}")
    write_line(f)


def text_length_stats(data, subject_col, message_col, f):
    subject_len = data[subject_col].astype(str).str.len() #median len of subject
    message_len = data[message_col].astype(str).str.len() #median len of message

    write_line(f, "=== TEXT LENGTH STATS ===")
    write_line(f, f"Average subject length: {subject_len.mean():.2f}")
    write_line(f, f"Average message length: {message_len.mean():.2f}")
    write_line(f)

def raw_data_check(data,subject_col, message_col, data_col, f):
    dataset_overview(data, f)
    class_distribution(data, data_col, f)
    empty_text_stats(data, subject_col, message_col, data_col, f)
    duplicate_stats(data, subject_col, message_col, data_col, f)
    text_length_stats(data, subject_col, message_col, f)
    print("Report saved to output/first_eda/raw_data_check.txt")

def plt_distribution(data, subject_col, message_col, data_col):

    plt.figure(figsize=(6, 4))  # size figure
    sns.countplot(x=data_col, data=data)  # x - Spam and Ham, Data Set - raw data
    plt.title('Class Distribution')  # Title
    plt.savefig("Output/first_eda/class_distribution.png", dpi=300, bbox_inches='tight')  # save plot
    plt.show()  # show

def plt_length_stats(data, subject_col, message_col, data_col):

    data['subject_len'] = data[subject_col].str.len()  # new column
    data['message_len'] = data[message_col].str.len()  # new column

    if skew(data['subject_len']) > 0.5 or skew(data['subject_len']) < -0.5:
        data['subject_len'] = np.log1p(data['subject_len'])  # log of len

    if skew(data['message_len']) > 0.5 or skew(data['message_len']) < -0.5:
        data['message_len'] = np.log1p(data['message_len'])  # log of len

    plt.figure(figsize=(12, 5))  # size
    sns.histplot(data=data, x='subject_len', hue=data_col, bins=50, kde=True)
    plt.title('Subject Length Distribution by Class')  # title
    plt.savefig("Output/first_eda/length_subject_distribution.png", dpi=300, bbox_inches='tight')  # save
    plt.show()

    plt.figure(figsize=(12, 5))  # size
    sns.histplot(data=data, x='message_len', hue=data_col, bins=50, kde=True)
    plt.title('Message Length Distribution by Class')  # title
    plt.savefig("Output/first_eda/length_message_distribution.png", dpi=300, bbox_inches='tight')  # save
    plt.show()

def plt_spam_empty(data,subject_col, message_col, data_col):

    data['is_empty_subject'] = (data[subject_col] == 'NO_SUBJECT').astype(int) # mapping 0 or ham
    data['is_empty_message'] = (data[message_col] == 'NO_MESSAGE').astype(int) # mapping 0 or ham

    data['is_spam'] = (data[data_col] == 'spam').astype(int) # mapping 0 or ham

    plt.figure(figsize=(6,4)) #size
    sns.barplot(
    x='is_empty_subject',
    y='is_spam',
    data=data
    )
    plt.title('Probability of Spam vs Empty Subject')
    plt.savefig("Output/first_eda/spam_empty_subject.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6,4)) #size
    sns.barplot(
    x='is_empty_message',
    y='is_spam',
    data=data
    )
    plt.title('Probability of Spam vs Empty Message')
    plt.savefig("Output/first_eda/spam_empty_message.png", dpi=300, bbox_inches='tight')
    plt.show()

def plt_wordcloud(data,subject_col, message_col, text):

    data[text] = "[SUBJECT] " + data[subject_col] + " [BODY] " + data[message_col]

    text = " ".join(data[text])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    plt.savefig("Output/first_eda/wordcloud.png", dpi=300, bbox_inches='tight')
    plt.show()

def plt_first_eda(data,subject_col, message_col, data_col, text):

    data = data_to_plt(data, subject_col, message_col)
    plt_distribution(data,subject_col, message_col, data_col)
    plt_length_stats(data,subject_col, message_col, data_col)
    plt_spam_empty(data,subject_col, message_col, data_col)
    plt_wordcloud(data, subject_col, message_col, text)

    print("Plots saved to output")

def plt_tdif(data):

    tfidf, X_tfidf, y = fe.tfid_data(data)

    feature_names = tfidf.get_feature_names_out()  # name of word

    spam_idx = (y == 1).to_numpy()  # array with spam
    ham_idx = (y == 0).to_numpy()  # array with ham

    tfidf_spam_mean = X_tfidf[spam_idx].mean(axis=0).A1  # mean of spam word
    tfidf_ham_mean = X_tfidf[ham_idx].mean(axis=0).A1  # mean of ham word
    top_n = 20

    spam_top_idx = np.argsort(tfidf_spam_mean)[-top_n:]  # index
    ham_top_idx = np.argsort(tfidf_ham_mean)[-top_n:]  # index

    spam_words = feature_names[spam_top_idx]  # spam words
    ham_words = feature_names[ham_top_idx]  # ham words

    plt.figure(figsize=(10, 5))
    plt.barh(spam_words, tfidf_spam_mean[spam_top_idx])  # correlation word with score
    plt.title("Top TF-IDF words in SPAM")
    plt.xlabel("Mean TF-IDF")
    plt.tight_layout()
    plt.savefig("Output/second_eda/top_word_spam.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.barh(ham_words, tfidf_ham_mean[ham_top_idx])  # correlation word with score
    plt.title("Top TF-IDF words in HAM")
    plt.xlabel("Mean TF-IDF")
    plt.tight_layout()
    plt.savefig("Output/second_eda/top_word_ham.png", dpi=300, bbox_inches='tight')
    plt.show()

    tfidf_diff = tfidf_spam_mean - tfidf_ham_mean
    top_diff_idx = np.argsort(tfidf_diff)[-20:]
    diff_words = feature_names[top_diff_idx]

    plt.figure(figsize=(10, 5))
    plt.barh(diff_words, tfidf_diff[top_diff_idx])
    plt.title("Words with highest TF-IDF difference (Spam - Ham)")
    plt.xlabel("TF-IDF difference")
    plt.tight_layout()
    plt.savefig("Output/second_eda/diff_spam_ham.png", dpi=300, bbox_inches='tight')
    plt.show()

    non_zero_ratio = X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])
    print(f"Non-zero TF-IDF ratio: {non_zero_ratio:.4f}")

def plt_corr(data, label, feature):

    corr = data[feature + [label]].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation matrix")
    plt.savefig("Output/second_eda/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def plt_second_eda(data):

    plt_tdif(data)
    features = fe.get_structual_features(data)

    plt_corr(data, 'Label', features)

    for f in features:
        plt_boxplot(data, 'Label', f)

    print("Plots saved to output")