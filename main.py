import pandas as pd
import eda
import pipeline as pp
import test

data = pd.read_csv("Data/enron_spam_data.csv")
data_email = pp.load_email_folder_dataset('Data/email-dataset-main/dataset')
with open("Output/first_eda/raw_data_check.txt", "w", encoding="utf-8") as f:
    eda.raw_data_check(data, 'Subject', 'Message', 'Spam/Ham', f)

eda.plt_first_eda(data, 'Subject', 'Message', 'Spam/Ham', 'Text')

data = pp.save_new_data(data, 'Subject', 'Message',
                         'Spam/Ham', 'Message ID', "Output/new_data.csv")
data_email = pp.save_new_data(data_email, message_col='Text', PATH="Output/new_data_email.csv")

eda.plt_second_eda(data)

test.data_test(data)
test.data_vs_data(data, data_email)