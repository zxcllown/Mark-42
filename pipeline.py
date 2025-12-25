import feature
import pandas as pd
import os

def drop_useless_column(data, useless_column=None):

    if useless_column is None:
        return data
    else :
        data = data.drop(columns=[useless_column])#delete a useless column

    return data

def replace_blank(data, subject_col=None, message_col=None):
    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        empty_message = data[message_col].isna() | (data[message_col].astype(str).str.strip() == "")  # mask of message
        data.loc[empty_message, message_col] = "NO_MESSAGE"  # replace blank message
    elif message_col is None:
        empty_subject = data[subject_col].isna() | (data[subject_col].astype(str).str.strip() == "")  # mask of subject
        data.loc[empty_subject, subject_col] = "NO_SUBJECT"  # replace blank subject
    else:
        empty_subject = data[subject_col].isna() | (data[subject_col].astype(str).str.strip() == "") #mask of subject
        empty_message = data[message_col].isna() | (data[message_col].astype(str).str.strip() == "") #mask of message
        data.loc[empty_message, message_col] = "NO_MESSAGE" #replace blank message
        data.loc[empty_subject, subject_col] = "NO_SUBJECT" #replace blank subject

    return data

def clear_text(data, subject_col=None, message_col=None):

    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        data[message_col] = data[message_col].str.lower()  # lowercase
        data[message_col] = data[message_col].str.replace(r'\s+', ' ', regex=True).str.strip()  # delete redundant space and lines
    elif message_col is None:
        data[subject_col] = data[subject_col].str.lower()  # lowercase
        data[subject_col] = data[subject_col].str.replace(r'\s+', ' ', regex=True).str.strip()  # delete redundant space and lines
    else:
        data['Text'] = "[SUBJECT] " + data[subject_col] + " [BODY] " + data[message_col] #union spam column to ham
        data = data.drop(columns=[subject_col, message_col]) #delete useless columns
        data['Text'] = data['Text'].str.lower() #lowercase
        data['Text'] = data['Text'].str.replace(r'\s+', ' ', regex=True).str.strip() #delete redundant space and lines

    return data

def mapping_label(data, column):
    if column is None:
        return data
    else:
        data = data[
            data[column].notna() &
            (data[column].astype(str).str.strip() != "") #save data only with not null Ham/Spam
        ]
        data[column] = (
            data[column]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        data['Label'] = data[column].map({'ham': 0, 'spam': 1}) #mapping ham/spam to 0/ham
        data = data.drop(columns=[column]) #drop column

        return data

def drop_duplicate(data, text=None):

    data = data.drop_duplicates() #drop full duplicates
    if text is None:
        return data
    else:
        data = data.drop_duplicates(subset=[text]) #drop duplicate in text column

    return data

def dropna_label(data):

    data = data.dropna(subset=['Label'])
    return data

def save_new_data(data, subject_col=None, message_col=None, column=None, useless_column=None, PATH=None):
    data = drop_useless_column(data, useless_column)
    data = feature.add_structural_features(data, subject_col, message_col)
    data = replace_blank(data, subject_col, message_col)
    data = clear_text(data, subject_col, message_col)
    data = mapping_label(data, column)
    data = drop_duplicate(data)
    data = dropna_label(data)
    data.to_csv(PATH, index=False, encoding="utf-8")
    print("Report saved to " + PATH)

    return data

def load_email_folder_dataset(base_path):
    texts = []
    labels = []

    for label_name in ['ham', 'spam']:
        label = 0 if label_name == 'ham' else 1
        folder_path = os.path.join(base_path, label_name)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith('.eml'):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
            except Exception:
                continue

            if text.strip() == "":
                continue

            texts.append(text)
            labels.append(label)

    df = pd.DataFrame({
        'Text': texts,
        'Label': labels
    })

    return df