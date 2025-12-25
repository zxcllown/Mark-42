from sklearn.feature_extraction.text import TfidfVectorizer

def tfid_two_data(data1, data2):
    tfidf = TfidfVectorizer(
        max_features=5000,  # top 5000 words
        stop_words='english'  # leave the, is , and
    )

    X_tfidf1 = tfidf.fit_transform(data1['Text'])  # fit is learn and transform word become int
    y1 = data1['Label']  # label 0 or ham

    X_tfidf2 = tfidf.transform(data2['Text'])  # fit is learn and transform word become int
    y2 = data2['Label']  # label 0 or ham

    return tfidf, X_tfidf1, y1, X_tfidf2, y2

def tfid_data(data):

    tfidf = TfidfVectorizer(
        max_features=5000,  # top 5000 words
        stop_words='english'  # leave the, is , and
    )

    X_tfidf = tfidf.fit_transform(data['Text'])  # fit is learn and transform word become int
    y = data['Label']  # label 0 or ham
    return tfidf, X_tfidf, y

def empty_tag(data, subject_col=None, message_col=None):

    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        empty_message = data[message_col].isna() | (data[message_col].astype(str).str.strip() == "")  # mask of message
        data['is_empty_message'] = empty_message.astype(int)  # mapping if no has message than ham
    elif message_col is None:
        empty_subject = data[subject_col].isna() | (data[subject_col].astype(str).str.strip() == "")  # mask of subject
        data['is_empty_subject'] = empty_subject.astype(int)  # mapping if no has subject than ham
    else:
        empty_subject = data[subject_col].isna() | (data[subject_col].astype(str).str.strip() == "")  # mask of subject
        empty_message = data[message_col].isna() | (data[message_col].astype(str).str.strip() == "")  # mask of message

        data['is_empty_subject'] = empty_subject.astype(int)  # mapping if no has subject than ham
        data['is_empty_message'] = empty_message.astype(int)  # mapping if no has message than ham

    return data

def len_tag(data, subject_col=None, message_col=None):

    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        data[message_col] = data[message_col].fillna('').astype(str)  # fill Na to '' and type str
        data['message_len'] = data[message_col].str.len()  # length of message
    elif message_col is None:
        data[subject_col] = data[subject_col].fillna('').astype(str)  # fill Na to '' and type str
        data['subject_len'] = data[subject_col].str.len()
    else:
        data[subject_col] = data[subject_col].fillna('').astype(str)  # fill Na to '' and type str
        data[message_col] = data[message_col].fillna('').astype(str)  # fill Na to '' and type str

        data['subject_len'] = data[subject_col].str.len()  # length of subject
        data['message_len'] = data[message_col].str.len()  # length of message

    return data

def caps_tag(data, subject_col=None, message_col=None):

    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        data['message_caps_ratio'] = data[message_col].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            # 0 if len 0, else each upper letter counting
        )
    elif message_col is None:
        data['subject_caps_ratio'] = data[subject_col].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            # 0 if len 0, else each upper letter counting
        )
    else:
        data['message_caps_ratio'] = data[message_col].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            # 0 if len 0, else each upper letter counting
        )
        data['subject_caps_ratio'] = data[subject_col].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            # 0 if len 0, else each upper letter counting
        )
    return data

def digit_tag(data, subject_col=None, message_col=None):
    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        data['message_digit_ratio'] = (data[message_col].str.count(r'\d') /
                                       data['message_len'].replace(0, 1).astype(
                                           float))  # count of digits divide by ham or len
    elif message_col is None:
        data['subject_digit_ratio'] = (data[subject_col].str.count(r'\d') /
                                       data['subject_len'].replace(0, 1).astype(
                                           float))  # count of digits divide by ham or len
    else:
        data['message_digit_ratio'] = (data[message_col].str.count(r'\d') /
                                       data['message_len'].replace(0, 1).astype(
                                           float))  # count of digits divide by ham or len
        data['subject_digit_ratio'] = (data[subject_col].str.count(r'\d') /
                                       data['subject_len'].replace(0, 1).astype(
                                           float))  # count of digits divide by ham or len
    return data

def special_tag(data, subject_col=None, message_col=None):

    special_pattern = r'[!$%#@]'

    if subject_col is None and message_col is None:
        return data
    elif subject_col is None:
        data['message_special_count'] = data[message_col].str.count(special_pattern)  # count of special pattern
    elif message_col is None:
        data['subject_special_count'] = data[subject_col].str.count(special_pattern)  # count of special pattern
    else:
        data['message_special_count'] = data[message_col].str.count(special_pattern)  # count of special pattern
        data['subject_special_count'] = data[subject_col].str.count(special_pattern)  # count of special pattern

    return data


def add_structural_features(data, subject_col, message_col):

    data = empty_tag(data, subject_col, message_col)
    data = len_tag(data, subject_col, message_col)
    data = caps_tag(data, subject_col, message_col)
    data = digit_tag(data, subject_col, message_col)
    data = special_tag(data, subject_col, message_col)

    return data

def get_structual_features(data):

    features = [col for col in data.columns if col not in ['Label', 'Text', 'Date']]
    return features