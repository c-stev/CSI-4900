import pandas as pd

# Returns a cleaned dataset sourced from the FA-KES dataset.
# NOTE: 0 is false, 1 is credible, source: https://zenodo.org/records/2607278
def get_training_crime():
    # Reading the dataset and only extracting the useful columns
    df = pd.read_csv("../data/training/FA-KES.csv", encoding='latin1')[
        ['article_title', 'article_content', 'labels']
    ]
    # Renaming columns
    df = df.rename(columns={'labels': 'is_true','article_title': 'title', 'article_content': 'text'})
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    df = df[df['is_true'].isin([0, 1])]
    return df

# Returns a cleaned dataset sourced from the COVID-FN and COVID-FNIR datasets.
def get_training_health():
    df1 = pd.read_csv("../data/COVID19-FNIR/fakeNews.csv")[['Link', 'text', 'Binary Label']] #Covid-FNIR
    df2 = pd.read_csv("../data/COVID19-FNIR/trueNews.csv")[['Link', 'text', 'Binary Label']] #Covid-FNIR
    df = pd.concat([df1, df2], ignore_index=True)
    return df

# Returns a cleaned dataset sourced from the FakeNews, ISOT, and LIAR datasets.
def get_training_politics():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../data/ISOT True.csv")[['title', 'text', 'subject']] #ISOT
    df2 = pd.read_csv("../data/ISOT Fake.csv")[['title', 'text', 'subject']] #ISOT
    df3 = pd.read_csv("../data/training/FakeNews/train.csv")[['title', 'text', 'label']] #FakeNews
    # TODO: LIAR dataset
    df4 = pd.read_json("../data/liar/train/dataset_info.json")[['label', 'statement', 'subject']] #LIAR
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes (ISOT True and ISOT Fake)
    df = pd.concat([df1, df2], ignore_index=True)
    # Renaming columns
    df = df.rename(columns={'subject': 'category'})
    df3 = df3.rename(columns={'label': 'is_true'})
    #df4 = df4.rename(columns={'': ''})
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    # Filtering entries to only include Politics articles, and dropping the category column
    df = df[df['category'].str.contains('politics', case=False, na=False)]
    df = df.drop(columns=['category'])
    # Combing the dataframes (ISOT + FakeNews)
    df = pd.concat([df, df3], ignore_index=True)
    # TODO: LIAR dataset
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    df = df[df['is_true'].isin([0, 1])]
    return df

# TODO: Climate is a .parquet file, which has had trouble interacting with pandas. Might need a new download?
# Returns a cleaned dataset sourced from the Climate dataset.
def get_training_science():
    df = pd.read_parquet("../data/training/Climate.parquet") #Climate data
    return df

# TODO: GossipCop dataset only has article titles, no text
# Returns a cleaned dataset sourced from the GossipCop dataset.
def get_training_social():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../data/training/GossipCop Real.csv")[['news_url', 'title']]
    df2 = pd.read_csv("../data/training/GossipCop Fake.csv")[['news_url', 'title']]
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # TODO: Missing value checks; must wait until we get the text
    return df

# TODO: Snopes dataset only has article titles, no text
# Returns a cleaned dataset sourced from the Snopes dataset.
def get_testing_crime():
    # Reading the dataset and only extracting the useful columns
    df = pd.read_csv("../data/training/snopes_phase1_clean_2018_7_3.csv")[
        ['fact_rating_phase1', 'article_origin_url_phase1', 'article_claim_phase1', 'article_category_phase1']
    ]
    # Renaming columns
    df = df.rename(columns={
        'fact_rating_phase1': 'is_true', 'article_origin_url_phase1': 'source',
        'article_claim_phase1': 'title', 'article_category_phase1': 'category'
                            }
    )
    # Filtering entries to only include Crime articles, and dropping the category column
    df = df[df['category'].str.contains('crime', case=False, na=False)]
    df = df.drop(columns=['category'])
    # Since snopes doesn't use binary ratings for an article's veracity, we must replace them
    # Category names were done manually by going through the original Excel file
    true_values = ["correct attribution", "mostly true", "TRUE"]
    false_values = ["legend", "misattributed", "miscaptioned", "mixture", "mostly false", "outdated", "scam",
                    "unproven", "FALSE"]
    df['is_true'] = df['is_true'].replace(true_values, 1)
    df['is_true'] = df['is_true'].replace(false_values, 0)
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'source', 'category'])
    df = df[(df['title'] != "") & (df['source'] != "") & (df['is_true'] != "")]
    df = df[df['is_true'].isin([0, 1])]
    # TODO: Missing value checks; must wait until we get the text
    return df

# Returns a cleaned dataset sourced from the COVID_claims dataset.
def get_testing_health():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../data/testing/COVID Claims True.csv")[['Text', 'Label']]
    df2 = pd.read_csv("../data/testing/COVID Claims False.csv")[['Text', 'Binary Label']]
    # Renaming df1 and df2's columns
    df1 = df1.rename(columns={'Text': 'text', 'Label': 'is_true'})
    df2 = df2.rename(columns={'Text': 'text', 'Binary Label': 'is_true'})
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    return df

# Returns a cleaned dataset sourced from the PHEME and PolitiFact datasets.
def get_testing_politics():
    df = pd.DataFrame()
    return df

# TODO: ISOT 'subject' doesn't have a science section
# Returns a cleaned dataset sourced from the ISOT dataset.
def get_testing_science():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../data/ISOT True.csv")[['title', 'text', 'subject']]
    df2 = pd.read_csv("../data/ISOT Fake.csv")[['title', 'text', 'subject']]
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    # TODO: Sort by Science News
    return df

# TODO: ISOT 'subject' doesn't have a social section
# Returns a cleaned dataset sourced from the ISOT dataset.
def get_testing_social():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../data/ISOT True.csv")[['title', 'text', 'subject']]
    df2 = pd.read_csv("../data/ISOT Fake.csv")[['title', 'text', 'subject']]
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = df.dropna(subset=['title', 'text', 'is_true'])
    df = df[(df['title'] != "") & (df['text'] != "") & (df['is_true'] != "")]
    # TODO: Sort by Social News
    return df