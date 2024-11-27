import pandas as pd
from datasets import load_dataset


# Removes any missing / improper values in the rows
def stage_df(df, cols):
    df = df.dropna(subset=cols)
    for col in cols: df = df[df[col] != ""]
    df = df[df['is_true'].isin([0, 1])]
    return df


def get_combined_isot():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../raw_data/ISOT True.csv")[['title', 'text', 'subject']]
    df2 = pd.read_csv("../raw_data/ISOT Fake.csv")[['title', 'text', 'subject']]
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    return df


# Returns a cleaned dataset sourced from the FA-KES dataset.
def get_training_crime():
    # Reading the dataset and only extracting the useful columns
    df = pd.read_csv('../raw_data/training/FA-KES.csv', encoding='latin1')
    df = df[['article_title', 'article_content', 'labels']]
    # Renaming columns
    df = df.rename(columns={'labels': 'is_true','article_title': 'title', 'article_content': 'text'})
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the COVID-FN and COVID-FNIR datasets.
def get_training_health():
    # Reading the datasets and only extracting the useful columns
    df1 = pd.read_csv("../raw_data/training/COVID-FNIR-True.csv")[['Text', 'Label']]  # Covid-FNIR
    df2 = pd.read_csv("../raw_data/training/COVID-FNIR-Fake.csv")[['Text', 'Binary Label']] #Covid-FNIR
    df3 = pd.read_csv("../raw_data/training/COVID-FN.csv")[['text', 'label']]
    # Renaming columns
    df1 = df1.rename(columns={'Text': 'text', 'Label': 'is_true'})
    df2 = df2.rename(columns={'Text': 'text', 'Binary Label': 'is_true'})
    df3 = df3.rename(columns={'label': 'is_true'})
    # Combining the dataframes
    df = pd.concat([df1, df2, df3], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the FakeNews, ISOT, and LIAR datasets.
def get_training_politics():
    # Reading the datasets and only extracting the useful columns
    df = get_combined_isot() #ISOT
    df1 = pd.read_csv("../raw_data/training/FakeNews.csv")[['title', 'text', 'label']] #FakeNews
    df2 = load_dataset("ucsbnlp/liar", split="train").to_pandas()[['statement', 'label']] #LIAR
    # Renaming columns
    df1 = df1.rename(columns={'label': 'is_true'})
    df2 = df2.rename(columns={'statement': 'text', 'label': 'is_true'})
    # Filtering df entries to only include Politics articles, and dropping the subject column
    df = df[df['subject'].str.contains('politics', case=False, na=False)]
    df = df.drop(columns=['subject'])
    # Fixing df2's is_true column
    df2['is_true'] = df2['is_true'].replace({0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0})
    # Combining the dataframes (ISOT + FakeNews)
    df = pd.concat([df, df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the Climate dataset.
def get_training_science():
    # Reading the dataset and only extracting the useful columns
    df = pd.read_parquet("hf://datasets/tdiggelm/climate_fever/data/test-00000-of-00001.parquet")
    df = df[['claim', 'claim_label']]
    # Renaming columns
    df = df.rename(columns={'claim_label': 'is_true', 'claim': 'text'})
    # Fixing the is_true column
    df['is_true'] = df['is_true'].replace({0: 1, 1: 0, 2: 0, 3: 0})
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the GossipCop dataset.
def get_training_social():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../raw_data/training/GossipCop_True.csv")[['title']]
    df2 = pd.read_csv("../raw_data/training/GossipCop_Fake.csv")[['title']]
    # Renaming 'title' to 'text'
    df1 = df1.rename(columns={'title': 'text'})
    df2 = df2.rename(columns={'title': 'text'})
    # Adding an 'is_true' column
    df1['is_true'] = 1
    df2['is_true'] = 0
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the Snopes dataset.
def get_testing_crime():
    # Reading the dataset and only extracting the useful columns
    df = pd.read_csv("../raw_data/testing/Snopes.csv")
    df = df[['fact_rating_phase1', 'article_claim_phase1', 'article_category_phase1']]
    # Renaming columns
    df = df.rename(columns={'fact_rating_phase1': 'is_true', 'article_claim_phase1': 'text',
                            'article_category_phase1': 'category'})
    # Filtering entries to only include Crime articles, and dropping the category column
    df = df[df['category'].str.contains('crime', case=False, na=False)]
    df = df.drop(columns=['category'])
    # Fixing the is_true column
    true_values = ["correct attribution", "mostly true", "TRUE"]
    false_values = ["legend", "misattributed", "miscaptioned", "mixture", "mostly false", "outdated", "scam",
                    "unproven", "FALSE"]
    df['is_true'] = df['is_true'].replace(true_values, 1)
    df['is_true'] = df['is_true'].replace(false_values, 0)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the COVID_claims dataset.
def get_testing_health():
    # Reading the two datasets and only extracting the useful columns
    df1 = pd.read_csv("../raw_data/testing/COVID_Claims_True.csv")[['Text', 'Label']]
    df2 = pd.read_csv("../raw_data/testing/COVID_Claims_Fake.csv")[['Text', 'Binary Label']]
    # Renaming df1 and df2's columns
    df1 = df1.rename(columns={'Text': 'text', 'Label': 'is_true'})
    df2 = df2.rename(columns={'Text': 'text', 'Binary Label': 'is_true'})
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the PHEME and PolitiFact datasets.
def get_testing_politics():
    # Reading the dataset and only extracting the useful columns
    df2 = pd.read_csv("../raw_data/testing/Politifact.csv")
    df1 = pd.read_csv("../raw_data/testing/Pheme_output.csv")[['text', 'target', 'event']]
    df2 = df2[['page_is_first_citation_phase1', 'article_title_phase1', 'article_categories_phase1']]
    # Renaming columns
    df2 = df2.rename(columns={'page_is_first_citation_phase1': 'is_true', 'article_title_phase1': 'text'})
    df1 = df1.rename(columns={'target': 'is_true'})
    # Creating a regex pattern to match any of the desired events
    pattern = ('charliehebdo-all-rnr-threads|ottawashooting-all-rnr-threads|ferguson-all-rnr-threads|'
               'sydneysiege-all-rnr-threads|russian-ukrainian-crisis-all-rnr-threads|germanwings-crash-all-rnr-threads')
    # Filtering entries to only include relevant articles
    df1 = df1[df1['event'].str.contains(pattern, case=False, na=False)]
    df1 = df1.drop(columns=['event'])
    # Combining the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the ISOT dataset.
def get_testing_science():
    df = get_combined_isot()
    # Filtering rows by science keywords (source: https://www.enchantedlearning.com/wordlist/science.shtml)
    science_terms = [
        "astronomy", "astrophysics", "atom", "beaker", "biochemistry", "biology", "botany", "Bunsen burner",
        "burette", "chemical", "chemistry", "climate", "climatologist", "control", "cuvette", "data", "datum",
        "electricity", "electrochemist", "element", "energy", "entomology", "evolution", "experiment", "flask",
        "fossil", "funnel", "genetics", "geology", "geophysics", "glassware", "graduated cylinder", "gravity",
        "herpetology", "hypothesis", "ichthyology", "immunology", "lab", "laboratory", "lepidoptery",
        "magnetism", "mass", "measure", "meteorologist", "meteorology", "microbiologist", "microbiology",
        "microscope", "mineral", "mineralogy", "molecule", "observatory", "observe", "organism", "ornithology",
        "paleontology", "particle", "Petri dish", "physical science", "physics", "pipette", "quantum mechanics",
        "radiology", "research", "retort", "science", "scientific method", "scientist", "seismology", "telescope",
        "temperature", "test tube", "thermometer", "variable", "virologist", "volcanology", "volume",
        "volumetric flask", "watch glass", "weather", "zoology"
    ]
    df = df[df['title'].str.contains('|'.join(science_terms), case=False, na=False) |
            df['text'].str.contains('|'.join(science_terms), case=False, na=False)]
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df


# Returns a cleaned dataset sourced from the ISOT dataset.
def get_testing_social():
    df = get_combined_isot()
    # Filtering rows by social keywords (source: ChatGPT)
    social_terms = [
        "social media", "Facebook", "Twitter", "Instagram", "LinkedIn", "Snapchat", "YouTube", "TikTok",
        "Pinterest", "Reddit", "hashtag", "influencer", "vlog", "blog", "feed", "timeline", "profile", "post",
        "share", "like", "comment", "reaction", "story", "DM", "direct message", "engagement", "followers",
        "trending", "viral", "content", "analytics", "reach", "impressions", "notification",
        "user-generated content", "platform", "community", "network", "branding", "advertising", "campaign",
        "live stream", "memes", "photography", "video", "GIF", "emoticons", "polls", "community guidelines",
        "privacy", "terms of service", "blocking", "reporting", "celebrity"
    ]
    df = df[df['title'].str.contains('|'.join(social_terms), case=False, na=False) |
            df['text'].str.contains('|'.join(social_terms), case=False, na=False)]
    # Ensuring there aren't any missing / improper values for the rows
    df = stage_df(df, df.columns.tolist())
    return df