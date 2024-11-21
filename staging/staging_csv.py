import staging, os

# Defining a dictionary of all dataframes to be made
dataframes = {
    'training_crime': staging.get_training_crime(),
    'training_health': staging.get_training_health(),
    'training_politics': staging.get_training_politics(),
    'training_science': staging.get_training_science(),
    'training_social': staging.get_training_social(),
    'testing_crime': staging.get_testing_crime(),
    'testing_health': staging.get_testing_health(),
    'testing_politics': staging.get_testing_politics(),
    'testing_science': staging.get_testing_science(),
    'testing_social': staging.get_testing_social()
}

# Creating a 'clean data' directory if it doesn't already exist
#os.makedirs('../clean_data/', exist_ok=True)

# Creating a .csv based off of every dataframe in dataframes
#for name, df in dataframes.items():
#    df.to_csv(f"../clean_data/{name}.csv", index=False)

print(dataframes['training_health'].head())