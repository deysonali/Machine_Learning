"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""
from sklearn.model_selection import train_test_split
import random
import pandas as pd
seed = random.seed(1)
data = pd.read_csv('data\\data.tsv', sep='\t')

print("Shape of data:", data.shape)
print("First few lines: ", data.head())

#Splitting into train, validation, and test. training = validation + train.

training, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['label'])
train, validation = train_test_split(training, test_size=0.2, random_state=seed, stratify=training['label'])
unnecessary_stuff, overfit = train_test_split(train, test_size=50, random_state=seed, stratify=train['label'])
#Checking that there are equal numbers of 0 and 1 labelled data in each set

print("Value counts for each label in train set:")
print(train['label'].value_counts())

print("Value counts for each label in valid set:")
print(validation['label'].value_counts())

print("Value counts for each label in test set:")
print(test['label'].value_counts())

print("Value counts for each label in overfit set:")
print(overfit['label'].value_counts())

#Saving the datasets to .tsv files

train.to_csv('data\\train.tsv', sep='\t', index=False)
validation.to_csv('data\\validation.tsv', sep='\t', index=False)
test.to_csv('data\\test.tsv', sep='\t', index=False)
overfit.to_csv('data\\overfit.tsv', sep='\t', index=False)
