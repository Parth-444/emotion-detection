import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# write all files with functions
# use the data type thingie in function input and output
# exception handling as well
# logging as well

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_params(params_path: str) -> float:

    test_size = yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
    logger.debug('Test size: %s', test_size)
    return test_size

def read_data(url: str) -> pd.DataFrame:

    df = pd.read_csv(url)

    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=['tweet_id'])

    final_df = df[(df['sentiment'] == 'sadness') | (df['sentiment'] == 'happiness')]

    final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
    
    return final_df

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:

    os.makedirs(data_path)   

    train_data.to_csv(os.path.join(data_path, "train_data.csv"))

    test_data.to_csv(os.path.join(data_path, "test_data.csv"))


def main():

    test_size = load_params('params.yaml')

    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    final_df = process_data(df)
    
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    
    data_path = os.path.join('data', 'raw')

    save_data(data_path=data_path, train_data=train_data, test_data=test_data)


if __name__ == '__main__':
    
    main()





