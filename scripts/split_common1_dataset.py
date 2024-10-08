import pandas as pd 
import os, shutil

BASE_DIR = 'dataset/Common1/'
CLIPS_DIR = BASE_DIR+'clips/'
TRAIN_FILE = BASE_DIR+'train.tsv'
TEST_FILE = BASE_DIR+'test.tsv'

def iter_and_split(df: pd.DataFrame, name: str) -> None :

    for _, row in df.iterrows():
        current_file = row['path']+'.mp3'
        gender = '_'+row['gender']+'/'

        dest_dir = BASE_DIR+name+gender
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        
        if not os.path.isfile(dest_dir+current_file):
            shutil.copy(CLIPS_DIR+current_file, dest_dir+current_file)
    
    return

if __name__ == '__main__':

    test_df = pd.read_csv(TEST_FILE, sep='\t').dropna(subset=['gender'])
    train_df = pd.read_csv(TRAIN_FILE, sep='\t').dropna(subset=['gender'])

    iter_and_split(test_df, 'test')
    iter_and_split(train_df, 'train')