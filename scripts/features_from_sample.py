import os
import numpy as np
import librosa

# acceder au dossier
BASE_DIR = '../dataset/Common1/'
CLIPS_DIR = os.path.join(BASE_DIR, 'clips/')

# creer dossier feature si il existe po
FEATURES_DIR = os.path.join(BASE_DIR, 'mfcc_features/')
if not os.path.isdir(FEATURES_DIR):
    os.mkdir(FEATURES_DIR)

# iterration au travers du dataset
files_list = os.listdir(CLIPS_DIR)
for file in files_list:
    # lecture du fichier
    file_path = os.path.join(CLIPS_DIR, file)
    y, sr = librosa.load(file_path, sr=None)

    # calcul de la mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    
    # enregistrer le .npy
    filename_no_extension = os.path.splitext(file)[0]
    new_file_path = os.path.join(FEATURES_DIR, filename_no_extension)
    np.save(new_file_path, mfcc_mean)