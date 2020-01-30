import os
import hashlib

LARGE_DATASET_NAME = 'notMNIST_large/'
SMALL_DATASET_NAME = 'notMNIST_small/'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def prefix(letter):
    return LARGE_DATASET_NAME + letter + "/"

def remove_duplicates(dir):
    unique = []
    for filename in os.listdir(dir):
        if os.path.isfile(dir + filename):
            filehash = hashlib.md5(open(dir + filename, "rb").read()).hexdigest()
        if filehash not in unique: 
            unique.append(filehash)
        else: 
            os.remove(dir + filename)

for letter in CLASSES:
    print(letter)
    remove_duplicates(prefix(letter))
