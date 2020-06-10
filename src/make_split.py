import os
import argparse
import random
import shutil
import pathlib


def list_files(dir):
    return [
        f for f in pathlib.Path(dir).iterdir()
        if f.is_file() and not f.name.startswith(".")
        ]

def split_files(files, split_train, split_test):
    files_train = files[:split_train]
    files_test = files[split_train:]
    li = [(files_train, "train"), (files_test, "test")]
    return li

def img_train_test_split(args): 
    source_dir = args.source
    output = args.output
    min_files = args.min_files,
    fixed_split = args.fixed_split
    seed = args.seed
    if not (isinstance(source_dir, str)):
        raise AttributeError('source_dir must be a string')
        
    if not os.path.exists(source_dir):
        raise OSError('source_dir does not exist')
        
    if isinstance(fixed_split, int):
        fixed_split = fixed_split


    subdirs = [x[0] for x in os.walk(source_dir)]


    removed_dirs = []
    for subdir in subdirs:
        if not any(fname.endswith('.jpg') for fname in os.listdir(subdir)):
            removed_dirs.append(subdir)

    
    subdirs = [subdir for subdir in subdirs if subdir not in removed_dirs]      

    categories = ['_'.join(subdir.split(os.path.sep)[1:]) for subdir in subdirs]
    
    lens = []
    for category, subdir in zip(categories, subdirs):
        # Randomly assign an image to train or test folder
        random.seed(seed)

        files = list_files(subdir)
        if len(files) < min_files:
            pass
        else:
            files.sort()
            random.shuffle(files)
            
            if not len(files) > fixed_split:
                print(f'not enough samples in "{subdir}"')
                #categories.remove(category)
            elif len(files) <= 2 * fixed_split:
                print(f'making equal split in "{subdir}"')
                split_train = int(len(files)/2)
                split_test = split_train + int(len(files)/2)

                li = split_files(files, split_train, split_test)

                for (files, folder_type) in li:
                    full_path = os.path.join(output, folder_type, category)
                    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                    for f in files:
                        shutil.copy2(f, full_path)
            else:
                lens.append(len(files))
                split_train = len(files) - fixed_split
                split_test = split_train + fixed_split

                li = split_files(files, split_train, split_test)

                for (files, folder_type) in li:
                    full_path = os.path.join(output, folder_type, category)
                    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                    for f in files:
                        shutil.copy2(f, full_path)
    
        

# if __name__ == '__main__':
    
    
