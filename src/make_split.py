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

def split_files(files, split_train, split_val, use_test):
    files_train = files[:split_train]
    files_val = files[split_train:split_val] if use_test else files[split_train:]

    li = [(files_train, "train"), (files_val, "val")]

    # optional test folder
    if use_test:
        files_test = files[split_val:]
        li.append((files_test, "test"))
    return li

def img_train_test_split(source_dir, output, fixed_split, seed, sampling): 
    if not (isinstance(source_dir, str)):
        raise AttributeError('source_dir must be a string')
        
    if not os.path.exists(source_dir):
        raise OSError('source_dir does not exist')
        
    if isinstance(fixed_split, int):
        fixed_split = fixed_split

    assert len(fixed_split) in (1, 2)
        
    # Set up empty folder structure if not exists
    # if not os.path.exists(output):
    #     os.makedirs('data')


    subdirs = [x[0] for x in os.walk(source_dir)]

    removed_dirs = []
    for index, subdir in enumerate(subdirs):
        if not any(fname.endswith('.jpg') for fname in os.listdir(subdir)):
            removed_dirs.append(subdir)
    
    subdirs = [subdir for subdir in subdirs if subdir not in removed_dirs]      

    categories = ['_'.join(subdir.split(os.path.sep)[1:]) for subdir in subdirs]
    

    
    lens = []
    for category, subdir in zip(categories, subdirs):
        # Randomly assign an image to train or validation folder
        random.seed(seed)

        files = list_files(subdir)
        if len(files) == 0:
            pass
        else:
            files.sort()
            random.shuffle(files)
            
            if not len(files) > sum(fixed_split):
                print(f'not enough samples in "{subdir}"')
                #categories.remove(category)
            elif len(files) <= 2 * sum(fixed_split):
                print(f'making equal split in "{subdir}"')
                split_train = int(len(files)/2)
                split_val = split_train + int(len(files)/2)

                li = split_files(files, split_train, split_val, use_test=False)

                for (files, folder_type) in li:
                    full_path = os.path.join(output, folder_type, category)
                    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                    for f in files:
                        shutil.copy2(f, full_path)
            else:
                lens.append(len(files))
                split_train = len(files) - sum(fixed_split)
                split_val = split_train + fixed_split[0]

                #li = split_files(files, split_train, split_val, len(fixed_split) == 2)
                li = split_files(files, split_train, split_val, use_test=False)

                for (files, folder_type) in li:
                    full_path = os.path.join(output, folder_type, category)
                    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                    for f in files:
                        shutil.copy2(f, full_path)
    
    # max_len = max(lens)
    # iteration = zip(lens, categories)


    # for length, category in iteration:
    #     print('oversampling!!')
    #     full_path = os.path.join(output, "train", category)
    #     print(full_path)
    #     train_files = list_files(full_path)
    #     for i in range(max_len - length):
    #         f_orig = random.choice(train_files)
    #         new_name = f_orig.stem + "_" + str(i) + f_orig.suffix
    #         f_dest = f_orig.with_name(new_name)
    #         shutil.copy2(f_orig, f_dest)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='deboer',
                    help='which dataset to split')

    parser.add_argument('--output', default='output',
                    help='output folder')

    parser.add_argument('--fixed_split', type=int, default=(10, 0), help='fixed number of validation and test samples')
    parser.add_argument('--seed', type=int, default='666', help='seed')
    parser.add_argument('--sampling', default='False', help='oversample')

    args = parser.parse_args()
    img_train_test_split(args.source, args.output, args.fixed_split, args.seed, args.sampling)
