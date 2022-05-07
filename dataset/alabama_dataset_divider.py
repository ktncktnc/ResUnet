import os
import random


def main(output_path='.', input_path='./image'):
    files = os.listdir(input_path)
    random.shuffle(files)

    size = len(files)
    train_size = int(size*0.7)
    val_size = int(size*0.15)
    test_size = size - train_size - val_size

    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[-test_size:]

    with open(os.path.join(output_path, 'train.txt'), 'w') as train_txt:
        for f in train_files:
            train_txt.write(f + '\n')
    
    with open(os.path.join(output_path, 'val.txt'), 'w') as val_txt:
        for f in val_files:
            val_txt.write(f + '\n')

    with open(os.path.join(output_path, 'test.txt'), 'w') as test_txt:
        for f in test_files:
            test_txt.write(f + '\n')

if __name__ == '__main__':
    main(
        '/root/alabama/alabama/',
        '/root/alabama/alabama/image'
    )