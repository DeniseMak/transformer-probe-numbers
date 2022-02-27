import time
import os
# result files with sentences
filenames = ["../results/sent/ja_syn_d-bert_test_preds.csv",
             "../results/sent/en_syn_d-bert_test_preds.csv",
             "../results/sent/dk_syn_d-bert_test_preds.csv",
             "../results/sent/fr_syn_d-bert_test_preds.csv" ]

# english result files
eng_filenames_syn = ["../results/sent/en_syn_d-bert_test_preds.csv",
                 "../results/sent/en_syn_d-bert_train_preds.csv"]
eng_filenames_sem = ["../results/sent/en_sem_d-bert_test_preds.csv",
                 "../results/sent/en_sem_d-bert_train_preds.csv"]




root_dir_syn = '../num-syn-tmp/'
root_dir_sem = '../num-sem-tmp/'

def create_dataset(file_list, root):
    '''

    :param file_list: list of length 2, containing paths to test and train files
    Currently, test should be the first, and train is the second.

    :param root:
    :return:
    '''
    train_file = file_list[1]
    test_file = file_list[0]

    if not os.path.exists(root):
        os.makedirs(root)

    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for subfolder_name in ['pos', 'neg']:
        if not os.path.exists(os.path.join(train_dir, subfolder_name)):
            os.makedirs(os.path.join(train_dir, subfolder_name))
        if not os.path.exists(os.path.join(test_dir, subfolder_name)):
            os.makedirs(os.path.join(test_dir, subfolder_name))
    train_dir = root + '/train/'
    test_dir = root + '/test/'
    for file in file_list:
        l_count = 0
        if file == train_file:
            dir = train_dir
        elif file == test_file:
            dir = test_dir
        with open(file, "r") as f:
            lines = f.readlines()
            # for each line in lines, make a file for it in a directory based on position[1]
            for l in lines:
                if (l.strip().split(',')[1]=='0'):
                    # positive
                    l_count += 1
                    with open( dir + 'pos' + '/' + str(l_count) + '.txt', "w") as f_out:
                        f_out.write(l.strip().split(',')[0])
                if (l.strip().split(',')[1]=='1'):
                    # negative
                    l_count += 1
                    with open( dir + 'neg' + '/' + str(l_count) + '.txt', "w") as f_out:
                        f_out.write(l.strip().split(',')[0])

def print_stats():
    for filename in filenames:
        with open(filename, "r") as f:
            lines = f.readlines()

        preds = [(l.strip().split(',')[1],l.strip().split(',')[2]) for l in lines]

        all_count = 0
        match_count = 0
        false_pos_0_1 = 0
        false_neg_1_0 = 0
        for p in preds:
            all_count += 1
            if p[0] == p[1]:
                match_count += 1
            elif p[0] == '0':
                false_pos_0_1 += 1
            elif p[0] == '1':
                false_neg_1_0 += 1

        print("file={}, All={}, match={}, false_pos={}, false_neg={}".format(filename, all_count-1,
                                                                    match_count, false_pos_0_1, false_neg_1_0))

# print_stats()

def create_eng_syn_sen_dataset(file_list= eng_filenames_syn, root = root_dir_syn):
    t = time.process_time()
    create_dataset(file_list, root)
    print('files created in {} seconds'.format(time.process_time()-t))

def create_eng_sem_sen_dataset(file_list= eng_filenames_sem, root = root_dir_syn):
    t = time.process_time()
    create_dataset(file_list, root)
    print('files created in {} seconds'.format(time.process_time()-t))