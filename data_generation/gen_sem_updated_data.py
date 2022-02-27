from sklearn.model_selection import train_test_split
from num2words import num2words
import pandas as pd
import argparse
import random
import os
import re

def main(): 
    args = parse_all_args()
    pairs = list()
    
    if args.load:
        pairs = load_prev(args.load)
    else:
        pairs = gen_pairs(args.range, args.samples)
    
    labels, pairs = gen_sem_labels(pairs)

    if not args.load:
        output_pairs(args.dir + "sem_int_pairs.txt", pairs)
    
    sent_text, nums = to_text(pairs, args.lang, True)
    no_sent_text, nums = to_text(pairs, args.lang, False)
    #final, nums = to_text(pairs, args.lang, args.sent)

    num_data = pd.DataFrame({'nums' : nums, 'labels' : labels})
    sent_data = pd.DataFrame({'sents' : sent_text, 'labels' : labels})
    no_sent_data = pd.DataFrame({'sents' : no_sent_text, 'labels' : labels})

    s_train_data, s_test_data = train_test_split(sent_data, test_size=0.2)

    # Code below ensures the same numbers in train and test across sent and no-set data sets
    ns_train_data = no_sent_data.loc[s_train_data.index]
    ns_test_data = no_sent_data.loc[s_test_data.index]
    s_train_data = s_train_data.reset_index()
    s_test_data = s_test_data.reset_index()
    ns_train_data = ns_train_data.reset_index()
    ns_test_data = ns_test_data.reset_index()

    del s_train_data['index']
    del s_test_data['index']
    del ns_train_data['index']
    del ns_test_data['index']


    # Output
    s_train_data.to_csv(args.dir + "sent/" + args.lang + "_sem_train.csv")
    ns_train_data.to_csv(args.dir + "no-sent/" + args.lang + "_sem_train.csv")
    s_test_data.to_csv(args.dir + "sent/" + args.lang + "_sem_test.csv")
    ns_test_data.to_csv(args.dir + "no-sent/" + args.lang + "_sem_test.csv")
    num_data.to_csv(args.dir + args.lang + "_sem_nums.csv")    

def load_prev(path):
    """
    Load random integers from previously generated dataset

    :param path: (str) Path to integer pair file
    :return pairs: (list) Data from file as list of integer pairs
    """
    with open(path, "r") as f:
        lines = f.readlines()

    pairs = list()
    for line in lines:
        line = line.split("; ")
        pair = [int(x) for x in line]
        pairs.append(pair)

    return pairs

def filter_pairs(pairs, s):
    """
    Generate random 'labels' for the syntactic task and use them to chose 
    whether to keep both numbers in a pair. Later, if there are 2 numbers in a
    pair they will be used to generate 'ungrammatical' words, and single 
    numbers will be grammatical

    :param pairs: (list) Randomly generated list of integer pairs
    :param s: (int) Number of pairs

    :return new_pairs: (list) Pairs with random entries reduced to length one
    :return labels: (list) Denotes which pairs have been reduced
    """
    labels = gen_ints(1, s)
    new_pairs = list()
    for i in range(0, s):
        new_pair = [pairs[i][0]]
        if labels[i] == 0:
            new_pair.append(-1)
        else:
            new_pair.append(pairs[i][1])
        new_pairs.append(new_pair)

    return new_pairs, labels

def to_text(pairs, lang, sent):
    """
    Convert positive integers in list of lists to word form

    :param pairs: (list) Integers to convert
    :param lang: (str) Language to convert them to

    :return text: (list) Integer pairs in word form
    """
    text = list()
    nums = list()

    with open('./templates/' + lang + '_templates.txt', 'r') as f:
        sentences = f.readlines()

    for pair in pairs:
        if sent:
            sent = random.choice(sentences)
        new = [num2words(pair[0], lang=lang)]
        num_pair = list()
        if pair[1] > -1:
            new.append(num2words(pair[1], lang=lang))

        for i in range(0, len(new)):
            new[i] = new[i].replace('-', ' ')
            new[i] = new[i].replace(',', ' ')
            new[i] = re.sub(' +', ' ', new[i])
            new[i] = new[i].strip()
            num_pair.append(new[i])
            if sent:
                # specific stuff for japanese counters:
                temp_sent = sent
                if lang == 'ja':
                    counter_idx = sent.find('***') + 3
                    if sent[counter_idx] == '個':
                        if pair[i] < 10 and pair[i] > -10:
                            #print('Replacing ko with tsu')
                            sent = sent[:counter_idx] + 'つ' + sent[counter_idx+1:]
                        elif pair[i] == 10:
                            #print('removing ko altogether')
                            sent = sent[:counter_idx] + sent[counter_idx+1:]
                # end japanese specific code
                new[i] = sent.replace('***', new[i]).strip()
                sent = temp_sent
        nums.append('; '.join(num_pair))
        text.append('; '.join(new))

    return text, nums

def gen_sem_labels(pairs):
    """
    Create labels based on whether integer in position 0 of pair is greater than (0), less 
    than (1), or equal (2) to the other.

    :param pairs: (list) Integer pairs to label
    :return labels: (list) Classes for given pairs
    """
    labels = list()
    for i in range(0, len(pairs)):
        if pairs[i][0] > pairs[i][1]:
            labels.append(0)
        elif pairs[i][0] < pairs[i][1]:
            labels.append(1)
        else:
            pairs[i][0] += 1
            labels.append(0)
    return labels, pairs

def gen_pairs(r, s):
    """
    Create of list of [s] pairs of integers in range (0, r)

    :param r: (int) Max value for integers in list
    :param s: (int) Number of pairs to 
    :return pairs: (list) List of generated number pairs
    """
    p1 = gen_ints(r, s)
    p2 = gen_ints(r, s)
    pairs = [[p1[i], p2[i]] for i in range(0, s)] 
    return pairs

def output_pairs(path, data):
    """
    Format data and write it to a specified file

    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    :param mode: (str) How to join the data into strings
    """
    string = ""
    
    for item in data:
        
        line = ", ".join(str(x) for x in item)
        string += line + "\n"

    with open(path, "w+") as f:
        f.write(string)

def output_indvs(path, data):
    """
    Format data and write it to a specified file

    :param path: (str) Filepath to write to
    :param data: (list) Data to format and write
    """
    string = ""
    for item in data:

        line = str(item)
        string += line + "\n"

    with open(path, "w+") as f:
        f.write(string)

def gen_ints(r, samples):
    """
    Generate a specified number of random integers

    :param r: (int) Max value of integers to be generated
    :param samples: (int) Amount of integers to be generated

    :return (list): Random integers in range (2,r) of length s
    """
    ints = list()
    for i in range(0, samples):
        ints.append(random.randint(2, r))
    return ints

def parse_all_args():
    """
    Parse commandline arguments and create folder for output if necessary

    :return args: Parsed arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-range",type=int,\
            help="Max value of integers to be generated [default=1000]",default=1000)
    parser.add_argument("-samples",type=int,\
            help="The number of integer pairs to be generated [default=100]",default=100)
    parser.add_argument("-dir",type=str,\
            help="Output directory for number pairs generated [default=data]", default="data")
    parser.add_argument("-lang",type=str,\
            help="Language of data to be generated [default=en]", default="en")
    parser.add_argument("-load",type=str,\
            help="Location of  previous set of integer pairs to create data")
    parser.add_argument('-sent', dest='sent', help='Whether or not to generate numbers in sentences', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.dir[-1] != "/":
        args.dir += "/"

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    #if args.sent:
    #    args.dir += 'sent/'
    #else:
    #    args.dir += 'no-sent/'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    return args

if __name__ == '__main__':
    main()