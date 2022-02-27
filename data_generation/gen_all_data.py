import os

langs = ['en', 'ja', 'dk', 'fr']
tasks = ['syn_len_match', 'sem_updated']
# NOTE: Moved sent vs no-sent functionality into individual gen files

for task in tasks:
    for lang in langs:
        print('Generating: \nTask: {}, Lang: {}'.format(task.upper(), lang.upper()))
        os.system('python3 ./gen_{}_data.py -lang {} -samples 50000 -range 1000 -sent'.format(task, lang))