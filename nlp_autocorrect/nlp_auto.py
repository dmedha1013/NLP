import re
from collections import Counter
import numpy as np
import pandas as pd

def process_data(file_name):
    words = []
    with open(file_name, 'r') as file:
        text = file.read()
    text = text.lower()
    words = re.findall(r'\w+', text)
    return words

word_l = process_data('./data/shakespeare.txt')
vocab = set(word_l)  # this will be your new vocabulary
print(f"The first ten words in the text are: \n{word_l[0:10]}")
print(f"There are {len(vocab)} unique words in the vocabulary.")

def get_count(word_l):
    word_count_dict = {}
    for word in word_l:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
    return word_count_dict

word_count_dict = get_count(word_l)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")

def get_probs(word_count_dict):
    probs = {} 
    total_words = sum(word_count_dict.values())
    for word, count in word_count_dict.items():
        probs[word] = count / total_words
    return probs

probs = get_probs(word_count_dict)
print(f"Length of probs is {len(probs)}")
print(f"P('thee') is {probs['thee']:.4f}")

def delete_letter(word, verbose=False):
    delete_l = []
    split_l = []
    
    for i in range(len(word) + 1):
        split_l.append((word[0:i], word[i:]))
        
    delete_l = [L + R[1:] for L, R in split_l if R]

    if verbose: 
        print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return  delete_l

delete_word_l = delete_letter(word="cans",
                        verbose=True)
print(f"Number of outputs of delete_letter('at') is {len(delete_letter('at'))}")

def switch_letter(word, verbose=False):
    switch_l = []
    split_l = []
    for i in range(len(word) + 1):
        split_l.append((word[0:i], word[i:]))
        
    switch_l = [L[:-1] + R[0] + L[-1] + R[1:] for L, R in split_l if len(L)>0 and len(R)>0]
    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}") 
    return switch_l

switch_word_l = switch_letter(word="eta",
                         verbose=True)

print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")

def replace_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    replace_l = []
    split_l = []
    for i in range(len(word) + 1):
        split_l.append((word[0:i], word[i:]))
        
    replace_l = [a + c + b[1:] for a, b in split_l if b for c in letters]
    
    while word in replace_l:
        replace_l.remove(word)
    replace_l = sorted(replace_l)
    
    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")   
    
    return replace_l
replace_l = replace_letter(word='can',
                              verbose=True)
print(f"Number of outputs of replace_letter('at') is {len(replace_letter('at'))}")

def insert_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []
    for i in range(len(word) + 1):
        split_l.append((word[0:i], word[i:]))
        
    insert_l = [a + c + b for a, b in split_l for c in letters]
    
    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")
    
    return insert_l

insert_l = insert_letter('at', True)
print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")

def edit_one_letter(word, allow_switches = True):
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    return set(edit_one_set)

tmp_word = "at"
tmp_edit_one_set = edit_one_letter(tmp_word)
# turn this into a list to sort it, in order to view it
tmp_edit_one_l = sorted(list(tmp_edit_one_set))

print(f"input word {tmp_word} \nedit_one_l \n{tmp_edit_one_l}\n")
print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")

def edit_two_letters(word, allow_switches = True):
    edit_two_set = set()

    edit_one_set = edit_one_letter(word, allow_switches)
    for word in edit_one_set:
        edit_two_set.update(edit_one_letter(word, allow_switches))
    return set(edit_two_set)

tmp_edit_two_set = edit_two_letters("a")
tmp_edit_two_l = sorted(list(tmp_edit_two_set))
print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
print(f"First 10 strings {tmp_edit_two_l[:10]}")
print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")

print( [] and ["a","b"] )
print( [] or ["a","b"] )
#example of Short circuit behavior
val1 =  ["Most","Likely"] or ["Less","so"] or ["least","of","all"]  # selects first, does not evalute remainder
print(val1)
val2 =  [] or [] or ["least","of","all"] # continues evaluation until there is a non-empty list
print(val2)


def get_corrections(word, probs, vocab, n=2, verbose = False):
    suggestions = []
    n_best = []

    suggestions = list((word in vocab and word) or 
                       edit_one_letter(word).intersection(vocab) or 
                       edit_two_letters(word).intersection(vocab) or
                       [word])
    
    n_best = sorted([(s, probs.get(s, 0)) for s in suggestions], key=lambda x: x[1], reverse=True)[:n]
    
    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best

my_word = 'dys' 
tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=True) 
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")


print(f"data type of corrections {type(tmp_corrections)}")

def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    m = len(source) 
    n = len(target) 
    D = np.zeros((m+1, n+1), dtype=int) 
    
    for row in range(1, m+1):
        D[row,0] = D[row-1,0] + del_cost
        
   
    for col in range(1, n+1): 
        D[0, col] = D[0, col-1] + ins_cost
        
    for row in range(1,m+1):
        
        for col in range(1,n+1):
            
           
            r_cost = None
           
            if source[row-1] == target[col-1]: 
                r_cost = 0
            else:
                r_cost = rep_cost
                
            D[row,col] =  min(D[row-1, col] + del_cost, D[row, col-1] + ins_cost, D[row-1, col-1] + r_cost)
            
    med = D[m, n]
    return D, med