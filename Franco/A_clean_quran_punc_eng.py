#*************************************************************************************************#
#*************************************************************************************************#
#*************************************************************************************************#
#Import
from tqdm import tqdm
import regex as re



def rmv_Quran_punct_num_eng(path):
  
    # Read data & get all unique words
    with open(path) as f:
        data_all_final = f.readlines()
    print( "len all data : " , len(data_all_final))


    words_main = []
    for i in tqdm(range(len(data_all_final))):
        words_main.extend(data_all_final[i].split(" ")) 

    words_main = [w.replace("\n","") for w in words_main]
    words_main = list(set(words_main))
    print( "len all words : " , len(words_main))


    #*************************************************************************************************#
    #*************************************************************************************************#
    #Read Quran words
    with open('/home/mmaher/CS_2/Data/words_quran.txt', encoding="utf8") as f:
        words_quran = f.readlines()
    words_quran = [w.replace("\n","") for w in words_quran]
    print( "len quran words : " , len(words_quran))

    #Filter Quran words
    words_a3gme = []
    for w in tqdm(range(len(words_main))):
        if(words_main[w] not in words_quran):
            words_a3gme.append(words_main[w])
    print( "len words Xquran : " , len(words_a3gme))



    #*************************************************************************************************#
    #*************************************************************************************************#
    #rmv not in letters
    words_a3gme_ndef = list(set([ ("".join([x for x in w if x.isalpha()]))  for w in words_a3gme]))
    print( "len words Xquran_Xpunct : " , len(words_a3gme_ndef))

    #rmv eng
    words_a3gme_ndef_no_eng = list(set([re.sub('[a-zA-Z]+', '', w) for w in words_a3gme_ndef]))
    print( "len words Xquran_Xpunct_Xeng : " , len(words_a3gme_ndef_no_eng))


    return words_a3gme_ndef_no_eng