
from numpy import mat
from regex import R
from tqdm import tqdm
import pandas as pd
from funcs import match, franco_2ar
from nltk.corpus import words
import os



def franco_match(ars , ens):   

    #Franco translated to ar 
    words_Xquran_Xpunct_Xeng_trans_franco = []
    for i in tqdm(range(0 , len(ens))):
        try:
            words_Xquran_Xpunct_Xeng_trans_franco.append( franco_2ar(ens[i]))
        except:
            words_Xquran_Xpunct_Xeng_trans_franco.append( "nan")
            print(ens[i])



    #match arFranco asle
    matches = []
    for i in tqdm(range(len(words_Xquran_Xpunct_Xeng_trans_franco))):
        try:
            matches.append(match(ars[i],words_Xquran_Xpunct_Xeng_trans_franco[i]))
        except:
            matches.append('nan')


    #*************************************************************************************************#
    #print & save 
    print("len(words_Xquran_Xpunct_Xeng)" , len(ars))
    print("len(words_Xquran_Xpunct_Xeng_trans)" , len(ens))
    print("len(words_Xquran_Xpunct_Xeng_trans_franco)" , len(words_Xquran_Xpunct_Xeng_trans_franco))
    print("len(matches)" , len(matches))


    return words_Xquran_Xpunct_Xeng_trans_franco , matches








#*************************************************************************************************#
#*************************************************************************************************#
def filter_match(filter, ars, ens, frs , matches):

    from nltk.corpus import words
    eng_dict = words.words()

    filtered_ar  = []
    filtered_en  = []
    filtered_fr  = []
    filtered_ma  = []

    for i in tqdm( range( len(matches)) ):
        try:
            if((matches[i])>= filter):
                if(ens[i].lower().strip() in eng_dict):    
                    filtered_ar.append(ars[i])
                    filtered_en.append(ens[i])
                    filtered_fr.append(frs[i])
                    filtered_ma.append(matches[i])
        except:
            print(matches[i])

    print("len after filter match > " + str(filter)  +  " : " +  str(len(filtered_ar)))


    return filtered_ar, filtered_en ,  filtered_fr, filtered_ma