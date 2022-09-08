

from difflib import SequenceMatcher


def match(W1,W2):
    return (SequenceMatcher(None, W1.replace(" ", ""), W2.replace(" ", "")).ratio())



def franco_2ar(W):
    en_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ','-',"'"]

    ar_letters = ['ا','ب','س','د','ي','ف','ج','ه','ي','ج','ك','ل','م'
                    ,'ن','و','ب','ق','ر','س','ت','و','ف','و','كس','ي','ز', ' ','-',"'"]

    W_ar = ""
    for l in W:
        if(l.lower() in en_letters):
            W_ar += ar_letters[en_letters.index(l.lower())]
        else:
            W_ar += l.lower()
    return W_ar
