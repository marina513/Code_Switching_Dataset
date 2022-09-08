#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Eskndrea imports
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer
from fairseq.models.transformer import TransformerModel
import torch
from apply_subword_nmt_bpe import BPE
import re
import sys
from tqdm import tqdm

model_dir = '/home/mmaher/trans_model_eskndrea/'
model = TransformerModel.from_pretrained(  
    model_dir, # model_name_or_path                                                                                    
    checkpoint_file='model.pt',
    data_name_or_path=model_dir,
    user_dir=f"{model_dir}/mcolt",
    task="translation",
    source_lang="LANG_TOK_AR",
    target_lang="LANG_TOK_EN"
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Camel imports
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
mle = MLEDisambiguator.pretrained('calima-msa-r13')
tokenizer_tshkel = MorphologicalTokenizer(mle, scheme='d3tok', split=True, diac=True)
tokenizer_no_tshkel = MorphologicalTokenizer(mle, scheme='d3tok', split=True)
mle_egy = MLEDisambiguator.pretrained('calima-egy-r13')
tokenizer_egy = MorphologicalTokenizer(mle_egy, scheme='d3tok', split=True)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# google trans imports
from googletrans import Translator
translator = Translator()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NLTK imports
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
english_words = words.words()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# spacy imports
import spacy
sp = spacy.load('en_core_web_sm')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# String imports
import string, os, re, inflect
punctuations = string.punctuation
punctuations = punctuations+"؟"
punctuations = punctuations+"“"
punctuations = punctuations+"”"
punctuations = punctuations+"’"

singularize = inflect.engine()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General imports
from tqdm import tqdm
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def is_English(string_test):
    string_test_list = string_test.strip().split(" ")
    
    for s in string_test_list:
        if(len(s)>0):
            s = rmv_punct(s, True)
            if((singularize.singular_noun(s) in english_words) or (s in english_words)  or (s[:-1] in english_words) or ((s[:-2] in english_words) and (s[-2:]=='ed') )):
                pass
            else:
                return False

    return True

def reverse_order(string_mix, dict_, string_en):

    dict_values = list(dict_.values())

    for k in range(len(dict_values)):
        if ((dict_values[k] + " " + dict_values[k-1]) in string_en):
            asl = dict_values[k-1] + " " + dict_values[k]
            b3d = dict_values[k] + " " + dict_values[k-1]
            string_mix = ( re.sub(rf'\b{asl}\b',b3d, string_mix  ))

    return string_mix


def tfkek_l_asl(s_ar):
    sennt = simple_word_tokenize(s_ar)
    tokens = tokenizer_no_tshkel.tokenize(sennt)
    tokens[0] = tokens[0].replace("+","")
    
    tokens2 = tokens
    for l in range(len(tokens)-1):
        if(tokens[l]=="ل" and tokens[l+1]== 'ال+'):
            tokens2.remove(tokens[l])
            tokens2.remove(tokens[l])
            tokens2.append("لل")

    return tokens
    
def tfkek_l_asl_egy(s_ar):
    sennt = simple_word_tokenize(s_ar)
    tokens = tokenizer_egy.tokenize(sennt)
    tokens[0] = tokens[0].replace("+","")

    return tokens

def shkl(str_no_tshkel):
    skip = False
    str_no_tshkel_tfkek = tfkek_l_asl(str_no_tshkel , True)
    s = ""
    for i in range(len(str_no_tshkel_tfkek)):
        if(not skip):
            if((str_no_tshkel_tfkek[i][0]=="+") ):
                s = s + str_no_tshkel_tfkek[i].replace("+","") 

            elif((str_no_tshkel_tfkek[i][-1]=="+") or (str_no_tshkel_tfkek[i]== 'ال')):
                s = s + " " +  str_no_tshkel_tfkek[i].replace("+","") + str_no_tshkel_tfkek[i+1]
                skip = True

            else:
                s = s + " " + str_no_tshkel_tfkek[i]
        else:
            skip = False
    return s.strip()
   

def translate_fun_google(s):
    translator = Translator()
    translated_text = translator.translate(s).text # Translate
    return translated_text

def translate_fun_google_ar(s):
    translator = Translator()
    translated_text = translator.translate(s,dest="ar").text # Translate
    return translated_text

def translate_fun(which, text):

    if(which=="eskndra"):
        return trans_ar_en_eskndrea(text)
        
    else:
        return translate_fun_google(text)

possessive_determiner = ["my","your","his","her","its","our","their","one's", "whose"]
def rmv_pssv(stri, sent):
    found  =  False
    if(stri in sent):
        return stri , found
    for i in possessive_determiner:
        if(re.findall(rf'\b{i}\b',stri)):
            stri = re.sub(rf'\b{i}\b',"",stri) 
            found = True
        
    return stri.strip(), found

def add_zeada(stri):
    zeadat = ["بت","ن","بي","فت","ي","مت","ت","ه","فلي","هات","ف","تُ","وي","يت", "أ","ا","بن"]
    zeada = ""
    for s in range(len(stri)):
        if(((stri[0:s+1] in zeadat) and (stri[0:s+2] not in zeadat) )):
            zeada = stri[0:s+1]
    return zeada

def check_verb(verb_ar):
    sentence = simple_word_tokenize(verb_ar)
    disambig = mle_egy.disambiguate(sentence)
    pos_tags = [d.analyses[0].analysis['pos'] for d in disambig][0]
    return pos_tags=="verb"

def check_verb_en(verb_en):
    sen = sp(verb_en)
    try:
        if(sen[0].pos_.lower()=='verb'): 
            return True

        if(verb_en[-1]=="s"):
            sen = sp(verb_en[:-1])#rmv s like plays
            if(sen[0].pos_.lower()=='verb'): 
                return True

        if(verb_en[-3:]=="ing"):
            sen = sp(verb_en[:-3])#rmv ing like playing
            if(sen[0].pos_.lower()=='verb'): 
                return True
        return False
    except:
        return False

def z7z7_punct(stri):
    for i in punctuations: 
        if(i in stri): stri =  stri.replace(i," "+ i + " ")
    return stri  

def al_and_arrang(s2):
    sent_split = s2.split()
    for i in range(len(sent_split)):
        if((i < len(sent_split)-1 ) and ((sent_split[i]=="ال") and (sent_split[i+1]=="و"))):
            sent_split[i] = "و"
            sent_split[i+1] = "ال"
    return " ".join(sent_split)

def rmv_repeated(sent_list):
    sent_list = (" ".join(sent_list)).split(" ")
    sent_list2 = sent_list.copy()
    for i in range(len(sent_list)-1):
        try:
            if(is_English(sent_list[i]) and (not is_English(sent_list[i+1])) ):
                if(sent_list[i] == ((translate_fun_google(sent_list[i+1]).split(" "))[0]).lower()):
                    sent_list2.remove(sent_list[i])
            if(not is_English(sent_list[i]) and (is_English(sent_list[i+1])) ):
                if((translate_fun_google(sent_list[i]).split(" ")[0]).lower() ==  sent_list[i+1]):
                    sent_list2.remove(sent_list[i+1])
        except:
            pass
    return  " ".join(sent_list2)

pronouns_ar = ["أ","ت","ن","+وا","ي","ت","","","","","",""]
def b_al_pronouns_arrange(s):
    sent_list = s.lower().split(" ")
    sent_list = [i for i in sent_list if(i not in ['', ' '])]

    sent_list = (" ".join(sent_list)).split(" ")
    for i in range(len(sent_list)-1):
        try:
            if((sent_list[i] == "ال") and ((sent_list[i+1]) in pronouns ) ):
                sent_list[i] = sent_list[i+1]
                sent_list[i+1] = "ال"
        except:
            pass

    for i in range(len(sent_list)-1):
        try:
            if((sent_list[i] in pronouns) and ((sent_list[i+1]) == "و" ) ):
                sent_list[i+1] = sent_list[i]
                sent_list[i] = "و"
        except:
            pass

    for i in range(len(sent_list)-1):
        try:
            if((sent_list[i] == "ب") and ((sent_list[i+1]) in pronouns ) ):
                if( (sent_list[i+1]) != "they"):
                    sent_list[i+1] = pronouns_ar[pronouns.index(sent_list[i+1])]
                else:
                    sent_list[i+1] = " ي "  + sent_list[i+2]
                    sent_list[i+2] = "وا" 
        except:
            pass
    return  " ".join(sent_list)
      
def check_pos_pronoun(en, s, i):
    en_list = en.split(" ")
    if(check_verb_en(en_list[i-2]) or (en_list[i-2] in ["am", "is", "are", "was", "were","about", "against"])):
        return False
    return True


def check_index(sent, w):
    i = (re.search(rf'\b{w}\b', sent)).span()[0]
    spaces = 0
    for i in range(0,i):
        if(sent[i]==" "):
            spaces+=1
    return spaces

def b_w_arrange(s):
    sent_list = s.split(" ")
    sent_list = (" ".join(sent_list)).split(" ")
    for i in range(len(sent_list)-1):
        try:
            if((sent_list[i] == "ب") and ((sent_list[i+1]) == "و" ) ):
                sent_list[i] = "و"
                sent_list[i+1] = "ب"
        except:
            pass
    return  " ".join(sent_list)
    
def rmv_punct(stri, all=False):

    for i in punctuations: 
        try:
            if(((i in stri)) and ( (not (( (i=="'") and (stri[stri.find("'")+1]=="s") ))) or all ) and ( (not (( (i=="’") and (stri[stri.find("’")+1]=="s") ))) or all ) ): 
                stri =  stri.replace(i,"")
        except:
            if(((i in stri))  ): 
                stri =  stri.replace(i,"")

    stri = stri.replace("'s"," 's ").replace("’s"," ’s")

    return stri


def check_pos_tag_en(sent, word):
    text = nltk.word_tokenize(sent)
    return dict(nltk.pos_tag(text))[word.strip().split(" ")[0]]

def check_ll_after_of(x, en_sent, ar_trans_sent_list, ar_sent_list):
    while( x < ( len( en_sent.split(" ")))):
        if( en_sent.split(" ")[x] not in stopwords.words() ):
            f = en_sent.split(" ")[x]
            break
        x+=1

    for i in range(len(ar_trans_sent_list)):
        if( re.search(rf'\b{f}\b', ar_trans_sent_list[i] ) ):
            if("لل" in tfkek_l_asl(ar_sent_list[i])):
                return True
    return False

    
def trans_ar_en_eskndrea(text):
    src='ar'  ; tgt='LANG_TOK_EN'

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalization & punctation
    moses_norm = MosesPunctNormalizer(lang="ar",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True
    )
    moses_tok = MosesTokenizer(lang="ar")

    normalized = moses_norm.normalize(text)
    tokenized = moses_tok.tokenize(normalized, aggressive_dash_splits=True, escape=True)
    tokenized = ' '.join(tokenized)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BPE
    subword_nmt_bpe_preprocessor = BPE(open(f"/home/mmaher/LM_CS/trans_methods/esk_trans/codes.bpe.32000", 'r'))
    text_subword_nmt_bpe_encoded = subword_nmt_bpe_preprocessor.process_line(tokenized)
    text_subword_nmt_bpe_encoded = tgt + " " + text_subword_nmt_bpe_encoded

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # predict
    model.eval()
    with torch.no_grad():
        translated_output = model.translate(text_subword_nmt_bpe_encoded, beam=5)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detokenize
    moses_detok = MosesDetokenizer(lang="en")
    subword_nmt_bpe_decoded = re.sub(r"(@@ )|(@@ ?$)", "", translated_output)
    tokens = subword_nmt_bpe_decoded.split(" ")
    detokenized = moses_detok.detokenize(tokens, return_str=True)

    return detokenized

def pos_tags_ar(ar):
    sentence = simple_word_tokenize(ar)
    disambig = mle_egy.disambiguate(sentence)
    return [d.analyses[0].analysis['pos'] for d in disambig]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main function    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pronouns = ["i" , "you" , "we" , "they" , "he", "she" , "it", "a" ,"an" , "in", "on", "at"]
def trans_aligment(ar_sent,en_sent, en_words, ar_words, print_=False, method_trans="google", first = True, last_dict_= {}):
    trans_dict = {}
    enter =True ; al_in_tokens = False ; b_in_tokens = False; and_in_tokens = False; 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # move punctuations away from words for correct searching in ".index" ex: civilian?
    ar_sent = rmv_punct(ar_sent)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # translate by CONTEXT
    en_sent = rmv_punct(en_sent.lower()).strip()   # ar trans to en

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # translate by WORD
    ar_sent_list = [i  for i in word_tokenize(ar_sent)]
    
    ar_trans_sent_list = []
    for i in ar_sent_list:
        try:
            ar_trans_sent_list.append( rmv_punct(en_words[ar_words.index(i)].lower()).strip() )
        except:
            print( i ," word not found, consider adding it later plz")
            ar_trans_sent_list.append( rmv_punct( translate_fun( which= method_trans , text = i).lower() ).strip() )
 
    ar_trans_sent = " ".join(ar_trans_sent_list)
    ar_sent_list2 = ar_sent_list.copy()

 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # replace ar by en if near
    for i in range(len(ar_trans_sent_list)):    
        al_in_tokens = False ; b_in_tokens = False; and_in_tokens = False; 
        search_word, pssv = rmv_pssv(ar_trans_sent_list[i] , en_sent)
        search_word = search_word.strip()
        
        if("the" in search_word.lower().split(" ")):
            al_in_tokens = True
            search_word = re.sub(rf'\b{"the"}\b',"",search_word.lower()).strip()
        
        if(("by" in search_word.lower().split(" ")) or ("with" in search_word.lower().split(" "))):
            b_in_tokens = True
            search_word = re.sub(rf'\b{"by"}\b',"",search_word.lower()).strip()
            search_word = re.sub(rf'\b{"with"}\b',"",search_word.lower()).strip()

        if(((search_word) not in stopwords.words() )  and ((search_word) not in punctuations ) and (re.search(rf'\b{search_word}\b', en_sent)) and (is_English(search_word))):
        
            index_context_trans = check_index(en_sent, search_word)# mkan el translated f context translation
            index_word_trans = check_index( ar_trans_sent , search_word)# mkan el translated f word translation
            a = ar_sent_list[i]
            index_context_ar = ar_sent_list.index(a)

            print(index_context_trans, index_word_trans, index_context_ar)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if((first) or (a in list(last_dict_.keys()))):
                
                if((abs(index_context_trans-index_word_trans) <= 4) and (abs(index_context_trans-index_context_ar) <= 4) ):
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # to skip sequentional words
                    if(enter):
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        #check pronouns
                        pieces_ar = [p.replace("+","") for p in tfkek_l_asl(a)]
                    
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # al & b in tokens check
                        if(("ال" in pieces_ar) and ("the" not in search_word.lower().split(" ")) )  : al_in_tokens = True
                        if(("ب" in pieces_ar) and ("by" not in search_word.lower().split(" ")) ) : b_in_tokens = True
                        if(("و" in pieces_ar) and ("and" not in search_word.lower().split(" ")) ) : and_in_tokens = True


                        search_word_no_zeada = search_word


                        if((len(pieces_ar) <= len(search_word.split(" "))) or (and_in_tokens and (len(pieces_ar) <= len(search_word.split(" "))+1)) or (al_in_tokens and (len(pieces_ar) <= len(search_word.split(" "))+1)) or (b_in_tokens and (len(pieces_ar) <= len(search_word.split(" "))+1)) or (b_in_tokens and al_in_tokens and (len(pieces_ar) <= len(search_word.split(" "))+2))  or (pssv and (len(pieces_ar) <= len(search_word.split(" "))+1)) ):# 3shan lw feh dmaer na2s mn translation
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # of check
                            n = en_sent.index(search_word)
                            if(en_sent[n+len(search_word)+1 :  n+len(search_word)+4] == "of "):
                                if( ( ar_sent_list[i+1] not in ["من", "عن", "الى", "", ""]) and ("لل" not in tfkek_l_asl(ar_sent_list[i+1])) and ("ل" not in tfkek_l_asl(ar_sent_list[i+1])) and ("ب" not in tfkek_l_asl(ar_sent_list[i+1])) and (pos_tags_ar(ar_sent)[i+1] !="adj") ):
                                    search_word = search_word + " of"               

                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # al & b in tokens update
                            if(al_in_tokens): search_word = " ال " + search_word ;  al_in_tokens = False  # put it after "of" check for correct searching
                            if(b_in_tokens): search_word = " ب " + search_word ; b_in_tokens = False  # put it after "of" check for correct searching
                            if(and_in_tokens): search_word = " و " + search_word ;  and_in_tokens = False  # put it after "of" check for correct searching

                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # The , And check
                            search_word =  re.sub(rf'\b{"the"}\b',"ال",search_word) ; search_word =  re.sub(rf'\b{"The"}\b',"ال",search_word)
                            search_word =  re.sub(rf'\b{"and"}\b',"و",search_word)  ; search_word =  re.sub(rf'\b{"And"}\b',"و",search_word)


                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # pronouns & posseisve pronouns check 3shan feh pronouns mbtgesh f trgma by word
                            if( ("ال" not in search_word) and (search_word.split(" ")[0] not in pronouns) and ((((en_sent).split(" "))[index_context_trans-1] in pronouns) and (((en_sent).split(" "))[index_context_trans-1]) not in search_word.split(" ")) and (index_context_trans!=0)):
                                # check mkan el pronoun ze how are you & you are beautiful
                                if(check_pos_pronoun(en_sent,search_word, index_context_trans)):
                                    #check if adjective not add 3shan beb2a aslha klmtee m2lobeen
                                    if( check_pos_tag_en(en_sent,search_word_no_zeada) != 'JJ'):
                                        search_word  =  ((en_sent).split(" "))[index_context_trans-1] + " " + search_word
                            

                            if( ("ال" not in search_word) and ((((en_sent).split(" "))[index_context_trans-1] in possessive_determiner) and (((en_sent).split(" "))[index_context_trans-1]) not in search_word.split(" ")) and (index_context_trans!=0)):
                                search_word  =  ((en_sent).split(" "))[index_context_trans-1] + " " + search_word

                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # check zeadat el af3al
                            if((check_verb( a )) and (check_verb_en( (search_word.split(" "))[0])) and (search_word[-3:]!="ing") and (search_word[-2:]!="ed")):
                                search_word = (add_zeada(a)) + search_word

                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            if(a not in ["طب","بدرى","بسرعة","فيها","بدري"]):
                                ar_sent_list2[index_context_ar] = search_word
                                trans_dict[a]=search_word
                                enter = False

                        else:
                            #print( "search_word : " , a , search_word)
                            #وأنه and that  ;;;  بحملات campaigns ;;;; بدري early
                            pass

        else:
            enter = True 

    if(print_):
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  context based ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("gomale 3arbe                 : ar_sent             " , ar_sent)
        print("trgmt el gomla el aslea      : en_sent             " , en_sent)

        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  word based ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("split el gomla el aslea      : ar_sent_list        " , ar_sent_list)
        print("trans word by word           : ar_trans_sent_list " , ar_trans_sent_list)
        print("trans word by word           : ar_trans_sent      " , ar_trans_sent)

        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("dict                         : trans_dict         " , trans_dict)
        print("mix out                      : ar_sent_list2        " , ar_sent_list2)    


    return trans_dict, en_sent, ar_sent_list2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Call function    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(p):
    print(p , " started decoding")
    with open(p) as f:
        lines = f.readlines()
    lines = list(set(lines))

    ar_en_sent_all = [] ; ar_sent_all = [] ; dicts = [] ; en_sents = []

    for i in tqdm(range(len(lines))):
        try:
            ar_sent = lines[i].replace("\n","") 
            
            # call       
            gomla  = ar_sent
            if(len(gomla.split(" "))==len(set(gomla.split(" ")))):
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #Aligment using eskndrea trans
                dict_, en_sent, s = trans_aligment(gomla,True,"eskndra")#ex: zeadat el af3al 

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #Aligment using google trans
                dict_, en_sent, s = trans_aligment(gomla,True,"google", first=False ,last_dict_= dict_)#ex: zeadat el af3al 
                s2 = b_w_arrange(b_al_pronouns_arrange(al_and_arrang(rmv_repeated(s))))
                if(s2.split(" ")[-1].strip()=="of"):
                    s2 = s2.strip()[:-2]


            # save
            if(len(dict_)>0):
                ar_en_sent_all.append("ااا  " + s2)
                ar_sent_all.append(ar_sent)
                dicts.append(dict_)
                en_sents.append(en_sent)
       
        except:
            print("error at file : " , p , " at line : " , i , lines[i])
             
            textfile = open("/home/mmaher/LM_CS/word_aligment/outs_150-300K/all/"         + str(i)  + "_"  + os.path.basename(p)  , "w")
            textfile_lines = open("/home/mmaher/LM_CS/word_aligment/outs_150-300K/lines/" + str(i)  + "_"  + os.path.basename(p) , "w")
            for e in range(len(ar_en_sent_all)):
                textfile.write(ar_sent_all[e]+  "\n")
                textfile.write(en_sents[e]+  "\n")
                textfile.write(ar_en_sent_all[e]+  "\n")
                textfile.write(str(dicts[e]))
                textfile.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
                
                textfile_lines.write(ar_en_sent_all[e]+  "\n")
                
            textfile.close()
            textfile_lines.close()



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save
    textfile = open("/home/mmaher/LM_CS/word_aligment/outs_150-300K/all/" + os.path.basename(p) , "w")
    textfile_lines = open("/home/mmaher/LM_CS/word_aligment/outs_150-300K/lines/" + os.path.basename(p) , "w")

    for e in range(len(ar_en_sent_all)):
        textfile.write(ar_sent_all[e]+  "\n")
        textfile.write(en_sents[e]+  "\n")
        textfile.write(ar_en_sent_all[e]+  "\n")
        textfile.write(str(dicts[e]))
        textfile.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        
        textfile_lines.write(ar_en_sent_all[e]+  "\n")
        
    textfile.close()
    textfile_lines.close()


if __name__ == "__main__" :
    file_name = sys.argv[1]
    main(file_name)
  
