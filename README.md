# Code-Switching Text Augmentation From Monolingual one
* Code-Switching (CS) is shifting from one linguistic code (a language or a dialect) to another.

* In this project, we seek to make CS data from a monolingual one, focusing on Egyptian-English pairs.

* We followed four augmentation methods. These methods result in about 16 million CS sentences out of our 37 million monolingual ones. By this, we can overcome the scarcity of CS data, which opens a door for more humanlike Deep Learning applications such as language, translation models, and real-life chatbots.


* The code for each method is here, and the description in detail is in the paper: 
  1. **Franco search:** This method aims to find transliterated English words in each sentence and write them in their mother language.

      * Code is in the "Franco/" folder. 

      * Files: "Franco\A_clean_quran_punc_eng.py", "Franco\B_read_trans_franco_match.py", "Franco\C_Filter_words.ipynb" are the filtration steps and all of them are called in "Franco/Call_All.ipynb" file. Further postprocessing steps are in "Franco\D_sort_words&split.ipynb", and "\Franco\E_GOML.ipynb".

      * "data/" folder contains the output:
        * "Franco\data\words_en_ar_after_filter_ALL.txt" contain the transliterated words and their English translation. They are sorted by their length in "Franco\data\words_en_ar_after_filter_ALL_sorted.txt". 
        
        * "Franco\data\splits_lines\" contains the lines that have transliterated words before and after translation. They are split to be smaller in processing.
    


        * Full output data for Franco method is here:
          -[out1](https://drive.google.com/file/d/1RgYQw2gQYKOyEDmoh7UxGXkObBH-5gTZ/view?usp=sharing)
          -[out2](https://drive.google.com/file/d/1cJEUIrGZorq_n_g05NRhBWGXrdywxLnQ/view?usp=sharing)







  2. **Word alignment:** This method substitutes a word contextually in an ar sentence with its en translation.
  
      * Code is in the "word_aligmemnt/" folder. 
      * "word_aligment\aligment_simi.ipynb" is the main file consisting of a cell for all the needed imports, a cell for the helper functions, and a cell containing the "trans_aligment" function, which is the main function to be called for producing an Arabic-English CS sentence from an Arabic one. Its parameters are as follows:
           1. test_ar: Arabic sentence to run CS on it.
           2. test_en_esk: translation of test_ar by the AIC model.
              test_en_google: translation of test_ar by the Google model.
           3. words_ar_eskndrea, words_en_eskndrea, words_ar_google, words_en_google: list of previously translated words for time-saving, if a word isn't found it will be translated. 
           4. method_trans: AIC or Google.
           5. print_: print the details of the steps or just the output.
           6. first_: first time to run the algorithm or last_dict = {} has inputs to be checked.
           7. last_dict: if not empty, consider only the words in it.
      * "word_aligment\aligment_simi.py" is the same as ".ipynb", but is used for multiple inputs. it can be run by ""word_aligment\multiple_cpu.sh" file.



  3. **Matching:** This method substitutes a word contextually in an ar sentence with its en translation.
  
      * This method matches an Arabic sentence to an English one.
      * Steps:
          1. Translate all ar sentences.
          2. Vectorize all the translated ar sentences by BERT vectorization.
          3. Collect some English sentences.
          4. Vectorize all the English sentences.
          5. For each Arabic vector, find the highest cosine similarity with all the English ones.
          6. Concatenate the ar sentence with its highest similarity.

      * Code is in the "Match/" folder:
        * "Match\A_data_prep_ar_trans.py" split data into chunks to upload them to the Google translation website, then collect the translation.
        * "Match\B_BERT.py" vectorize all of the sentences.
        * "Match\C_Match.ipynb" match the most similar English sentence to the Arabic one.

  4.  **Chatbot:** We collected English chatbot data of 74383 sentences from Kaggle, translated the Questions into Arabic via Google translation, and left the answers as it is.
      * Code and the data are here: "Q&A(Chatbot)/":
        * "Q&A(Chatbot)/Data/" contains the originally collected data, and "Q&A(Chatbot)\Q_ar_A_en.xlsx" contains the data after translation.
