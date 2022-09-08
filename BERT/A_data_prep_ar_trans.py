

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Write functions used in google translation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from tqdm import tqdm
import pandas as pd


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# list to excel and save
def list_excel(LIST,num,dest_path):
    df = pd.DataFrame(LIST)
    writer = pd.ExcelWriter(dest_path + str(num) +'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='welcome', index=False)
    writer.save()

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# text to excel 
def text_to_excel(src_path, dest_path, step):

    with open(src_path, encoding="utf8") as f:
        src_paths_texts = f.readlines()

    src_texts = [(" ".join(a.split(" ")[1:])).replace("\n","").strip() for a in src_paths_texts]
    src_paths = [ a.split(" ")[0].replace("\n","").strip() for a in src_paths_texts]

    for i in range(0,len(src_texts),step):
        if(len(src_texts)<(i+step)):
            list_excel(src_texts[i:len(src_texts)],i,dest_path)
            break
        list_excel(src_texts[i:i+step],i,dest_path)


    return src_paths