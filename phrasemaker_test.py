#Resume Phrase Matcher code


#importing all required libraries

from pdfminer.high_level import extract_text
import os
import csv
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

#Function to read resumes from the folder one by one
mypath='/home/pratik/phraseMatcher/resumes/' #enter your path here where you saved the resumes
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

def pdfextract(file):
    text =[]
    with open(file,'rb') as f:
        t = extract_text(f)
        print (type(t))
        text.append(t)
    return text
'''

def pdfextract(file):
    text =[]
    with open(file,'rb') as f:
        t = extract_text(f)
        #print(t)
        a = t.find("PROJECT")
        b = t.find("Project")
        if a:
            c1 = t[a:len(t)]
            text.append(c1)
        else:
            c2 = t[b:len(t)]
            text.append(c2)
    return text
'''
#function to read resume ends


#function that does phrase matching and builds a candidate profile
def create_profile(file):
    text = pdfextract(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('/home/pratik/phraseMatcher/template_csv/vrjxf-mwtor.csv', encoding='utf-8')
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]
    
    pattern1 = [[stats_words[0]], [stats_words[1]], [stats_words[2]],[stats_words[3]],[stats_words[4]],[stats_words[6]],[stats_words[7]]]
    pattern2 = [[NLP_words[0]], [NLP_words[1]], [NLP_words[4]],[NLP_words[5]],[NLP_words[6]],[NLP_words[7]],[NLP_words[8]], [NLP_words[12]],[NLP_words[17]],[NLP_words[18]]]
    pattern3 = [[ML_words[0]], [ML_words[1]], [ML_words[2]], [ML_words[3]], [ML_words[4]], [ML_words[5]], [ML_words[6]], [ML_words[8]]]
    pattern4 = [[DL_words[0]], [DL_words[1]], [DL_words[3]], [DL_words[4]], [DL_words[5]], [DL_words[6]], [DL_words[7]], [DL_words[11]], [DL_words[12]]]
    pattern5 = [[R_words[0]], [R_words[1]], [R_words[2]], [R_words[3]], [R_words[4]], [R_words[5]], [R_words[6]], [R_words[7]]]
    pattern6 = [[python_words[0]], [python_words[1]], [python_words[2]], [python_words[3]], [python_words[4]], [python_words[5]], [python_words[6]]]
    pattern7 = [[Data_Engineering_words[0]], [Data_Engineering_words[1]], [Data_Engineering_words[2]], [Data_Engineering_words[3]], [Data_Engineering_words[4]], [Data_Engineering_words[5]], [Data_Engineering_words[12]], [Data_Engineering_words[13]]]
            
    matcher = PhraseMatcher(nlp.vocab)
    
    matcher.add('Stats', *pattern1, on_match=None)
    matcher.add('Stats', *pattern2, on_match=None)
    matcher.add('Stats', *pattern3, on_match=None)
    matcher.add('Stats', *pattern4, on_match=None)
    matcher.add('Stats', *pattern6, on_match=None)
    matcher.add('Stats', *pattern7, on_match=None)
    matcher.add('Stats', None, *stats_words)
    
    matcher.add('NLP', *pattern1, on_match=None)
    matcher.add('NLP', *pattern3, on_match=None)
    matcher.add('NLP', *pattern4, on_match=None)
    matcher.add('NLP', *pattern6, on_match=None)
    matcher.add('NLP', *pattern7, on_match=None)
    matcher.add('NLP', None, *NLP_words)
    
    matcher.add('ML', *pattern2, on_match=None)
    matcher.add('ML', *pattern4, on_match=None)
    matcher.add('ML', *pattern7, on_match=None)
    matcher.add('ML', *pattern1, on_match=None)
    matcher.add('ML', None, *ML_words)
    
    matcher.add('DL', *pattern1, on_match=None)
    matcher.add('DL', *pattern2, on_match=None)
    matcher.add('DL', *pattern3, on_match=None)
    matcher.add('DL', *pattern7, on_match=None)
    matcher.add('DL', None, *DL_words)
    
    matcher.add('R', *pattern1, on_match=None)
    matcher.add('R', *pattern2, on_match=None)
    matcher.add('R', *pattern3, on_match=None)
    matcher.add('R', *pattern6, on_match=None)
    matcher.add('R', None, *R_words)
    
    matcher.add('Python', *pattern1, on_match=None)
    matcher.add('Python', *pattern2, on_match=None)
    matcher.add('Python', *pattern3, on_match=None)
    matcher.add('Python', *pattern4, on_match=None)
    matcher.add('Python', *pattern7, on_match=None)
    matcher.add('Python', None, *python_words)
    
    matcher.add('DE', *pattern1, on_match=None)
    matcher.add('DE', *pattern2, on_match=None)
    matcher.add('DE', *pattern3, on_match=None)
    matcher.add('DE', *pattern5, on_match=None)
    matcher.add('DE', None, *Data_Engineering_words)
    
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
       
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

    return(dataf)
        
#function ends
        
#code to execute/call the above functions

final_database=pd.DataFrame()
i = 0 
while i < len(onlyfiles):
    file = onlyfiles[i]
    dat = create_profile(file)
    final_database = final_database.append(dat)
    i +=1
    #print(final_database)
pre_sample = final_database.to_csv('Detailed.csv')

    
#code to count words under each category and visulaize it through Matplotlib

final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace = True)
final_database2.fillna(0,inplace=True)
new_data = final_database2.iloc[:,1:]
new_data.index = final_database2['Candidate Name']
#execute the below line if you want to see the candidate profile in a csv format
sample2=new_data.to_csv('Compound.csv')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        #label = str(j)+": " + str(new_data.loc[i][j])
        label = str(int(new_data.loc[i][j]))
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')
plt.show()
