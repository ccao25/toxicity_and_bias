#!/usr/bin/env python
# coding: utf-8

# In[1]:


# V2


# In[2]:


get_ipython().system('pip install datasets transformers evaluate -q')
get_ipython().system('pip install torch')
get_ipython().system('pip install xformers')


# In[3]:


import json
import requests


# In[4]:


API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"

# def get_completion(payload='',parameters=None,options={'use_cache': False}):
def get_completion(payload='',parameters=None,options={'use_cache': False}):

    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    body = {"inputs":payload, "parameters":parameters, "options":options}
    #print("body: " + str(body))
    #print("headers: " + str(headers))
    response = requests.request("POST", API_URL, headers=headers, data= json.dumps(body))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        return "Error:"+" ".join(response.json()['error'])
    else:
        return response.json()[0]['generated_text']
    #         return response.choices[0].message["content"]

    
parameters = {
    "max_new_tokens":50,  # number of generated tokens
    "temperature": 0.5,   # controlling the randomness of generations
#      "end_sequence": "###"  # stopping sequence for generation
}

# prompt="...."             # few-shot prompt

# data = get_completion(prompt,parameters,options)


# In[5]:


import random
import evaluate
from datasets import load_dataset


# In[6]:


def get_random_prompts(dataset, num_examples=100):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return(dataset[picks])


# In[7]:


from transformers import pipeline, AutoTokenizer

text_generation = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")


# In[8]:


# bold

bold = load_dataset("AlexaAI/bold", split="train")


# In[9]:


from random import sample

asian_bold = (sample([p for p in bold if p['category'] == 'Asian_Americans'],50))
african_bold = (sample([p for p in bold if p['category'] == 'African_Americans'],50))
euro_bold = (sample([p for p in bold if p['category'] == 'European_Americans'],50))

latino_bold = ([p for p in bold if p['category'] == 'Hispanic_and_Latino_Americans'])
print(len(latino_bold))


# asian_bold[0]
# print(asian_bold)


# In[10]:


asian_prompts = [p['prompts'][0] for p in asian_bold]
african_prompts = [p['prompts'][0] for p in african_bold]
euro_prompts = [p['prompts'][0] for p in euro_bold]



first_latino_prompts = ([p['prompts'] for p in latino_bold])

og_latino_prompts = []

for i in first_latino_prompts:
    if len(i) == 1:
        og_latino_prompts.extend(i)
    else:
        for prompt in i:
            og_latino_prompts.extend([prompt])
            
# print(latino_prompts)
print(len(og_latino_prompts))

# print(len([p['prompts'] for p in latino_bold]))

latino_prompts = (sample(og_latino_prompts, 50))
print(latino_prompts)

print(len(latino_prompts))


# euro_prompts[0]

# print(len(asian_prompts))
# print(asian_prompts)

# political, religion, race in that order


# In[11]:


print(asian_prompts)
print(african_prompts)
print(euro_prompts)
print(latino_prompts)


# In[12]:


asian_continuations=[]
for prompt in asian_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    asian_continuations.append(continuation)

print('Generated '+ str(len(asian_continuations))+ ' asian american continuations')

# for i in male_continuations:
#     print(i)

for i in asian_continuations:
    print(i)
    


# In[177]:


for i in range(len(asian_prompts)):
    print(i)
    print("\n")
    print(asian_prompts[i])


# In[146]:


for i in range(len(asian_continuations)):
    print(i)
    print("\n")
    print(asian_continuations[i])


# In[169]:


print(asian_prompts[35])


# In[170]:


asian_sample = (sample([p for p in bold if p['category'] == 'Asian_Americans'],1))

asian_replacement_prompt = [p['prompts'][0] for p in asian_sample]

print(asian_replacement_prompt)

for i in range(len(asian_prompts)):
    while asian_sample == asian_prompts[i]:
        asian_sample = (sample([p for p in bold if p['category'] == 'Asian_Americans'],1))

        asian_replacement_prompt = [p['prompts'][0] for p in asian_sample]

        print(asian_replacement_prompt) 


# In[171]:


prompt = asian_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(asian_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[172]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

asian_prompts[35] = asian_replacement_prompt[0]
asian_continuations[35] = continuation

print(asian_prompts[35])
print(asian_continuations[35])


# In[173]:


# asian_continuations.pop(13)


# In[174]:


# print(asian_prompts[25])print(asian_prompts[10])
# print(asian_continuations[25])


# In[175]:


print(len(asian_prompts))
print(asian_prompts)


# In[176]:


print(len(asian_continuations))
print(asian_continuations)

# for i in range(len(asian_continuations)):
#     print(i)
#     print("\n")
#     print(asian_continuations[i])


# In[ ]:





# In[92]:





# In[ ]:


# generation1 = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
# continuation1 = generation1[0]['generated_text'].replace(prompt,'')
# # asian_continuations.insert(12, continuation)

# print(asian_prompts[12])
# # print(asian_continuations[12])
# print(continuation1)

# # for i in male_continuations:
# #     print(i)

# # for i in asian_continuations:
# #     print(i)
    

    
#     prompt = asian_prompts[12]
# generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
# continuation = generation[0]['generated_text'].replace(prompt,'')
# asian_continuations.insert(12, continuation)

# print(asian_prompts[12])
# print(asian_continuations[12])

# # for i in male_continuations:
# #     print(i)

# # for i in asian_continuations:
# #     print(i)
    


# In[15]:


african_continuations=[]
for prompt in african_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    african_continuations.append(continuation)

print('Generated '+ str(len(african_continuations))+ ' african american continuations')

# for i in female_continuations:
#     print(i)

for i in african_continuations:
    print(i)
    


# In[179]:


for i in range(len(african_prompts)):
    print(i)
    print("\n")
    print(african_prompts[i])


# In[180]:


for i in range(len(african_continuations)):
    print(i)
    print("\n")
    print(african_continuations[i])


# In[218]:


print(african_prompts[44])


# In[219]:


african_sample = (sample([p for p in bold if p['category'] == 'African_Americans'],1))

african_replacement_prompt = [p['prompts'][0] for p in african_sample]

print(african_replacement_prompt)

for i in range(len(african_prompts)):
    while african_sample == african_prompts[i]:
        african_sample = (sample([p for p in bold if p['category'] == 'African_Americans'],1))

        african_replacement_prompt = [p['prompts'][0] for p in african_sample]

        print(african_replacement_prompt) 


# In[220]:


prompt = african_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(african_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[221]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

african_prompts[44] = african_replacement_prompt[0]
african_continuations[44] = continuation

print(african_prompts[44])
print(african_continuations[44])


# In[222]:


print(len(african_prompts))
print(african_prompts)


# In[223]:


print(len(african_continuations))
print(african_continuations)

# for i in range(len(asian_continuations)):
#     print(i)
#     print("\n")
#     print(asian_continuations[i])


# In[16]:


print(african_continuations[1])


# In[17]:


euro_continuations=[]
for prompt in euro_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    euro_continuations.append(continuation)

print('Generated '+ str(len(euro_continuations))+ ' european american continuations')

# for i in female_continuations:
#     print(i)

for i in euro_continuations:
    print(i)
    


# In[269]:


print(euro_prompts[47])


# In[276]:


euro_sample = (sample([p for p in bold if p['category'] == 'European_Americans'],1))

euro_replacement_prompt = [p['prompts'][0] for p in euro_sample]

print(euro_replacement_prompt)

for i in range(len(euro_prompts)):
    while euro_sample == euro_prompts[i]:
        euro_sample = (sample([p for p in bold if p['category'] == 'European_Americans'],1))

        euro_replacement_prompt = [p['prompts'][0] for p in euro_sample]

        print(euro_replacement_prompt) 


# In[277]:


prompt = euro_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(euro_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[278]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

euro_prompts[47] = euro_replacement_prompt[0]
euro_continuations[47] = continuation

print(euro_prompts[47])
print(euro_continuations[47])


# In[279]:


print(len(euro_prompts))
print(euro_prompts)


# In[268]:


print(len(euro_continuations))
print(euro_continuations)

# for i in range(len(asian_continuations)):
#     print(i)
#     print("\n")
#     print(asian_continuations[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


latino_continuations=[]
for prompt in latino_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    latino_continuations.append(continuation)

print('Generated '+ str(len(latino_continuations))+ ' latino/hispanic american continuations')

# for i in female_continuations:
#     print(i)

for i in latino_continuations:
    print(i)
    


# In[408]:


print(latino_prompts[44])


# In[411]:


latino_sample = (sample([p for p in bold if p['category'] == 'Hispanic_and_Latino_Americans'],1))

latino_replacement_prompt = [p['prompts'][0] for p in latino_sample]

print(latino_replacement_prompt)

for i in range(len(latino_prompts)):
    while latino_sample == latino_prompts[i]:
        latino_sample = (sample([p for p in bold if p['category'] == 'Hispanic_and_Latino_Americans'],1))

        latino_replacement_prompt = [p['prompts'][0] for p in latino_sample]

        print(latino_replacement_prompt) 

        


# In[ ]:





# In[412]:


prompt = latino_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(latino_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[413]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

latino_prompts[44] = latino_replacement_prompt[0]
latino_continuations[44] = continuation

print(latino_prompts[44])
print(latino_continuations[44])


# In[419]:


print(len(latino_prompts))
print(latino_prompts)


# In[422]:


print(len(latino_continuations))
print(latino_continuations)

# for i in range(len(asian_continuations)):
#     print(i)
#     print("\n")
#     print(asian_continuations[i])


# In[ ]:





# In[ ]:





# In[423]:


# for i in female_continuations:
#     for q in range(len(female_continuations)):
#         print(q)
#         print("\n")
    
#     print(i)


# In[424]:


print(asian_prompts[12])
print(asian_continuations[12])


# In[425]:


print(african_prompts[3])
print(african_continuations[3])


# In[426]:


print(euro_prompts[3])
print(euro_continuations[3])


# In[427]:


print(latino_prompts[3])
print(latino_continuations[3])


# In[428]:


regard = evaluate.load('regard', 'compare')


# In[429]:


toxicity = evaluate.load("toxicity")


# In[430]:


asiantoxicity_ratio = toxicity.compute(predictions=asian_continuations, aggregation="ratio")
print(asiantoxicity_ratio)


# In[431]:


asiantoxicity_ratio1 = "Asian American: {}".format(asiantoxicity_ratio)
print(asiantoxicity_ratio1)


# In[432]:


asianmax_toxicity = toxicity.compute(predictions=asian_continuations, aggregation="maximum")
print(asianmax_toxicity)


# In[433]:


asianmax_toxicity1 = "Asian American: {}".format(asianmax_toxicity)
print(asianmax_toxicity1)


# In[434]:


tox_dict= {}
all_toxicity1 = toxicity.compute(predictions=asian_continuations)
for text, score in zip(asian_continuations, all_toxicity1['toxicity']):
  tox_dict[text] = score


# In[436]:


toxic_asian = all_toxicity1['toxicity']
print(toxic_asian)

print("\n")

print(asian_prompts[10])
print(asian_continuations[10])
print(toxic_asian[10])


# In[437]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)


# In[438]:


print(asian_prompts[1])
print(asian_continuations[1])


# In[439]:


# m_completions = []
# toxic_male = []

# for text, value in tox_dict.items():
#     m_completions.append(text)
#     toxic_male.append(value)

# print(m_completions)     # List of text values
# print(toxic_male)


# In[440]:


africantoxicity_ratio = toxicity.compute(predictions=african_continuations, aggregation="ratio")
print(africantoxicity_ratio)


# In[441]:


africantoxicity_ratio1 = "African American: {}".format(africantoxicity_ratio)
print(africantoxicity_ratio1)


# In[442]:


africanmax_toxicity = toxicity.compute(predictions=african_continuations, aggregation="maximum")
print(africanmax_toxicity)


# In[443]:


africanmax_toxicity1 = "African American: {}".format(africanmax_toxicity)
print(africanmax_toxicity1)


# In[444]:


tox_dict1= {}
all_toxicity2 = toxicity.compute(predictions=african_continuations)
for text, score in zip(african_continuations, all_toxicity2['toxicity']):
  tox_dict1[text] = score

tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[445]:


toxic_african = all_toxicity2['toxicity']
print(toxic_african)

print("\n")

print(african_prompts[10])
print(african_continuations[10])
print(toxic_african[10])


# In[446]:


# f_completions = []
# toxic_female = []

# for text, value in tox_dict1.items():
#     f_completions.append(text)
#     toxic_female.append(value)

# print(f_completions)     # List of text values
# print(toxic_female)


# In[447]:


eurotoxicity_ratio = toxicity.compute(predictions=euro_continuations, aggregation="ratio")
print(eurotoxicity_ratio)


# In[448]:


eurotoxicity_ratio1 = "European American: {}".format(eurotoxicity_ratio)
print(eurotoxicity_ratio1)


# In[449]:


euromax_toxicity = toxicity.compute(predictions=euro_continuations, aggregation="maximum")
print(euromax_toxicity)


# In[450]:


euromax_toxicity1 = "European American: {}".format(euromax_toxicity)
print(euromax_toxicity1)


# In[451]:


tox_dict2= {}
all_toxicity3 = toxicity.compute(predictions=euro_continuations)
for text, score in zip(euro_continuations, all_toxicity3['toxicity']):
  tox_dict2[text] = score


# In[452]:


toxic_euro = all_toxicity3['toxicity']
print(toxic_euro)

print("\n")

print(euro_prompts[10])
print(euro_continuations[10])
print(toxic_euro[10])


# In[453]:


tox_dict2 = (dict(sorted(tox_dict2.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict2)


# In[454]:


# ______________________


# In[455]:


latinotoxicity_ratio = toxicity.compute(predictions=latino_continuations, aggregation="ratio")
print(latinotoxicity_ratio)


# In[456]:


latinotoxicity_ratio1 = "Latino/Hispanic American: {}".format(latinotoxicity_ratio)
print(latinotoxicity_ratio1)


# In[457]:


latinomax_toxicity = toxicity.compute(predictions=latino_continuations, aggregation="maximum")
print(latinomax_toxicity)


# In[458]:


latinomax_toxicity1 = "Latino/Hispanic: {}".format(latinomax_toxicity)
print(latinomax_toxicity1)


# In[459]:


tox_dict3= {}
all_toxicity4 = toxicity.compute(predictions=latino_continuations)
for text, score in zip(latino_continuations, all_toxicity4['toxicity']):
  tox_dict3[text] = score


# In[460]:


toxic_latino = all_toxicity4['toxicity']
print(toxic_latino)

print("\n")

print(latino_prompts[10])
print(latino_continuations[10])
print(toxic_latino[10])


# In[461]:


tox_dict3 = (dict(sorted(tox_dict3.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict3)


# In[462]:


# __________________


# In[463]:


# all_regard = regard.compute(data = asian_continuations, references= african_continuations)
# print(all_regard)


# In[464]:


# avg_regard = regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')


# In[465]:


honest = evaluate.load('honest', 'en')


# In[466]:


groups = ['Asian Americans'] * 50 + ['African Americans'] * 50 + ['European Americans'] * 50 + ['Hispanic and Latino_Americans'] * 50
continuations = [c.split() for c in asian_continuations] + [q.split() for q in african_continuations] + [a.split() for a in euro_continuations] + [b.split() for b in latino_continuations]


# In[467]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[468]:


get_ipython().system('pip install xlsxwriter ')


# In[471]:


import xlsxwriter

file_path = 'comp_toxicrace_neo.xlsx'


workbook = xlsxwriter.Workbook('comp_toxicrace_neo.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()




# Use the worksheet object to write
# data via the write() method.
worksheet.write('A1', 'Asian American Prompt')
worksheet.write('B1', 'Asian American Continuation')
worksheet.write('C1', 'Asian American Toxicity Value')
worksheet.write('D1', 'African American Prompt')
worksheet.write('E1', 'African American Continuation')
worksheet.write('F1', 'African American Toxicity Value')
worksheet.write('G1', 'European American Prompt')
worksheet.write('H1', 'European American Continuation')
worksheet.write('I1', 'European American Toxicity Value')
worksheet.write('J1', 'Hispanic and Latino American Prompt')
worksheet.write('K1', 'Hispanic and Latino American Continuation')
worksheet.write('L1', 'Hispanic and Latino American Toxicity Value')

# Start from the first cell.
# Rows and columns are zero indexed.
row = 1
column = 0
for i in range(50) :
    worksheet.write(row, column, asian_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 1
for i in range(50) :
    worksheet.write(row, column, asian_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 2
for i in range(50) :
    worksheet.write(row, column, toxic_asian[i])  # Write data rows
    row += 1
    
# ____________________
row = 1
column = 3
for i in range(50) :
    worksheet.write(row, column, african_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 4
for i in range(50) :
    worksheet.write(row, column, african_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 5
for i in range(50) :
    worksheet.write(row, column, toxic_african[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 6
for i in range(50) :
    worksheet.write(row, column, euro_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 7
for i in range(50) :
    worksheet.write(row, column, euro_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 8
for i in range(50) :
    worksheet.write(row, column, toxic_euro[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 9
for i in range(50) :
    worksheet.write(row, column, latino_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 10
for i in range(50) :
    worksheet.write(row, column, latino_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 11
for i in range(50) :
    worksheet.write(row, column, toxic_latino[i])  # Write data rows
    row += 1
    
# ________________


worksheet.write(52, 0, asiantoxicity_ratio1)  # Write data rows
worksheet.write(52, 1, africantoxicity_ratio1)  # Write data rows
worksheet.write(52, 2, eurotoxicity_ratio1)  # Write data rows
worksheet.write(52, 3, latinotoxicity_ratio1)  # Write data rows

worksheet.write(53, 0, asianmax_toxicity1)  # Write data rows
worksheet.write(53, 1, africanmax_toxicity1)  # Write data rows
worksheet.write(53, 2, euromax_toxicity1)  # Write data rows
worksheet.write(53, 3, latinomax_toxicity1)  # Write data rows

worksheet.write(54, 0, str(honest_score))  # Write data rows

   
print(f"Excel file '{file_path}' has been created successfully.")

workbook.close()


# In[472]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'comp_toxicrace_neo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
    writer = csv.writer(csvfile)
    writer.writerow(['Asian American Prompt', 'Asian American Continuation', 'Asian American Toxicity Value','African American Prompt','African American Continuation', 'African American Toxicity Value','European American Prompt','European American Continuation', 'European American Toxicity Value','Hispanic and Latino American Prompt','Hispanic and Latino American Continuation', 'Hispanic and Latino American Toxicity Value'])  # Write header row
    
    for i in range(50):
        writer.writerow([asian_prompts[i], asian_continuations[i], toxic_asian[i], african_prompts[i], african_continuations[i], toxic_african[i], euro_prompts[i], euro_continuations[i], toxic_euro[i], latino_prompts[i], latino_continuations[i], toxic_latino[i]])  # Write data rows

    writer.writerow([asiantoxicity_ratio1, africantoxicity_ratio1, eurotoxicity_ratio1, latinotoxicity_ratio1])  # Write data rows
    
    writer.writerow([asianmax_toxicity1, africanmax_toxicity1, euromax_toxicity1, latinomax_toxicity1])  # Write data rows

#     writer.writerow([all_regard, avg_regard])  # Write data rows
    writer.writerow([honest_score])  # Write data rows

#     for toxic_prompts, completion, toxicity_val in data.items():
#         writer.writerow([key, value])

#         with open('Names.csv', 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=field_names)
#     writer.writeheader()
#     writer.writerows(cars)

print(f"CSV file '{file_path}' has been created successfully.")


# In[ ]:


regard.compute(data = male_continuations, references= female_continuations)


# In[ ]:


regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')


# In[ ]:


# for i in range(51):
#     print(i)


# In[ ]:


# # Assuming you have the male_continuations list and an empty result list
# m_continuations = []

# # Iterate over each male_continuation, print the text, and add it to the result list

# for m in range(51):
#     for i in male_continuations:
# #     print(i)
#         m_continuations.insert(m,i)



    
# for i in m_continuations:
#     for q in range(len(m_continuations)):
#         print(q)
#         print("\n")
    
#     print(i)   

# # Assuming you have the male_continuations list and an empty result list
# f_continuations = []

# # Iterate over each male_continuation, print the text, and add it to the result list
# for i in female_continuations:
# #     print(i)
#     f_continuations.append(i)


# In[ ]:


# # for i in male_continuations:
# #     print(i)

# # for i in range(1, 51):
# regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')
# print(male_continuations[50])
# print(female_continuations[50])


# for i in range(len(male_continuations)):
#     regard_dict= {}
#     all_regard = regard.compute(data = male_continuations, references= female_continuations)
#     print(all_regard)
    
    # Print the regard scores for each male_continuation

for i in range(len(m_continuations)):
    regard_scores = regard.compute(data=m_continuations[i], references=f_continuations)
    print(regard_scores)

# for male_continuation, regard_score in regard_scores.items():
#     print(f"Regard score for male_continuation '{male_continuation}': {regard_score}")



# In[ ]:


honest = evaluate.load('honest', 'en')


# In[ ]:


groups = ['male'] * 50 + ['female'] * 50
continuations = [c.split() for c in male_continuations] + [q.split() for q in female_continuations]


# In[ ]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[ ]:


# for i in male_continuations:
#     print(i)
    
    
# for i in range(1,51):
#    print(regard.compute(data=male_continuations[i], references=female_continuations[i], aggregation='average'))


# In[ ]:


# honest

honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')


# In[ ]:


from collections import Counter
categories= [h['category'] for h in honest_dataset]
Counter(categories)


# In[ ]:


queer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], 10)
nonqueer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], 10)


# In[ ]:


print(queer_prompts[2])
print(nonqueer_prompts[2])


# In[ ]:


queer_continuations=[]
for prompt in queer_prompts:
  generation = text_generation(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, num_return_sequences=k, pad_token_id=50256)
  continuation = generation[0]['generated_text'].replace(prompt,'')
  queer_continuations.append(continuation)

print('Generated '+ str(len(queer_continuations))+ ' continuations')

# for prompt in male_prompts:
#     generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
#     continuation = generation[0]['generated_text'].replace(prompt,'')
#     male_continuations.append(continuation)

# print('Generated '+ str(len(male_continuations))+ ' male continuations')

for i in queer_continuations:
    print(i)


# In[ ]:


nonqueer_continuations=[]
for prompt in nonqueer_prompts:
  generation = text_generation(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, num_return_sequences=k, pad_token_id=50256)
  continuation = generation[0]['generated_text'].replace(prompt,'')
  nonqueer_continuations.append(continuation)

print('Generated '+ str(len(nonqueer_continuations))+ ' continuations')

for i in nonqueer_continuations:
    print(i)


# In[ ]:


get_ipython().system('pip install unidecode')


# In[ ]:


honest = evaluate.load('/content/evaluate/measurements/honest', 'en')


# In[ ]:


groups = ['queer'] * 50 + ['nonqueer'] * 50
continuations = [c.split() for c in queer_continuations] + [q.split() for q in nonqueer_continuations]


# In[ ]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




