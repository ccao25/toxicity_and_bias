#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().system('pip install datasets transformers evaluate -q')
get_ipython().system('pip install torch')
get_ipython().system('pip install xformers')


# In[2]:


import json
import requests


# In[3]:


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


# In[4]:


import random
import evaluate
from datasets import load_dataset


# In[5]:


def get_random_prompts(dataset, num_examples=100):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    return(dataset[picks])


# In[6]:


from transformers import pipeline, AutoTokenizer

text_generation = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")


# In[7]:


# bold

bold = load_dataset("AlexaAI/bold", split="train")


# In[8]:


from random import sample
female_bold = (sample([p for p in bold if p['category'] == 'American_actresses'],50))
male_bold = (sample([p for p in bold if p['category'] == 'American_actors'],50))
female_bold[0]


# In[9]:


male_prompts = [p['prompts'][0] for p in male_bold]
female_prompts = [p['prompts'][0] for p in female_bold]
male_prompts[0]


# In[10]:


print(male_prompts)
print(female_prompts)


# In[12]:


male_continuations=[]
for prompt in male_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    male_continuations.append(continuation)

print('Generated '+ str(len(male_continuations))+ ' male continuations')

# for i in male_continuations:
#     print(i)

for i in male_continuations:
    print(i)


# In[13]:


female_continuations=[]
for prompt in female_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    female_continuations.append(continuation)

print('Generated '+ str(len(female_continuations))+ ' female continuations')

# for i in female_continuations:
#     print(i)

for i in female_continuations:
    print(i)


# In[14]:


# for i in female_continuations:
#     for q in range(len(female_continuations)):
#         print(q)
#         print("\n")
    
#     print(i)


# In[15]:


print(male_prompts[2])
print(male_continuations[2])


# In[16]:


print(female_prompts[2])
print(female_continuations[2])


# In[17]:


regard = evaluate.load('regard', 'compare')


# In[18]:


toxicity = evaluate.load("toxicity")


# In[19]:


maletoxicity_ratio = toxicity.compute(predictions=male_continuations, aggregation="ratio")
print(maletoxicity_ratio)


# In[20]:


maletoxicity_ratio1 = "Male: {}".format(maletoxicity_ratio)
print(maletoxicity_ratio1)


# In[21]:


malemax_toxicity = toxicity.compute(predictions=male_continuations, aggregation="maximum")
print(malemax_toxicity)


# In[22]:


malemax_toxicity1 = "Male: {}".format(malemax_toxicity)
print(malemax_toxicity1)


# In[23]:


tox_dict= {}
all_toxicity1 = toxicity.compute(predictions=male_continuations)
for text, score in zip(male_continuations, all_toxicity1['toxicity']):
  tox_dict[text] = score


# In[24]:


toxic_male = all_toxicity1['toxicity']
print(toxic_male)

print("\n")

print(male_prompts[10])
print(male_continuations[10])
print(toxic_male[10])


# In[25]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)


# In[26]:


print(male_prompts[1])
print(male_continuations[1])


# In[27]:


# m_completions = []
# toxic_male = []

# for text, value in tox_dict.items():
#     m_completions.append(text)
#     toxic_male.append(value)

# print(m_completions)     # List of text values
# print(toxic_male)


# In[28]:


femaletoxicity_ratio = toxicity.compute(predictions=female_continuations, aggregation="ratio")
print(femaletoxicity_ratio)


# In[29]:


femaletoxicity_ratio1 = "Female: {}".format(femaletoxicity_ratio)
print(femaletoxicity_ratio1)


# In[30]:


femalemax_toxicity = toxicity.compute(predictions=female_continuations, aggregation="maximum")
print(femalemax_toxicity)


# In[31]:


femalemax_toxicity1 = "Female: {}".format(femalemax_toxicity)
print(femalemax_toxicity1)


# In[32]:


tox_dict1= {}
all_toxicity2 = toxicity.compute(predictions=female_continuations)
for text, score in zip(female_continuations, all_toxicity2['toxicity']):
  tox_dict1[text] = score

tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[33]:


toxic_female = all_toxicity2['toxicity']
print(toxic_female)

print("\n")

print(female_prompts[10])
print(female_continuations[10])
print(toxic_female[10])


# In[34]:


# f_completions = []
# toxic_female = []

# for text, value in tox_dict1.items():
#     f_completions.append(text)
#     toxic_female.append(value)

# print(f_completions)     # List of text values
# print(toxic_female)


# In[35]:


all_regard = regard.compute(data = male_continuations, references= female_continuations)


# In[36]:


avg_regard = regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')


# In[37]:


honest = evaluate.load('honest', 'en')


# In[38]:


groups = ['male'] * 50 + ['female'] * 50
continuations = [c.split() for c in male_continuations] + [q.split() for q in female_continuations]


# In[39]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[40]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'toxicfemale_male_neo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
    writer = csv.writer(csvfile)
    writer.writerow(['Male Prompt', 'Male Continuation', 'Male Toxicity Value','Female Prompt','Female Continuation', 'Female Toxicity Value'])  # Write header row
    
    for i in range(50):
        writer.writerow([male_prompts[i], male_continuations[i], toxic_male[i], female_prompts[i], female_continuations[i], toxic_female[i]])  # Write data rows

    writer.writerow([maletoxicity_ratio1, femaletoxicity_ratio1])  # Write data rows
    
    writer.writerow([malemax_toxicity1, femalemax_toxicity1])  # Write data rows

    writer.writerow([all_regard, avg_regard])  # Write data rows
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




