#!/usr/bin/env python
# coding: utf-8

# In[1]:


# non_queerneo
# main____


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


# honest

honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')


# In[9]:


from collections import Counter
from random import sample

categories= [h['category'] for h in honest_dataset]
Counter(categories)


# In[10]:


queer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], 50)
nonqueer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], 50)


# In[11]:


print(queer_prompts)
print("\n")
print(nonqueer_prompts)


# In[12]:


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")


# In[13]:


k=20


# In[14]:


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


# In[15]:


nonqueer_continuations=[]
for prompt in nonqueer_prompts:
  generation = text_generation(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, num_return_sequences=k, pad_token_id=50256)
  continuation = generation[0]['generated_text'].replace(prompt,'')
  nonqueer_continuations.append(continuation)

print('Generated '+ str(len(nonqueer_continuations))+ ' continuations')

for i in nonqueer_continuations:
    print(i)


# In[16]:


regard = evaluate.load('regard', 'compare')


# In[17]:


toxicity = evaluate.load("toxicity")


# In[18]:


queertoxicity_ratio = toxicity.compute(predictions=queer_continuations, aggregation="ratio")
print(queertoxicity_ratio)


# In[19]:


queertoxicity_ratio1 = "Queer: {}".format(queertoxicity_ratio)
print(queertoxicity_ratio1)


# In[20]:


queermax_toxicity = toxicity.compute(predictions=queer_continuations, aggregation="maximum")
print(queermax_toxicity)


# In[21]:


queermax_toxicity1 = "Queer: {}".format(queermax_toxicity)
print(queermax_toxicity1)


# In[22]:


tox_dict= {}
all_toxicity1 = toxicity.compute(predictions=queer_continuations)
for text, score in zip(queer_continuations, all_toxicity1['toxicity']):
  tox_dict[text] = score


# In[23]:


toxic_queer = all_toxicity1['toxicity']
print(toxic_queer)

print("\n")

print(queer_prompts[10])
print(queer_continuations[10])
print(toxic_queer[10])


# In[24]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)


# In[25]:


nonqueertoxicity_ratio = toxicity.compute(predictions=nonqueer_continuations, aggregation="ratio")
print(nonqueertoxicity_ratio)


# In[26]:


nonqueertoxicity_ratio1 = "Nonqueer: {}".format(nonqueertoxicity_ratio)
print(nonqueertoxicity_ratio1)


# In[27]:


nonqueermax_toxicity = toxicity.compute(predictions=nonqueer_continuations, aggregation="maximum")
print(nonqueermax_toxicity)


# In[28]:


nonqueermax_toxicity1 = "Nonqueer: {}".format(nonqueermax_toxicity)
print(nonqueermax_toxicity1)


# In[29]:


tox_dict1= {}
all_toxicity2 = toxicity.compute(predictions=nonqueer_continuations)
for text, score in zip(nonqueer_continuations, all_toxicity2['toxicity']):
  tox_dict1[text] = score


# In[30]:


toxic_nonqueer = all_toxicity2['toxicity']
print(toxic_nonqueer)

print("\n")

print(nonqueer_prompts[10])
print(nonqueer_continuations[10])
print(toxic_nonqueer[10])


# In[31]:


tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[32]:


all_regard = regard.compute(data = queer_continuations, references= nonqueer_continuations)


# In[33]:


avg_regard = regard.compute(data = queer_continuations, references= nonqueer_continuations, aggregation = 'average')


# In[34]:


honest = evaluate.load('honest', 'en')


# In[35]:


groups = ['queer'] * 50 + ['nonqueer'] * 550
continuations = [c.split() for c in queer_continuations] + [q.split() for q in nonqueer_continuations]


# In[36]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[37]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'fintoxicnon_queer_neo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
     
    writer = csv.writer(csvfile)
    writer.writerow(['Queer Prompt', 'Queer Continuation', 'Queer Toxicity Value','Nonqueer Prompt','Nonqueer Continuation', 'Nonqueer Toxicity Value'])  # Write header row
    
    for i in range(50):
        writer.writerow([queer_prompts[i], queer_continuations[i], toxic_queer[i], nonqueer_prompts[i], nonqueer_continuations[i], toxic_nonqueer[i]])  # Write data rows


    writer.writerow([queertoxicity_ratio1, nonqueertoxicity_ratio1])  # Write data rows
    
    writer.writerow([queermax_toxicity1, nonqueermax_toxicity1])  # Write data rows

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


get_ipython().system('pip install unidecode')


# In[ ]:





# In[ ]:


honest = evaluate.load('honest', 'en')


# In[ ]:


groups = ['queer'] * 50 + ['nonqueer'] * 50
continuations = [c.split() for c in queer_continuations] + [q.split() for q in nonqueer_continuations]


# In[ ]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[ ]:





# In[ ]:





# In[ ]:




