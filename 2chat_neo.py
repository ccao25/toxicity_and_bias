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

toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")


# In[5]:


print(toxicity_prompts[10]['prompt'])


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


toxic_sample= get_random_prompts(toxicity_prompts)
toxic_prompts = [p['text'] for p in toxic_sample['prompt']]

for i in toxic_prompts:
    print(i)


# In[8]:


from transformers import pipeline, AutoTokenizer

text_generation = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")


# In[9]:


model_continuations=[]
for prompt in toxic_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    model_continuations.append(continuation)

print('Generated '+ str(len(model_continuations))+ ' continuations')

for i in model_continuations:
    print(i)


# In[10]:


print(toxic_prompts[1:3])


# In[11]:


print(model_continuations[1:3])


# In[12]:


print(toxic_prompts[7])
print(model_continuations[7])


# In[ ]:





# In[13]:


toxicity = evaluate.load("toxicity")


# In[14]:


toxicity_ratio = toxicity.compute(predictions=model_continuations, aggregation="ratio")
print(toxicity_ratio)


# In[15]:


max_toxicity = toxicity.compute(predictions=model_continuations, aggregation="maximum")
print(max_toxicity)


# In[16]:


tox_dict= {}
all_toxicity = toxicity.compute(predictions=model_continuations)
for text, score in zip(model_continuations, all_toxicity['toxicity']):
  tox_dict[text] = score


# In[17]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)


# In[20]:



tox_dict1= {}
all_toxicity = toxicity.compute(predictions=model_continuations)
for text, score in zip(model_continuations, all_toxicity['toxicity']):
  tox_dict1[text] = score
tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[21]:


toxic_val = all_toxicity['toxicity']
print(toxic_val)

print("\n")

print(toxic_prompts[10])
print(model_continuations[10])
print(toxic_val[10])


# In[22]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'toxic_mainneo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
    writer = csv.writer(csvfile)
    writer.writerow(['Toxic Prompt', 'Continuation', 'Toxicity Value'])  # Write header row
    
    for i in range(100):
        writer.writerow([toxic_prompts[i], model_continuations[i], toxic_val[i]])  # Write data rows

    writer.writerow([toxicity_ratio, max_toxicity])  # Write data rows

#     for toxic_prompts, completion, toxicity_val in data.items():
#         writer.writerow([key, value])

#         with open('Names.csv', 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=field_names)
#     writer.writeheader()
#     writer.writerows(cars)

print(f"CSV file '{file_path}' has been created successfully.")


# In[ ]:


# bold

bold = load_dataset("AlexaAI/bold", split="train")


# In[ ]:


from random import sample
female_bold = (sample([p for p in bold if p['category'] == 'American_actresses'],50))
male_bold = (sample([p for p in bold if p['category'] == 'American_actors'],50))
female_bold[0]


# In[ ]:


male_prompts = [p['prompts'][0] for p in male_bold]
female_prompts = [p['prompts'][0] for p in female_bold]
male_prompts[0]


# In[ ]:


male_continuations=[]
for prompt in male_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    male_continuations.append(continuation)

print('Generated '+ str(len(male_continuations))+ ' male continuations')

# for i in male_continuations:
#     print(i)


# In[ ]:


female_continuations=[]
for prompt in female_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    female_continuations.append(continuation)

print('Generated '+ str(len(female_continuations))+ ' female continuations')

# for i in female_continuations:
#     print(i)


# In[ ]:


print(male_prompts[2])
print(male_continuations[2])


# In[ ]:


print(female_prompts[2])
print(female_continuations[2])


# In[ ]:


regard = evaluate.load('regard', 'compare')


# In[ ]:


regard.compute(data = male_continuations, references= female_continuations)


# In[ ]:


regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')


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




