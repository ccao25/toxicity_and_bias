#!/usr/bin/env python
# coding: utf-8

# In[1]:


# v2


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

jud_bold = ([p for p in bold if p['category'] == 'judaism'])
print(len(jud_bold))

christ_bold = ([p for p in bold if p['category'] == 'christianity'])
print(len(christ_bold))

islam_bold = ([p for p in bold if p['category'] == 'islam'])
print(len(islam_bold))

hindu_bold = ([p for p in bold if p['category'] == 'hinduism'])
print(len(hindu_bold))

bud_bold = ([p for p in bold if p['category'] == 'buddhism'])
print(len(bud_bold))

sik_bold = ([p for p in bold if p['category'] == 'sikhism'])
print(len(sik_bold))

ath_bold = ([p for p in bold if p['category'] == 'atheism'])
print(len(ath_bold))


# In[10]:


first_jud_prompts = ([p['prompts'] for p in jud_bold])
og_jud_prompts = []

for i in first_jud_prompts:
    if len(i) == 1:
        og_jud_prompts.extend(i)
    else:
        for prompt in i:
            og_jud_prompts.extend([prompt])
            
print(len(og_jud_prompts))

jud_prompts = (sample(og_jud_prompts, 50))
# print(latino_prompts)


first_christ_prompts = ([p['prompts'] for p in christ_bold])
og_christ_prompts = []

for i in first_christ_prompts:
    if len(i) == 1:
        og_christ_prompts.extend(i)
    else:
        for prompt in i:
            og_christ_prompts.extend([prompt])
            
print(len(og_christ_prompts))

christ_prompts = (sample(og_christ_prompts, 50))
# print(latino_prompts)


first_islam_prompts = ([p['prompts'] for p in islam_bold])
og_islam_prompts = []

for i in first_islam_prompts:
    if len(i) == 1:
        og_islam_prompts.extend(i)
    else:
        for prompt in i:
            og_islam_prompts.extend([prompt])
            
print(len(og_islam_prompts))

islam_prompts = (sample(og_islam_prompts, 50))
# print(latino_prompts)


first_hindu_prompts = ([p['prompts'] for p in hindu_bold])
og_hindu_prompts = []

for i in first_hindu_prompts:
    if len(i) == 1:
        og_hindu_prompts.extend(i)
    else:
        for prompt in i:
            og_hindu_prompts.extend([prompt])
            
print(len(og_hindu_prompts))

hindu_prompts = (sample(og_hindu_prompts, 12))


first_bud_prompts = ([p['prompts'] for p in bud_bold])
og_bud_prompts = []

for i in first_bud_prompts:
    if len(i) == 1:
        og_bud_prompts.extend(i)
    else:
        for prompt in i:
            og_bud_prompts.extend([prompt])
            
print(len(og_bud_prompts))

bud_prompts = (sample(og_bud_prompts, 50))
# print(latino_prompts)


first_sik_prompts = ([p['prompts'] for p in sik_bold])
og_sik_prompts = []

for i in first_sik_prompts:
    if len(i) == 1:
        og_sik_prompts.extend(i)
    else:
        for prompt in i:
            og_sik_prompts.extend([prompt])
            
print(len(og_sik_prompts))

sik_prompts = (sample(og_sik_prompts, 50))
# print(latino_prompts)


first_ath_prompts = ([p['prompts'] for p in ath_bold])
og_ath_prompts = []

for i in first_ath_prompts:
    if len(i) == 1:
        og_ath_prompts.extend(i)
    else:
        for prompt in i:
            og_ath_prompts.extend([prompt])
            
print(len(og_ath_prompts))

ath_prompts = (sample(og_ath_prompts, 29))


# In[11]:


print(jud_prompts)
print(christ_prompts)
print(islam_prompts)
print(hindu_prompts)
print(bud_prompts)
print(sik_prompts)
print(ath_prompts)


# In[ ]:





# In[12]:


# ___________________________________ here


# In[ ]:





# In[13]:


jud_continuations=[]
for prompt in jud_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    jud_continuations.append(continuation)

print('Generated '+ str(len(jud_continuations))+ ' judaism continuations')

# for i in male_continuations:
#     print(i)

for i in jud_continuations:
    print(i)
    
 


# In[117]:


print(jud_prompts[34])


# In[121]:


jud_sample = (sample([p for p in bold if p['category'] == 'judaism'],1))

jud_replacement_prompt = [p['prompts'][0] for p in jud_sample]

print(jud_replacement_prompt)

for i in range(len(jud_prompts)):
    while jud_sample == jud_prompts[i]:
        jud_sample = (sample([p for p in bold if p['category'] == 'judaism'],1))

        jud_replacement_prompt = [p['prompts'][0] for p in jud_sample]

        print(jud_replacement_prompt) 


# In[122]:


prompt = jud_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(jud_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[123]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

jud_prompts[34] = jud_replacement_prompt[0]
jud_continuations[34] = continuation

print(jud_prompts[34])
print(jud_continuations[34])


# In[124]:


print(len(jud_prompts))
print(jud_prompts)

print(len(jud_continuations))
print(jud_continuations)


# In[ ]:





# In[ ]:





# In[14]:


christ_continuations=[]
for prompt in christ_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    christ_continuations.append(continuation)

print('Generated '+ str(len(christ_continuations))+ ' christianity continuations')

# for i in female_continuations:
#     print(i)

for i in christ_continuations:
    print(i)
    


# In[166]:


print(christ_prompts[47])


# In[169]:


christ_sample = (sample([p for p in bold if p['category'] == 'christianity'],1))

christ_replacement_prompt = [p['prompts'][0] for p in christ_sample]

print(christ_replacement_prompt)

for i in range(len(christ_prompts)):
    while christ_sample == christ_prompts[i]:
        christ_sample = (sample([p for p in bold if p['category'] == 'christianity'],1))

        christ_replacement_prompt = [p['prompts'][0] for p in christ_sample]

        print(christ_replacement_prompt) 

    
  


# In[170]:


prompt = christ_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(christ_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[171]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

christ_prompts[47] = christ_replacement_prompt[0]
christ_continuations[47] = continuation

print(christ_prompts[47])
print(christ_continuations[47])


# In[172]:


print(len(christ_prompts))
print(christ_prompts)

print(len(christ_continuations))
print(christ_continuations)


# In[ ]:





# In[15]:


islam_continuations=[]
for prompt in islam_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    islam_continuations.append(continuation)

print('Generated '+ str(len(islam_continuations))+ ' islam continuations')

# for i in female_continuations:
#     print(i)

for i in islam_continuations:
    print(i)



# In[317]:


print(islam_prompts[43])


# In[325]:


# islam_sample = (sample([p for p in bold if p['category'] == 'islam'],1))

# islam_replacement_prompt = [p['prompts'][0] for p in islam_sample]

# print(islam_replacement_prompt)

islam_bold1 = ([p for p in bold if p['category'] == 'islam'])
print(len(islam_bold1))


first_islam_prompts1 = ([p['prompts'] for p in islam_bold])
og_islam_prompts1 = []

for i in first_islam_prompts1:
    if len(i) == 1:
        og_islam_prompts1.extend(i)
    else:
        for prompt in i:
            og_islam_prompts1.extend([prompt])
            
print(len(og_islam_prompts1))

islam_replacement_prompt = (sample(og_islam_prompts1, 1))
# print(latino_prompts)

print(islam_replacement_prompt)


# In[326]:


prompt = islam_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(islam_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[327]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

islam_prompts[43] = islam_replacement_prompt[0]
islam_continuations[43] = continuation

print(islam_prompts[43])
print(islam_continuations[43])


# In[328]:


print(len(islam_prompts))
print(islam_prompts)

print(len(islam_continuations))
print(islam_continuations)


# In[ ]:





# In[16]:


hindu_continuations=[]
for prompt in hindu_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    hindu_continuations.append(continuation)

print('Generated '+ str(len(hindu_continuations))+ ' hinduism continuations')

# for i in female_continuations:
#     print(i)

for i in hindu_continuations:
    print(i)
   


# In[ ]:





# In[ ]:





# In[17]:


bud_continuations=[]
for prompt in bud_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    bud_continuations.append(continuation)

print('Generated '+ str(len(bud_continuations))+ ' buddhism continuations')

# for i in male_continuations:
#     print(i)

for i in bud_continuations:
    print(i)
    
    


# In[ ]:





# In[497]:


print(bud_prompts[48])


# In[499]:


# bud_sample = (sample([p for p in bold if p['category'] == 'buddhism'],1))

# bud_replacement_prompt = [p['prompts'][0] for p in bud_sample]

# print(bud_replacement_prompt)

bud_bold1 = ([p for p in bold if p['category'] == 'buddhism'])
print(len(bud_bold1))


first_bud_prompts1 = ([p['prompts'] for p in bud_bold])
og_bud_prompts1 = []

for i in first_bud_prompts1:
    if len(i) == 1:
        og_bud_prompts1.extend(i)
    else:
        for prompt in i:
            og_bud_prompts1.extend([prompt])
            
print(len(og_bud_prompts1))

bud_replacement_prompt = (sample(og_bud_prompts1, 1))
# print(latino_prompts)

print(bud_replacement_prompt)


# In[500]:


prompt = bud_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(bud_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[501]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

bud_prompts[48] = bud_replacement_prompt[0]
bud_continuations[48] = continuation

print(bud_prompts[48])
print(bud_continuations[48])


# In[502]:


print(len(bud_prompts))
print(bud_prompts)

print(len(bud_continuations))
print(bud_continuations)


# In[ ]:





# In[18]:


sik_continuations=[]
for prompt in sik_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    sik_continuations.append(continuation)

print('Generated '+ str(len(sik_continuations))+ ' sikhism continuations')

# for i in male_continuations:
#     print(i)

for i in sik_continuations:
    print(i)
    


# In[ ]:





# In[570]:


print(sik_prompts[45])


# In[571]:


# sik_sample = (sample([p for p in bold if p['category'] == 'sikhism'],1))

# sik_replacement_prompt = [p['prompts'][0] for p in sik_sample]

# print(sik_replacement_prompt)

sik_bold1 = ([p for p in bold if p['category'] == 'sikhism'])
print(len(sik_bold1))


first_sik_prompts1 = ([p['prompts'] for p in sik_bold])
og_sik_prompts1 = []

for i in first_sik_prompts1:
    if len(i) == 1:
        og_sik_prompts1.extend(i)
    else:
        for prompt in i:
            og_sik_prompts1.extend([prompt])
            
print(len(og_sik_prompts1))

sik_replacement_prompt = (sample(og_sik_prompts1, 1))
# print(latino_prompts)

print(sik_replacement_prompt)


# In[572]:


prompt = sik_replacement_prompt[0]
generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
continuation = generation[0]['generated_text'].replace(prompt, '')

print(sik_replacement_prompt)
print(generation)
print(continuation)

# for i in male_continuations:
#     print(i)

# for i in asian_continuations:
#     print(i)
    


# In[573]:


# asian_prompts.insert(10, asian_replacement_prompt[0])
# asian_continuations.insert(10, continuation)

sik_prompts[45] = sik_replacement_prompt[0]
sik_continuations[45] = continuation

print(sik_prompts[45])
print(sik_continuations[45])


# In[574]:


print(len(sik_prompts))
print(sik_prompts)

print(len(sik_continuations))
print(sik_continuations)


# In[ ]:





# In[19]:


ath_continuations=[]
for prompt in ath_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    ath_continuations.append(continuation)

print('Generated '+ str(len(ath_continuations))+ ' atheism continuations')

# for i in male_continuations:
#     print(i)

for i in ath_continuations:
    print(i)
    


# In[604]:


print(ath_prompts[24])
# print(ath_continuations[20])


# In[605]:


ath_prompts.pop(24)
# ath_continuations.pop(20)

# print(ath_prompts[20])
# print(ath_continuations[20])


# In[606]:


print(len(ath_prompts))
print(ath_prompts)

print(len(ath_continuations))
print(ath_continuations)


# In[20]:


# for i in female_continuations:
#     for q in range(len(female_continuations)):
#         print(q)
#         print("\n")
    
#     print(i)


# In[21]:


# _____


# In[607]:


print(jud_prompts[2])
print(jud_continuations[2])


# In[608]:


print(christ_prompts[2])
print(christ_continuations[2])


# In[609]:


print(islam_prompts[2])
print(islam_continuations[2])


# In[610]:


print(hindu_prompts[2])
print(hindu_continuations[2])


# In[611]:


print(bud_prompts[2])
print(bud_continuations[2])


# In[612]:


print(sik_prompts[2])
print(sik_continuations[2])


# In[614]:


print(ath_prompts[2])
print(ath_continuations[2])


# In[615]:


regard = evaluate.load('regard', 'compare')


# In[616]:


toxicity = evaluate.load("toxicity")


# In[617]:


judtoxicity_ratio = toxicity.compute(predictions=jud_continuations, aggregation="ratio")
print(judtoxicity_ratio)


# In[618]:


judtoxicity_ratio1 = "Judaism: {}".format(judtoxicity_ratio)
print(judtoxicity_ratio1)


# In[619]:


judmax_toxicity = toxicity.compute(predictions=jud_continuations, aggregation="maximum")
print(judmax_toxicity)


# In[620]:


judmax_toxicity1 = "Judaism: {}".format(judmax_toxicity)
print(judmax_toxicity1)


# In[621]:


tox_dict= {}
all_toxicity1 = toxicity.compute(predictions=jud_continuations)
for text, score in zip(jud_continuations, all_toxicity1['toxicity']):
  tox_dict[text] = score


# In[622]:


toxic_jud = all_toxicity1['toxicity']
print(toxic_jud)

print("\n")

print(jud_prompts[10])
print(jud_continuations[10])
print(toxic_jud[10])


# In[623]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)


# In[624]:


# m_completions = []
# toxic_male = []

# for text, value in tox_dict.items():
#     m_completions.append(text)
#     toxic_male.append(value)

# print(m_completions)     # List of text values
# print(toxic_male)


# In[625]:


christtoxicity_ratio = toxicity.compute(predictions=christ_continuations, aggregation="ratio")
print(christtoxicity_ratio)


# In[626]:


christtoxicity_ratio1 = "Christianity: {}".format(christtoxicity_ratio)
print(christtoxicity_ratio1)


# In[627]:


christmax_toxicity = toxicity.compute(predictions=christ_continuations, aggregation="maximum")
print(christmax_toxicity)


# In[628]:


christmax_toxicity1 = "Christianity: {}".format(christmax_toxicity)
print(christmax_toxicity1)


# In[629]:


tox_dict1= {}
all_toxicity2 = toxicity.compute(predictions=christ_continuations)
for text, score in zip(christ_continuations, all_toxicity2['toxicity']):
  tox_dict1[text] = score

tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[630]:


toxic_christ = all_toxicity2['toxicity']
print(toxic_christ)

print("\n")

print(christ_prompts[10])
print(christ_continuations[10])
print(toxic_christ[10])


# In[631]:


# f_completions = []
# toxic_female = []

# for text, value in tox_dict1.items():
#     f_completions.append(text)
#     toxic_female.append(value)

# print(f_completions)     # List of text values
# print(toxic_female)


# In[632]:


islamtoxicity_ratio = toxicity.compute(predictions=islam_continuations, aggregation="ratio")
print(islamtoxicity_ratio)


# In[633]:


islamtoxicity_ratio1 = "Islam: {}".format(islamtoxicity_ratio)
print(islamtoxicity_ratio1)


# In[634]:


islammax_toxicity = toxicity.compute(predictions=islam_continuations, aggregation="maximum")
print(islammax_toxicity)


# In[635]:


islammax_toxicity1 = "Islam: {}".format(islammax_toxicity)
print(islammax_toxicity1)


# In[636]:


tox_dict2= {}
all_toxicity3 = toxicity.compute(predictions=islam_continuations)
for text, score in zip(islam_continuations, all_toxicity3['toxicity']):
  tox_dict2[text] = score


# In[637]:


toxic_islam = all_toxicity3['toxicity']
print(toxic_islam)

print("\n")

print(islam_prompts[10])
print(islam_continuations[10])
print(toxic_islam[10])


# In[638]:


tox_dict2 = (dict(sorted(tox_dict2.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict2)


# In[639]:


# ______________________


# In[640]:


hindutoxicity_ratio = toxicity.compute(predictions=hindu_continuations, aggregation="ratio")
print(hindutoxicity_ratio)


# In[641]:


hindutoxicity_ratio1 = "Hinduism: {}".format(hindutoxicity_ratio)
print(hindutoxicity_ratio1)


# In[642]:


hindumax_toxicity = toxicity.compute(predictions=hindu_continuations, aggregation="maximum")
print(hindumax_toxicity)


# In[643]:


hindumax_toxicity1 = "Hinduism: {}".format(hindumax_toxicity)
print(hindumax_toxicity1)


# In[644]:


tox_dict3= {}
all_toxicity4 = toxicity.compute(predictions=hindu_continuations)
for text, score in zip(hindu_continuations, all_toxicity4['toxicity']):
  tox_dict3[text] = score


# In[645]:


toxic_hindu = all_toxicity4['toxicity']
print(toxic_hindu)

print("\n")

print(hindu_prompts[10])
print(hindu_continuations[10])
print(toxic_hindu[10])


# In[646]:


tox_dict3 = (dict(sorted(tox_dict3.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict3)


# In[647]:


budtoxicity_ratio = toxicity.compute(predictions=bud_continuations, aggregation="ratio")
print(budtoxicity_ratio)

budtoxicity_ratio1 = "Buddhism: {}".format(budtoxicity_ratio)
print(budtoxicity_ratio1)

budmax_toxicity = toxicity.compute(predictions=bud_continuations, aggregation="maximum")
print(budmax_toxicity)

budmax_toxicity1 = "Buddhism: {}".format(budmax_toxicity)
print(budmax_toxicity1)


# In[648]:


tox_dict4= {}
all_toxicity5 = toxicity.compute(predictions=bud_continuations)
for text, score in zip(bud_continuations, all_toxicity5['toxicity']):
  tox_dict4[text] = score

tox_dict4 = (dict(sorted(tox_dict4.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict4)


toxic_bud = all_toxicity5['toxicity']
print(toxic_bud)

print("\n")

print(bud_prompts[10])
print(bud_continuations[10])
print(toxic_bud[10])


# In[649]:


siktoxicity_ratio = toxicity.compute(predictions=sik_continuations, aggregation="ratio")
print(siktoxicity_ratio)

siktoxicity_ratio1 = "Sikhism: {}".format(siktoxicity_ratio)
print(siktoxicity_ratio1)

sikmax_toxicity = toxicity.compute(predictions=sik_continuations, aggregation="maximum")
print(sikmax_toxicity)

sikmax_toxicity1 = "Sikhism: {}".format(sikmax_toxicity)
print(sikmax_toxicity1)


# In[650]:


tox_dict5= {}
all_toxicity6 = toxicity.compute(predictions=sik_continuations)
for text, score in zip(sik_continuations, all_toxicity6['toxicity']):
  tox_dict5[text] = score

tox_dict5 = (dict(sorted(tox_dict5.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict5)


toxic_sik = all_toxicity6['toxicity']
print(toxic_sik)

print("\n")

print(sik_prompts[10])
print(sik_continuations[10])
print(toxic_sik[10])


# In[651]:


athtoxicity_ratio = toxicity.compute(predictions=ath_continuations, aggregation="ratio")
print(athtoxicity_ratio)

athtoxicity_ratio1 = "Atheism: {}".format(athtoxicity_ratio)
print(athtoxicity_ratio1)

athmax_toxicity = toxicity.compute(predictions=ath_continuations, aggregation="maximum")
print(athmax_toxicity)

athmax_toxicity1 = "Atheism: {}".format(athmax_toxicity)
print(athmax_toxicity1)


# In[652]:


tox_dict6= {}
all_toxicity7 = toxicity.compute(predictions=ath_continuations)
for text, score in zip(ath_continuations, all_toxicity7['toxicity']):
  tox_dict6[text] = score

tox_dict6 = (dict(sorted(tox_dict6.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict6)


toxic_ath = all_toxicity7['toxicity']
print(toxic_ath)

print("\n")

print(ath_prompts[10])
print(ath_continuations[10])
print(toxic_ath[10])


# In[653]:


# __________________


# In[654]:


# all_regard = regard.compute(data = asian_continuations, references= african_continuations)
# print(all_regard)


# In[655]:


# avg_regard = regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')


# In[656]:


honest = evaluate.load('honest', 'en')


# In[657]:


groups = ['Judaism'] * 50 + ['Christianity'] * 50 + ['Islam'] * 50 + ['Hinduism'] * 50 + ['Buddhism'] * 50 + ['Sikhism'] * 50 + ['Atheism']
continuations = [c.split() for c in jud_continuations] + [q.split() for q in christ_continuations] + [a.split() for a in islam_continuations] + [b.split() for b in hindu_continuations] + [d.split() for d in bud_continuations] + [e.split() for e in sik_continuations] + [f.split() for f in ath_continuations]


# In[658]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[659]:


# _____________________________________


# In[ ]:





# In[660]:


get_ipython().system('pip install xlsxwriter ')


# In[661]:


import xlsxwriter

file_path = 'comp_toxicreligion_neo.xlsx'


workbook = xlsxwriter.Workbook('comp_toxicreligion_neo.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Judaism Prompt')
worksheet.write('B1', 'Judaism Continuation')
worksheet.write('C1', 'Judaism Toxicity Value')

worksheet.write('D1', 'Christianity Prompt')
worksheet.write('E1', 'Christianity Continuation')
worksheet.write('F1', 'Christianity Toxicity Value')

worksheet.write('G1', 'Islam Prompt')
worksheet.write('H1', 'Islam Continuation')
worksheet.write('I1', 'Islam Toxicity Value')

worksheet.write('J1', 'Hinduism Prompt')
worksheet.write('K1', 'Hinduism Continuation')
worksheet.write('L1', 'Hinduism Toxicity Value')

worksheet.write('M1', 'Buddhism Prompt')
worksheet.write('N1', 'Buddhism Continuation')
worksheet.write('O1', 'Buddhism Toxicity Value')

worksheet.write('P1', 'Sikhism Prompt')
worksheet.write('Q1', 'Sikhism Continuation')
worksheet.write('R1', 'Sikhism Toxicity Value')

worksheet.write('S1', 'Atheism Prompt')
worksheet.write('T1', 'Atheism Continuation')
worksheet.write('U1', 'Atheism Toxicity Value')


# Start from the first cell.
# Rows and columns are zero indexed.
row = 1
column = 0
for i in range(50) :
    worksheet.write(row, column, jud_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 1
for i in range(50) :
    worksheet.write(row, column, jud_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 2
for i in range(50) :
    worksheet.write(row, column, toxic_jud[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 3
for i in range(50) :
    worksheet.write(row, column, christ_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 4
for i in range(50) :
    worksheet.write(row, column, christ_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 5
for i in range(50) :
    worksheet.write(row, column, toxic_christ[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 6
for i in range(50) :
    worksheet.write(row, column, islam_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 7
for i in range(50) :
    worksheet.write(row, column, islam_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 8
for i in range(50) :
    worksheet.write(row, column, toxic_islam[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 9
for i in range(12) :
    worksheet.write(row, column, hindu_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 10
for i in range(12) :
    worksheet.write(row, column, hindu_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 11
for i in range(12) :
    worksheet.write(row, column, toxic_hindu[i])  # Write data rows
    row += 1
    

# ________________

row = 1
column = 12
for i in range(50) :
    worksheet.write(row, column, bud_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 13
for i in range(50) :
    worksheet.write(row, column, bud_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 14
for i in range(50) :
    worksheet.write(row, column, toxic_bud[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 15
for i in range(50) :
    worksheet.write(row, column, sik_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 16
for i in range(50) :
    worksheet.write(row, column, sik_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 17
for i in range(50) :
    worksheet.write(row, column, toxic_sik[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 18
for i in range(24) :
    worksheet.write(row, column, ath_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 19
for i in range(24) :
    worksheet.write(row, column, ath_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 20
for i in range(24) :
    worksheet.write(row, column, toxic_ath[i])  # Write data rows
    row += 1
    
# ________________


worksheet.write(52, 0, judtoxicity_ratio1)  # Write data rows
worksheet.write(52, 1, christtoxicity_ratio1)  # Write data rows
worksheet.write(52, 2, islamtoxicity_ratio1)  # Write data rows
worksheet.write(52, 3, hindutoxicity_ratio1)  # Write data rows
worksheet.write(52, 4, budtoxicity_ratio1)  # Write data rows
worksheet.write(52, 5, siktoxicity_ratio1)  # Write data rows
worksheet.write(52, 6, athtoxicity_ratio1)  # Write data rows

worksheet.write(53, 0, judmax_toxicity1)  # Write data rows
worksheet.write(53, 1, christmax_toxicity1)  # Write data rows
worksheet.write(53, 2, islammax_toxicity1)  # Write data rows
worksheet.write(53, 3, hindumax_toxicity1)  # Write data rows
worksheet.write(53, 4, budmax_toxicity1)  # Write data rows
worksheet.write(53, 5, sikmax_toxicity1)  # Write data rows
worksheet.write(53, 6, athmax_toxicity1)  # Write data rows

worksheet.write(54, 0, str(honest_score))  # Write data rows
   
print(f"Excel file '{file_path}' has been created successfully.")

workbook.close()


# In[662]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'comp_toxicreligion_neo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
    writer = csv.writer(csvfile)
    writer.writerow(['Judaism Prompt', 'Judaism Continuation', 'Judaism Toxicity Value','Christianity Prompt','Christianity Continuation', 'Christianity Toxicity Value','Islam Prompt','Islam Continuation', 'Islam Toxicity Value','Hinduism Prompt','Hinduism Continuation', 'Hinduism Toxicity Value', 'Buddhism Prompt', 'Buddhism Continuation', 'Buddhism Toxicity Value', 'Sikhism Prompt', 'Sikhism Continuation', 'Sikhism Toxicity Value', 'Atheism Prompt', 'Atheism Continuation', 'Atheism Toxicity Value'])  # Write header r
    


for i in range(37):
    hindu_prompts.append("")

for i in range(25):
    ath_prompts.append("")

    for i in range(50):
        writer.writerow([jud_prompts[i], jud_continuations[i], toxic_jud[i], christ_prompts[i], christ_continuations[i], toxic_christ[i], islam_prompts[i], islam_continuations[i], toxic_islam[i], hindu_prompts[i], hindu_continuations[i], toxic_hindu[i], bud_prompts[i], bud_continuations[i], toxic_bud[i], sik_prompts[i], sik_continuations[i], toxic_sik[i], ath_prompts[i], ath_continuations[i], toxic_ath[i]])  # Write data rows
   

    writer.writerow([judtoxicity_ratio1, christtoxicity_ratio1, islamtoxicity_ratio1, hindutoxicity_ratio1, budtoxicity_ratio1, siktoxicity_ratio1, athtoxicity_ratio1])  # Write data rows
    
    writer.writerow([judmax_toxicity1, christmax_toxicity1, islammax_toxicity1, hindumax_toxicity1, budmax_toxicity1, sikmax_toxicity1, athmax_toxicity1])  # Write data rows

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




