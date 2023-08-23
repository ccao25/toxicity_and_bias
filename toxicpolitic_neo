!pip install datasets transformers evaluate -q
!pip install torch
!pip install xformers

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

left_bold = ([p for p in bold if p['category'] == 'left-wing'])
print(len(left_bold))

right_bold = ([p for p in bold if p['category'] == 'right-wing'])
print(len(right_bold))

comm_bold = ([p for p in bold if p['category'] == 'communism'])
print(len(comm_bold))

social_bold = (sample([p for p in bold if p['category'] == 'socialism'],50))
print(len(social_bold))
# socialism __________

demo_bold = (sample([p for p in bold if p['category'] == 'democracy'],50))
print(len(demo_bold))
# democracy_____

lib_bold = ([p for p in bold if p['category'] == 'liberalism'])
print(len(lib_bold))

pop_bold = ([p for p in bold if p['category'] == 'populism'])
print(len(pop_bold))

conserv_bold = ([p for p in bold if p['category'] == 'conservatism'])
print(len(conserv_bold))

nat_bold = (sample([p for p in bold if p['category'] == 'nationalism'],50))
print(len(nat_bold))
# nationalism_______

anarch_bold = (sample([p for p in bold if p['category'] == 'anarchism'],50))
print(len(anarch_bold))
# anarchism________

cap_bold = ([p for p in bold if p['category'] == 'capitalism'])
print(len(cap_bold))

fasc_bold = ([p for p in bold if p['category'] == 'fascism'])
print(len(fasc_bold))

# In[10]:


first_left_prompts = ([p['prompts'] for p in left_bold])
og_left_prompts = []

for i in first_left_prompts:
    if len(i) == 1:
        og_left_prompts.extend(i)
    else:
        for prompt in i:
            og_left_prompts.extend([prompt])
            
print(len(og_left_prompts))

left_prompts = (sample(og_left_prompts, 50))


first_right_prompts = ([p['prompts'] for p in right_bold])
og_right_prompts = []

for i in first_right_prompts:
    if len(i) == 1:
        og_right_prompts.extend(i)
    else:
        for prompt in i:
            og_right_prompts.extend([prompt])
            
print(len(og_right_prompts))

right_prompts = (sample(og_right_prompts, 50))



first_comm_prompts = ([p['prompts'] for p in comm_bold])
og_comm_prompts = []

for i in first_comm_prompts:
    if len(i) == 1:
        og_comm_prompts.extend(i)
    else:
        for prompt in i:
            og_comm_prompts.extend([prompt])
            
print(len(og_comm_prompts))

comm_prompts = (sample(og_comm_prompts, 50))


social_prompts = [p['prompts'][0] for p in social_bold]

demo_prompts = [p['prompts'][0] for p in demo_bold]



first_lib_prompts = ([p['prompts'] for p in lib_bold])
og_lib_prompts = []

for i in first_lib_prompts:
    if len(i) == 1:
        og_lib_prompts.extend(i)
    else:
        for prompt in i:
            og_lib_prompts.extend([prompt])
            
print(len(og_lib_prompts))

lib_prompts = (sample(og_lib_prompts, 50))


first_pop_prompts = ([p['prompts'] for p in pop_bold])
og_pop_prompts = []

for i in first_pop_prompts:
    if len(i) == 1:
        og_pop_prompts.extend(i)
    else:
        for prompt in i:
            og_pop_prompts.extend([prompt])
            
print(len(og_pop_prompts))

pop_prompts = (sample(og_pop_prompts, 50))



first_conserv_prompts = ([p['prompts'] for p in conserv_bold])
og_conserv_prompts = []

for i in first_conserv_prompts:
    if len(i) == 1:
        og_conserv_prompts.extend(i)
    else:
        for prompt in i:
            og_conserv_prompts.extend([prompt])
            
print(len(og_conserv_prompts))

conserv_prompts = (sample(og_conserv_prompts, 50))


nat_prompts = [p['prompts'][0] for p in nat_bold]

anarch_prompts = [p['prompts'][0] for p in anarch_bold]



first_cap_prompts = ([p['prompts'] for p in cap_bold])
og_cap_prompts = []

for i in first_cap_prompts:
    if len(i) == 1:
        og_cap_prompts.extend(i)
    else:
        for prompt in i:
            og_cap_prompts.extend([prompt])
            
print(len(og_cap_prompts))

cap_prompts = (sample(og_cap_prompts, 50))



first_fasc_prompts = ([p['prompts'] for p in fasc_bold])
og_fasc_prompts = []

for i in first_fasc_prompts:
    if len(i) == 1:
        og_fasc_prompts.extend(i)
    else:
        for prompt in i:
            og_fasc_prompts.extend([prompt])
            
print(len(og_fasc_prompts))

fasc_prompts = (sample(og_fasc_prompts, 50))


# In[11]:


print(left_prompts)
print(right_prompts)
print(comm_prompts)
print(social_prompts)
print(demo_prompts)
print(lib_prompts)
print(pop_prompts)
print(conserv_prompts)
print(nat_prompts)
print(anarch_prompts)
print(cap_prompts)
print(fasc_prompts)


# In[12]:


left_continuations=[]
for prompt in left_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    left_continuations.append(continuation)

print('Generated '+ str(len(left_continuations))+ ' left-wing continuations')

for i in left_continuations:
    print(i)


# In[13]:


right_continuations=[]
for prompt in right_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    right_continuations.append(continuation)

print('Generated '+ str(len(right_continuations))+ ' right-wing continuations')

for i in right_continuations:
    print(i)
    
# In[14]:


comm_continuations=[]
for prompt in comm_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    comm_continuations.append(continuation)

print('Generated '+ str(len(comm_continuations))+ ' communism continuations')

for i in comm_continuations:
    print(i)

# In[15]:


social_continuations=[]
for prompt in social_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    social_continuations.append(continuation)

print('Generated '+ str(len(social_continuations))+ ' socialism american continuations')

for i in social_continuations:
    print(i)


# In[16]:


demo_continuations=[]
for prompt in demo_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    demo_continuations.append(continuation)

print('Generated '+ str(len(demo_continuations))+ ' democracy continuations')

for i in demo_continuations:
    print(i)

# In[17]:


lib_continuations=[]
for prompt in lib_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    lib_continuations.append(continuation)

print('Generated '+ str(len(lib_continuations))+ ' liberalism continuations')

for i in lib_continuations:
    print(i)

# In[18]:


pop_continuations=[]
for prompt in pop_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    pop_continuations.append(continuation)

print('Generated '+ str(len(pop_continuations))+ ' populism continuations')

for i in pop_continuations:
    print(i)

# In[19]:


conserv_continuations=[]
for prompt in conserv_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    conserv_continuations.append(continuation)

print('Generated '+ str(len(conserv_continuations))+ ' conservatism continuations')

for i in conserv_continuations:
    print(i)
    
# In[20]:


nat_continuations=[]
for prompt in nat_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    nat_continuations.append(continuation)

print('Generated '+ str(len(nat_continuations))+ ' nationalism continuations')

for i in nat_continuations:
    print(i)
    
# In[21]:


anarch_continuations=[]
for prompt in anarch_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    anarch_continuations.append(continuation)

print('Generated '+ str(len(anarch_continuations))+ ' anarchism continuations')

for i in anarch_continuations:
    print(i)

# In[22]:


cap_continuations=[]
for prompt in cap_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    cap_continuations.append(continuation)

print('Generated '+ str(len(cap_continuations))+ ' capitalism continuations')

for i in cap_continuations:
    print(i)

# In[23]:


fasc_continuations=[]
for prompt in fasc_prompts:
    generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    fasc_continuations.append(continuation)

print('Generated '+ str(len(fasc_continuations))+ ' fascism continuations')

for i in fasc_continuations:
    print(i)

# In[891]:


print(left_prompts[2])
print(left_continuations[2])


# In[892]:


print(right_prompts[2])
print(right_continuations[2])


# In[893]:


print(comm_prompts[2])
print(comm_continuations[2])


# In[894]:


print(social_prompts[2])
print(social_continuations[2])


# In[895]:


print(demo_prompts[2])
print(demo_continuations[2])


# In[896]:


print(lib_prompts[2])
print(lib_continuations[2])


# In[897]:


print(pop_prompts[2])
print(pop_continuations[2])


# In[898]:


print(conserv_prompts[2])
print(conserv_continuations[2])


# In[899]:


print(nat_prompts[2])
print(nat_continuations[2])


# In[900]:


print(anarch_prompts[2])
print(anarch_continuations[2])


# In[901]:


print(cap_prompts[2])
print(cap_continuations[2])


# In[902]:


print(fasc_prompts[2])
print(fasc_continuations[2])


# In[903]:


regard = evaluate.load('regard', 'compare')


# In[904]:


toxicity = evaluate.load("toxicity")


# In[905]:


lefttoxicity_ratio = toxicity.compute(predictions=left_continuations, aggregation="ratio")
print(lefttoxicity_ratio)


# In[906]:


lefttoxicity_ratio1 = "Left-Wing: {}".format(lefttoxicity_ratio)
print(lefttoxicity_ratio1)


# In[907]:


leftmax_toxicity = toxicity.compute(predictions=left_continuations, aggregation="maximum")
print(leftmax_toxicity)


# In[908]:


leftmax_toxicity1 = "Left-Wing: {}".format(leftmax_toxicity)
print(leftmax_toxicity1)


# In[909]:


tox_dict= {}
all_toxicity1 = toxicity.compute(predictions=left_continuations)
for text, score in zip(left_continuations, all_toxicity1['toxicity']):
  tox_dict[text] = score


# In[910]:


toxic_left = all_toxicity1['toxicity']
print(toxic_left)

print("\n")

print(left_prompts[10])
print(left_continuations[10])
print(toxic_left[10])


# In[911]:


tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict)

# In[913]:


righttoxicity_ratio = toxicity.compute(predictions=right_continuations, aggregation="ratio")
print(righttoxicity_ratio)


# In[914]:


righttoxicity_ratio1 = "Right-Wing: {}".format(righttoxicity_ratio)
print(righttoxicity_ratio1)


# In[915]:


rightmax_toxicity = toxicity.compute(predictions=right_continuations, aggregation="maximum")
print(rightmax_toxicity)


# In[916]:


rightmax_toxicity1 = "Right-Wing: {}".format(rightmax_toxicity)
print(rightmax_toxicity1)


# In[917]:


tox_dict1= {}
all_toxicity2 = toxicity.compute(predictions=right_continuations)
for text, score in zip(right_continuations, all_toxicity2['toxicity']):
  tox_dict1[text] = score

tox_dict1 = (dict(sorted(tox_dict1.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict1)


# In[918]:


toxic_right = all_toxicity2['toxicity']
print(toxic_right)

print("\n")

print(right_prompts[10])
print(right_continuations[10])
print(toxic_right[10])


# In[920]:


commtoxicity_ratio = toxicity.compute(predictions=comm_continuations, aggregation="ratio")
print(commtoxicity_ratio)


# In[921]:


commtoxicity_ratio1 = "Communism: {}".format(commtoxicity_ratio)
print(commtoxicity_ratio1)


# In[922]:


commmax_toxicity = toxicity.compute(predictions=comm_continuations, aggregation="maximum")
print(commmax_toxicity)


# In[923]:


commmax_toxicity1 = "Communism: {}".format(commmax_toxicity)
print(commmax_toxicity1)


# In[924]:


tox_dict2= {}
all_toxicity3 = toxicity.compute(predictions=comm_continuations)
for text, score in zip(comm_continuations, all_toxicity3['toxicity']):
  tox_dict2[text] = score


# In[925]:


toxic_comm = all_toxicity3['toxicity']
print(toxic_comm)

print("\n")

print(comm_prompts[10])
print(comm_continuations[10])
print(toxic_comm[10])


# In[926]:


tox_dict2 = (dict(sorted(tox_dict2.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict2)


# In[927]:


# ______________________


# In[928]:


socialtoxicity_ratio = toxicity.compute(predictions=social_continuations, aggregation="ratio")
print(socialtoxicity_ratio)


# In[929]:


socialtoxicity_ratio1 = "Socialism: {}".format(socialtoxicity_ratio)
print(socialtoxicity_ratio1)


# In[930]:


socialmax_toxicity = toxicity.compute(predictions=social_continuations, aggregation="maximum")
print(socialmax_toxicity)


# In[931]:


socialmax_toxicity1 = "Socialism: {}".format(socialmax_toxicity)
print(socialmax_toxicity1)


# In[932]:


tox_dict3= {}
all_toxicity4 = toxicity.compute(predictions=social_continuations)
for text, score in zip(social_continuations, all_toxicity4['toxicity']):
  tox_dict3[text] = score


# In[933]:


toxic_social = all_toxicity4['toxicity']
print(toxic_social)

print("\n")

print(social_prompts[10])
print(social_continuations[10])
print(toxic_social[10])


# In[934]:


tox_dict3 = (dict(sorted(tox_dict3.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict3)


# In[935]:


demotoxicity_ratio = toxicity.compute(predictions=demo_continuations, aggregation="ratio")
print(demotoxicity_ratio)

demotoxicity_ratio1 = "Democracy: {}".format(demotoxicity_ratio)
print(demotoxicity_ratio1)

demomax_toxicity = toxicity.compute(predictions=demo_continuations, aggregation="maximum")
print(demomax_toxicity)

demomax_toxicity1 = "Democracy: {}".format(demomax_toxicity)
print(demomax_toxicity1)


# In[936]:


tox_dict4= {}
all_toxicity5 = toxicity.compute(predictions=demo_continuations)
for text, score in zip(demo_continuations, all_toxicity5['toxicity']):
  tox_dict4[text] = score

tox_dict4 = (dict(sorted(tox_dict4.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict4)


toxic_demo = all_toxicity5['toxicity']
print(toxic_demo)

print("\n")

print(demo_prompts[10])
print(demo_continuations[10])
print(toxic_demo[10])


# In[937]:


libtoxicity_ratio = toxicity.compute(predictions=lib_continuations, aggregation="ratio")
print(libtoxicity_ratio)

libtoxicity_ratio1 = "Liberalism: {}".format(libtoxicity_ratio)
print(libtoxicity_ratio1)

libmax_toxicity = toxicity.compute(predictions=lib_continuations, aggregation="maximum")
print(libmax_toxicity)

libmax_toxicity1 = "Liberalism: {}".format(libmax_toxicity)
print(libmax_toxicity1)


# In[938]:


tox_dict5= {}
all_toxicity6 = toxicity.compute(predictions=lib_continuations)
for text, score in zip(lib_continuations, all_toxicity6['toxicity']):
  tox_dict5[text] = score

tox_dict5 = (dict(sorted(tox_dict5.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict5)


toxic_lib = all_toxicity6['toxicity']
print(toxic_lib)

print("\n")

print(lib_prompts[10])
print(lib_continuations[10])
print(toxic_lib[10])


# In[939]:


poptoxicity_ratio = toxicity.compute(predictions=pop_continuations, aggregation="ratio")
print(poptoxicity_ratio)

poptoxicity_ratio1 = "Populism: {}".format(poptoxicity_ratio)
print(poptoxicity_ratio1)

popmax_toxicity = toxicity.compute(predictions=pop_continuations, aggregation="maximum")
print(popmax_toxicity)

popmax_toxicity1 = "Populism: {}".format(popmax_toxicity)
print(popmax_toxicity1)


# In[940]:


tox_dict6= {}
all_toxicity7 = toxicity.compute(predictions=pop_continuations)
for text, score in zip(pop_continuations, all_toxicity7['toxicity']):
  tox_dict6[text] = score

tox_dict6 = (dict(sorted(tox_dict6.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict6)


toxic_pop = all_toxicity7['toxicity']
print(toxic_pop)

print("\n")

print(pop_prompts[10])
print(pop_continuations[10])
print(toxic_pop[10])


# In[941]:


conservtoxicity_ratio = toxicity.compute(predictions=conserv_continuations, aggregation="ratio")
print(conservtoxicity_ratio)

conservtoxicity_ratio1 = "Conservatism: {}".format(conservtoxicity_ratio)
print(conservtoxicity_ratio1)

conservmax_toxicity = toxicity.compute(predictions=conserv_continuations, aggregation="maximum")
print(conservmax_toxicity)

conservmax_toxicity1 = "Conservatism: {}".format(conservmax_toxicity)
print(conservmax_toxicity1)


# In[942]:


tox_dict7= {}
all_toxicity8 = toxicity.compute(predictions=conserv_continuations)
for text, score in zip(conserv_continuations, all_toxicity8['toxicity']):
  tox_dict7[text] = score

tox_dict7 = (dict(sorted(tox_dict7.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict7)


toxic_conserv = all_toxicity8['toxicity']
print(toxic_conserv)

print("\n")

print(conserv_prompts[10])
print(conserv_continuations[10])
print(toxic_conserv[10])


# In[943]:


nattoxicity_ratio = toxicity.compute(predictions=nat_continuations, aggregation="ratio")
print(nattoxicity_ratio)

nattoxicity_ratio1 = "Nationalism: {}".format(nattoxicity_ratio)
print(nattoxicity_ratio1)

natmax_toxicity = toxicity.compute(predictions=nat_continuations, aggregation="maximum")
print(natmax_toxicity)

natmax_toxicity1 = "Nationalism: {}".format(natmax_toxicity)
print(natmax_toxicity1)


# In[944]:


tox_dict8= {}
all_toxicity9 = toxicity.compute(predictions=nat_continuations)
for text, score in zip(nat_continuations, all_toxicity9['toxicity']):
  tox_dict8[text] = score

tox_dict8 = (dict(sorted(tox_dict8.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict8)


toxic_nat = all_toxicity9['toxicity']
print(toxic_nat)

print("\n")

print(nat_prompts[10])
print(nat_continuations[10])
print(toxic_nat[10])


# In[945]:


anarchtoxicity_ratio = toxicity.compute(predictions=anarch_continuations, aggregation="ratio")
print(anarchtoxicity_ratio)

anarchtoxicity_ratio1 = "Anarchism: {}".format(anarchtoxicity_ratio)
print(anarchtoxicity_ratio1)

anarchmax_toxicity = toxicity.compute(predictions=anarch_continuations, aggregation="maximum")
print(anarchmax_toxicity)

anarchmax_toxicity1 = "Anarchism: {}".format(anarchmax_toxicity)
print(anarchmax_toxicity1)


# In[946]:


tox_dict9= {}
all_toxicity10 = toxicity.compute(predictions=anarch_continuations)
for text, score in zip(anarch_continuations, all_toxicity10['toxicity']):
  tox_dict9[text] = score

tox_dict9 = (dict(sorted(tox_dict9.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict9)


toxic_anarch = all_toxicity10['toxicity']
print(toxic_anarch)

print("\n")

print(anarch_prompts[10])
print(anarch_continuations[10])
print(toxic_anarch[10])


# In[947]:


captoxicity_ratio = toxicity.compute(predictions=cap_continuations, aggregation="ratio")
print(captoxicity_ratio)

captoxicity_ratio1 = "Capitalism: {}".format(captoxicity_ratio)
print(captoxicity_ratio1)

capmax_toxicity = toxicity.compute(predictions=cap_continuations, aggregation="maximum")
print(capmax_toxicity)

capmax_toxicity1 = "Capitalism: {}".format(capmax_toxicity)
print(capmax_toxicity1)


# In[948]:


tox_dict10= {}
all_toxicity11 = toxicity.compute(predictions=cap_continuations)
for text, score in zip(cap_continuations, all_toxicity11['toxicity']):
  tox_dict10[text] = score

tox_dict10 = (dict(sorted(tox_dict10.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict10)


toxic_cap = all_toxicity11['toxicity']
print(toxic_cap)

print("\n")

print(cap_prompts[10])
print(cap_continuations[10])
print(toxic_cap[10])


# In[949]:


fasctoxicity_ratio = toxicity.compute(predictions=fasc_continuations, aggregation="ratio")
print(fasctoxicity_ratio)

fasctoxicity_ratio1 = "Fascism: {}".format(fasctoxicity_ratio)
print(fasctoxicity_ratio1)

fascmax_toxicity = toxicity.compute(predictions=fasc_continuations, aggregation="maximum")
print(fascmax_toxicity)

fascmax_toxicity1 = "Fascism: {}".format(fascmax_toxicity)
print(fascmax_toxicity1)


# In[950]:


tox_dict11= {}
all_toxicity12 = toxicity.compute(predictions=fasc_continuations)
for text, score in zip(fasc_continuations, all_toxicity12['toxicity']):
  tox_dict11[text] = score

tox_dict11 = (dict(sorted(tox_dict11.items(), key=lambda item: item[1], reverse=True)))
print(tox_dict11)


toxic_fasc = all_toxicity12['toxicity']
print(toxic_fasc)

print("\n")

print(fasc_prompts[10])
print(fasc_continuations[10])
print(toxic_fasc[10])


# In[951]:


# __________________


# In[954]:


honest = evaluate.load('honest', 'en')


# In[955]:


groups = ['Left-Wing'] * 50 + ['Right-Wing'] * 50 + ['Communism'] * 50 + ['Socialism'] * 50 + ['Democracy'] * 50 + ['Liberalism'] * 50 + ['Populism'] * 50+ ['Conservatism'] * 50 + ['Nationalism'] * 50 + ['Anarchism'] * 50+ ['Capitalism'] * 50 + ['Fascism']
continuations = [c.split() for c in left_continuations] + [q.split() for q in right_continuations] + [a.split() for a in comm_continuations] + [b.split() for b in social_continuations] + [d.split() for d in demo_continuations] + [e.split() for e in lib_continuations] + [f.split() for f in pop_continuations] + [g.split() for g in conserv_continuations] + [h.split() for h in nat_continuations] + [i.split() for i in anarch_continuations] + [j.split() for j in cap_continuations] + [k.split() for k in fasc_continuations]




# In[956]:


honest_score = honest.compute(predictions=continuations, groups = groups)
print(honest_score)


# In[957]:


!pip install xlsxwriter 

# In[958]:


import xlsxwriter

file_path = 'comp_toxicpolitic_neo.xlsx'


workbook = xlsxwriter.Workbook('comp_toxicpolitic_neo.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()




# Use the worksheet object to write
# data via the write() method.

worksheet.write('A1', 'Left-Wing Prompt')
worksheet.write('B1', 'Left-Wing Continuation')
worksheet.write('C1', 'Left-Wing Toxicity Value')

worksheet.write('D1', 'Right-Wing Prompt')
worksheet.write('E1', 'Right-Wing Continuation')
worksheet.write('F1', 'Right-Wing Toxicity Value')

worksheet.write('G1', 'Communism Prompt')
worksheet.write('H1', 'Communism Continuation')
worksheet.write('I1', 'Communism Toxicity Value')

worksheet.write('J1', 'Socialism Prompt')
worksheet.write('K1', 'Socialism Continuation')
worksheet.write('L1', 'Socialism Toxicity Value')

worksheet.write('M1', 'Democracy Prompt')
worksheet.write('N1', 'Democracy Continuation')
worksheet.write('O1', 'Democracy Toxicity Value')

worksheet.write('P1', 'Liberalism Prompt')
worksheet.write('Q1', 'Liberalism Continuation')
worksheet.write('R1', 'Liberalism Toxicity Value')

worksheet.write('S1', 'Populism Prompt')
worksheet.write('T1', 'Populism Continuation')
worksheet.write('U1', 'Populism Toxicity Value')

worksheet.write('V1', 'Conservatism Prompt')
worksheet.write('W1', 'Conservatism Continuation')
worksheet.write('X1', 'Conservatism Toxicity Value')

worksheet.write('Y1', 'Nationalism Prompt')
worksheet.write('Z1', 'Nationalism Continuation')
worksheet.write('AA1', 'Nationalism Toxicity Value')

worksheet.write('AB1', 'Anarchism Prompt')
worksheet.write('AC1', 'Anarchism Continuation')
worksheet.write('AD1', 'Anarchism Toxicity Value')

worksheet.write('AE1', 'Capitalism Prompt')
worksheet.write('AF1', 'Capitalism Continuation')
worksheet.write('AG1', 'Capitalism Toxicity Value')

worksheet.write('AH1', 'Fascism Prompt')
worksheet.write('AI1', 'Fascism Continuation')
worksheet.write('AJ1', 'Fascism Toxicity Value')

# Start from the first cell.
# Rows and columns are zero indexed.
row = 1
column = 0
for i in range(50) :
    worksheet.write(row, column, left_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 1
for i in range(50) :
    worksheet.write(row, column, left_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 2
for i in range(50) :
    worksheet.write(row, column, toxic_left[i])  # Write data rows
    row += 1
    
    
# ____________________
row = 1
column = 3
for i in range(50) :
    worksheet.write(row, column, right_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 4
for i in range(50) :
    worksheet.write(row, column, right_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 5
for i in range(50) :
    worksheet.write(row, column, toxic_right[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 6
for i in range(50) :
    worksheet.write(row, column, comm_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 7
for i in range(50) :
    worksheet.write(row, column, comm_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 8
for i in range(50) :
    worksheet.write(row, column, toxic_comm[i])  # Write data rows
    row += 1

# ____________________
row = 1
column = 9
for i in range(50) :
    worksheet.write(row, column, social_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 10
for i in range(50) :
    worksheet.write(row, column, social_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 11
for i in range(50) :
    worksheet.write(row, column, toxic_social[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 12
for i in range(50) :
    worksheet.write(row, column, demo_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 13
for i in range(50) :
    worksheet.write(row, column, demo_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 14
for i in range(50) :
    worksheet.write(row, column, toxic_demo[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 15
for i in range(50) :
    worksheet.write(row, column, lib_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 16
for i in range(50) :
    worksheet.write(row, column, lib_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 17
for i in range(50) :
    worksheet.write(row, column, toxic_lib[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 18
for i in range(50) :
    worksheet.write(row, column, pop_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 19
for i in range(50) :
    worksheet.write(row, column, pop_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 20
for i in range(50) :
    worksheet.write(row, column, toxic_pop[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 21
for i in range(50) :
    worksheet.write(row, column, conserv_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 22
for i in range(50) :
    worksheet.write(row, column, conserv_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 23
for i in range(50) :
    worksheet.write(row, column, toxic_conserv[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 24
for i in range(50) :
    worksheet.write(row, column, nat_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 25
for i in range(50) :
    worksheet.write(row, column, nat_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 26
for i in range(50) :
    worksheet.write(row, column, toxic_nat[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 27
for i in range(50) :
    worksheet.write(row, column, anarch_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 28
for i in range(50) :
    worksheet.write(row, column, anarch_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 29
for i in range(50) :
    worksheet.write(row, column, toxic_anarch[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 30
for i in range(50) :
    worksheet.write(row, column, cap_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 31
for i in range(50) :
    worksheet.write(row, column, cap_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 32
for i in range(50) :
    worksheet.write(row, column, toxic_cap[i])  # Write data rows
    row += 1
    
# ________________

row = 1
column = 33
for i in range(50) :
    worksheet.write(row, column, fasc_prompts[i])  # Write data rows
    row += 1
    
row = 1
column = 34
for i in range(50) :
    worksheet.write(row, column, fasc_continuations[i])  # Write data rows
    row += 1
    
row = 1
column = 35
for i in range(50) :
    worksheet.write(row, column, toxic_fasc[i])  # Write data rows
    row += 1
    
# ________________

worksheet.write(52, 0, lefttoxicity_ratio1)  # Write data rows
worksheet.write(52, 1, righttoxicity_ratio1)  # Write data rows
worksheet.write(52, 2, commtoxicity_ratio1)  # Write data rows
worksheet.write(52, 3, socialtoxicity_ratio1)  # Write data rows
worksheet.write(52, 4, demotoxicity_ratio1)  # Write data rows
worksheet.write(52, 5, libtoxicity_ratio1)  # Write data rows
worksheet.write(52, 6, poptoxicity_ratio1)  # Write data rows
worksheet.write(52, 7, conservtoxicity_ratio1)  # Write data rows
worksheet.write(52, 8, nattoxicity_ratio1)  # Write data rows
worksheet.write(52, 9, anarchtoxicity_ratio1)  # Write data rows
worksheet.write(52, 10, captoxicity_ratio1)  # Write data rows
worksheet.write(52, 11, fasctoxicity_ratio1)  # Write data rows


worksheet.write(53, 0, leftmax_toxicity1)  # Write data rows
worksheet.write(53, 1, rightmax_toxicity1)  # Write data rows
worksheet.write(53, 2, commmax_toxicity1)  # Write data rows
worksheet.write(53, 3, socialmax_toxicity1)  # Write data rows
worksheet.write(53, 4, demomax_toxicity1)  # Write data rows
worksheet.write(53, 5, libmax_toxicity1)  # Write data rows
worksheet.write(53, 6, popmax_toxicity1)  # Write data rows
worksheet.write(53, 7, conservmax_toxicity1)  # Write data rowsworksheet.write(53, 0, asianmax_toxicity1)  # Write data rows
worksheet.write(53, 8, natmax_toxicity1)  # Write data rows
worksheet.write(53, 9, anarchmax_toxicity1)  # Write data rows
worksheet.write(53, 10, capmax_toxicity1)  # Write data rows
worksheet.write(53, 11, fascmax_toxicity1)  # Write data rows

worksheet.write(54, 0, str(honest_score))  # Write data rows
   
print(f"Excel file '{file_path}' has been created successfully.")

workbook.close()


# In[959]:


import csv


# Specify the file path where you want to save the CSV
file_path = 'comp_toxicpolitic_neo.csv'

# Open the file in write mode with newline='' to avoid extra line breaks
with open(file_path, 'w', newline='',encoding='utf-8') as csvfile:
        
    writer = csv.writer(csvfile)
    writer.writerow(['Left-Wing Prompt', 'Left-Wing Continuation', 'Left-Wing Toxicity Value','Right-Wing Prompt','Right-Wing Continuation', 'Right-Wing Toxicity Value','Communism Prompt','Communism Continuation', 'Communism Toxicity Value','Socialism Prompt','Socialism Continuation', 'Socialism Toxicity Value', 'Democracy Prompt', 'Democracy Continuation', 'Democracy Toxicity Value', 'Liberalism Prompt', 'Liberalism Continuation', 'Liberalism Toxicity Value', 'Populism Prompt', 'Populism Continuation', 'Populism Toxicity Value', 'Conservatism Prompt', 'Conservatism Continuation', 'Conservatism Toxicity Value', 'Nationalism Prompt', 'Nationalism Continuation', 'Nationalism Toxicity Value', 'Anarchism Prompt', 'Anarchism Continuation', 'Anarchism Toxicity Value', 'Capitalism Prompt', 'Capitalism Continuation', 'Capitalism Toxicity Value', 'Fascism Prompt', 'Fascism Continuation', 'Fascism Toxicity Value'])  # Write header row
    
    for i in range(50):
        writer.writerow([left_prompts[i], left_continuations[i], toxic_left[i], right_prompts[i], right_continuations[i], toxic_right[i], comm_prompts[i], comm_continuations[i], toxic_comm[i], social_prompts[i], social_continuations[i], toxic_social[i], demo_prompts[i], demo_continuations[i], toxic_demo[i], lib_prompts[i], lib_continuations[i], toxic_lib[i], pop_prompts[i], pop_continuations[i], toxic_pop[i], conserv_prompts[i], conserv_continuations[i], toxic_conserv[i], nat_prompts[i], nat_continuations[i], toxic_nat[i], anarch_prompts[i], anarch_continuations[i], toxic_anarch[i], cap_prompts[i], cap_continuations[i], toxic_cap[i], fasc_prompts[i], fasc_continuations[i], toxic_fasc[i]])  # Write data rows

    writer.writerow([lefttoxicity_ratio1, righttoxicity_ratio1, commtoxicity_ratio1, socialtoxicity_ratio1, demotoxicity_ratio1, libtoxicity_ratio1, poptoxicity_ratio1, conservtoxicity_ratio1, nattoxicity_ratio1, anarchtoxicity_ratio1, captoxicity_ratio1, fasctoxicity_ratio1])  # Write data rows
    
    writer.writerow([leftmax_toxicity1, rightmax_toxicity1, commmax_toxicity1, socialmax_toxicity1, demomax_toxicity1, libmax_toxicity1, popmax_toxicity1, conservmax_toxicity1, natmax_toxicity1, anarchmax_toxicity1, capmax_toxicity1, fascmax_toxicity1])  # Write data rows

#     writer.writerow([all_regard, avg_regard])  # Write data rows
    writer.writerow([honest_score])  # Write data rows

print(f"CSV file '{file_path}' has been created successfully.")

