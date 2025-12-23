
from tqdm import tqdm
import os


samples = []
with open("./becnhmark results/data.json", "r") as f:
    for line in f:
        samples.append(eval(line.strip()))

def read_txt(path):
    with open(path, "r") as f:
        return f.read()

save_path = "./becnhmark results"
#model = "full_scientist" # scientist with five steps reflection and plan step
#model = "react" # scientist with ReAct framewor, without reflection or plan step
#model = "reflect1" #scientist with with one step reflection, without plan step
#model = "reflect" #scientist with with five steps reflection, without plan step
model = "full_scientist1" # scientist with with one step reflection and plan step
#model = "plan" # scientist with plan step, without reflection
#model="gpt-4.1-mini-2025-04-14" # LLM only baseline



results = []
with open(os.path.join(save_path, model, "results.json"), "r") as f:
    for line in f:
        results += eval(line.strip())

costs = []
time_cost = []

id = 0
for sample in tqdm(samples):
    result = []
    if len(sample["questions"]) > 0:
        predicts = []
        with open(os.path.join(save_path, model, sample['id']+".json"), "r") as f:
            for line in f:
                pre = eval(line.strip())
                predicts.append(pre)
                costs.append(pre['cost'])
                time_cost.append(pre['time'])
    id += 1




results_c = []
for i, result in enumerate(results):
    if "true" in result.lower():
        results_c.append(True)
    else:
        results_c.append(False)
    # if i>=11:
    #     break

idx = 0
score4cha = []

for i, sample in enumerate(samples):
    if len(sample["questions"]) > 0:
        score_ = sum(results_c[idx:idx+len(sample["questions"])]) / len(sample["questions"])
        idx += len(sample["questions"])
        score4cha.append(score_)

acc = sum(results_c) / len(results_c)
print(f"Accuracy for all the {len(results_c)} questions is {acc}")
print(f"Cost for all the {len(results_c)} questions is {sum(costs)}")
print(f"Consume time for all the {len(results_c)} questions is {sum(time_cost)}")
print()


print(f"Accuracy for each challenge is {score4cha}")
print(f"Average accuracy for {len(score4cha)} challenge is {sum(score4cha)/len(score4cha)}")
