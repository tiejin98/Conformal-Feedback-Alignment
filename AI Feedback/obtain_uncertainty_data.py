import json
import pickle

weight_dict = {}


with open('./Generation With CP/prediction_set_quantile0.2_threshold0.7_llama2.json', 'r') as file:
    eight_coverage = json.load(file)

with open('./Generation With CP/prediction_set_quantile0.5_threshold0.7_llama2.json', 'r') as file:
    five_coverage = json.load(file)

for gen_list in five_coverage:
    for sample in gen_list:
        weight_dict[sample] = 0.5

for gen_list in eight_coverage:
    for sample in gen_list:
        if sample not in weight_dict.keys():
            weight_dict[sample] = 0.8
        else:
            weight_dict[sample] = (0.5+0.8)/2

input_file = 'dpo_data_llama2.json'
output_file = 'dpo_data_llama2_withuncertainty.json'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line.strip())

        weight = (weight_dict.get(data["chosen"], 0) + weight_dict.get(data["rejected"], 0)) / 2
        data["weight"] = weight

        outfile.write(json.dumps(data) + '\n')


