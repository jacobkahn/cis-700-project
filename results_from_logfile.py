# run awk 'f{print;f=0} /RESULTS/{f=1}' run100.log > data to get results

import ast
import json
import csv

with open('datarnn') as f:
    content = f.readlines()
content = [x.strip() for x in content]


content = [ast.literal_eval(item) for item in content]

print content

with open('rnn_results.csv', "a+") as outfile:
    writer = csv.DictWriter(outfile, content[0].keys())
    writer.writeheader()

with open('rnn_results.csv', "a+") as outfile:
    writer = csv.DictWriter(outfile, content[0].keys())
    for result in content:
        writer.writerow({k: str(v) for k,v in result.items()})
