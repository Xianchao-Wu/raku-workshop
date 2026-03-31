# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json
import os
import openai
import time
import re
import sys

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv('NV_API_KEY') # NOTE apply at: https://build.nvidia.com/settings/api-keys 
)
#model="deepseek-ai/deepseek-r1"
model="nvidia/nemotron-3-super-120b-a12b"
#model="meta/llama-3.1-405b-instruct"

def call_ds(prompt):
    for i in range(3):
        try:
            response = client.chat.completions.create(
                #model="deepseek-chat",
                model=model, # NOTE "deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096, # TODO up to 1,000,000 if necessary
                stream=False
            )

            #import ipdb; ipdb.set_trace()
            print('response:', response)

            out = response.choices[0].message.content
            think = response.choices[0].message.reasoning_content
            #import ipdb; ipdb.set_trace()

            return out, think, response
        except Exception as ex:
            print(f'Retry {i+1}: {ex}')
            time.sleep(2)
    print('Failed. skip: ', prompt)
    return None, None, None

question_gen_prompt = '''
You are a helpful financial assistant. Given the financial report table together with text contents that appear before or after the table, I ask your help of doing the following things:

1. focusing on the table and its before and after textual contents;
2. propose 5 questions in japanese so that when answing the questions, we need to retrieve information from the table or from the table's before/after textual contents or we need both the table's content and the table's surrounding textual contents;
3. the question shall be a digital computing problem that requires to compute common financial indicators. You are allowed to use no more than 5 operator times and the operator includes addition, subtraction, multiplication, division, greater, smaller, and exp. For the table retrieving part, you can use table-max (max value of a row or a column), table-min (min value of a row or a column), table-sum (sum up of a row or a column), and table-average (average value of a row or a column) operations. Put the question generated into a tag pair such as <question> generated question </question>.
4. Generate the detailed step-by-step reasoning solution (or, chain-of-thought) of answering the questions generated as well. The reasoning steps shall be put into a tag pair of <think> reasoning steps </think>.
5. Provide the reference answer to each question. The reference answer shall be put into a tag pair of <answer> the correct answer </answer>.
6. The question, the reasoning chain-of-thought, and the answer shall all in Japanese langauge. The answer shall be a digital number.
7. Try your best to ensure that the question, the chain-of-thought reasoning and the final answer are correct and align with each other. Provide with me a float confidence score with six levels that reflects the general difficulty of correctly answering the question of from 0 to 1 where the values are 0, 0.2, 0.4, 0.6, 0.8, and 1.0. Here, larger scores stand for more difficult problems. Put your score into a tag pair of <difficulty> score </difficulty>. Try you best to generate relatively difficult questions that requires longer reasoning steps.
8. Also check the table's content if it contains enough number or digital numbers for the computing. If you can not find 5 or more digital numbers, you can skip the question generation process since the given contents is not suitable for the question generation task.

The tables and contents are:
{contents}
'''

def loadtxt(atxtfn):
    outlist = list()
    with open(atxtfn) as br:
        for aline in br.readlines():
            outlist.append(aline)
    return '\n'.join(outlist)

def proc1fn(contents, aoutfn):
    #contents = loadtxt(atxtfn)
    #import ipdb; ipdb.set_trace()
    aprompt = question_gen_prompt.format(contents=contents)

    outans, think, response = call_ds(aprompt)
    if outans is not None and len(outans) > 0:
        with open(aoutfn, 'w', encoding='utf8') as bw:
           outdict = dict() 
           outdict['prompt'] = aprompt
           outdict['outans'] = outans

           outline = json.dumps(outdict, ensure_ascii=False)
           bw.write(outline + '\n')

           # to stdout 
           print(json.dumps(outdict, indent=4, ensure_ascii=False))

def get_table_num(cols):
    count = 0
    for col in cols:
        if 'TABLE FOUND' in col:
            count += 1
    return count

def isok_for_qgen(alllines):
    cols = alllines.split('\n')
    table_num = get_table_num(cols)
    if table_num == 0 or table_num > 2:
        return False

    wordnum = len(alllines.split())
    if wordnum > 100000:
        return False
    return True

def process_txt_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):
                fullfile = os.path.join(root, file)
                process_text_1file(fullfile)

def process_txt_1file(fullfile):
    aoutfile = fullfile + ".q.jsonl"
    if os.path.exists(aoutfile):
        print('processed, skip:', fullfile)
        return
    alllines = loadtxt(fullfile)
    if isok_for_qgen(alllines):
        proc1fn(alllines, aoutfile)

def main():
    atxtfn = '0105010_honbun_jpcrp030000-asr-001_E02687-000_2015-03-31_03_2016-06-01_ixbrl.htm.txt'
    process_txt_1file(atxtfn)
    # output file is:
    # 0105010_honbun_jpcrp030000-asr-001_E02687-000_2015-03-31_03_2016-06-01_ixbrl.htm.txt.q.jsonl

if __name__ == "__main__":
    main()

