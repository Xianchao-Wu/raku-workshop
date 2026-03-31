from nemo_curator import ScoreFilter,Modify
import time
import os
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import UnicodeReformatter

t0 = time.time()

# Define desired language
target_language = "JA"

# Output path
cur_dir = os.getcwd()

data_dir = f"{cur_dir}/"
language_base_output_path = os.path.join(data_dir,"language_sep") # /workspace/asr/brev.nemo.curator.20260324/language_sep
language_data_output_path = os.path.join(language_base_output_path,"data") # /workspace/asr/brev.nemo.curator.20260324/language_sep/data
language_separated_output_path = os.path.join(language_data_output_path,"language") # /workspace/asr/brev.nemo.curator.20260324/language_sep/data/language

lang_sep_cleaned_data_output_path = os.path.join(language_data_output_path, "cleaned")

# Read the language specific data and fix the unicode in it
lang_data_path = os.path.join(language_separated_output_path, target_language)
lang_data = DocumentDataset.read_json(lang_data_path,add_filename=True)

cleaner = Modify(UnicodeReformatter())
cleaned_data = cleaner(lang_data)

# Write the cleaned_data
cleaned_data.to_json(lang_sep_cleaned_data_output_path, write_to_filename=True)

print(f"Time taken for fixing unicode:{time.time()-t0}")

'''
root@933d37897f65:/workspace/asr/brev.nemo.curator.20260324# python 3.ftfy.py
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Reading 1 files
Writing to disk complete for 1 partitions
Time taken for fixing unicode:461.67655992507935
'''
