import os
import time
from dask.distributed import Client, LocalCluster

from nemo_curator import ScoreFilter,Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextLangId
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.utils.file_utils import separate_by_metadata

def main():

    cur_dir = os.getcwd()
    print(cur_dir)
    data_dir = f"{cur_dir}/"

    # 前の処理でclusterを落としている場合は以下をアンコメントして再度起動してください
    cluster = LocalCluster(n_workers=48, processes=True, memory_limit='24GB')
    client = Client(cluster)

    # Input path
    #multilingual_data_path = "./wiki_downloads/data/jawiki-20240801-pages-articles-multistream1.xml-p1p114794.bz2.jsonl"
    multilingual_data_path = "./wiki_downloads/data/jawiki-20251020-pages-articles-multistream1.xml-p1p114794.bz2.jsonl"

    # Output path
    language_base_output_path = os.path.join(data_dir,"language_sep") # /workspace/asr/brev.nemo.curator.20260324/language_sep
    language_data_output_path = os.path.join(language_base_output_path,"data") # /workspace/asr/brev.nemo.curator.20260324/language_sep/data
    language_separated_output_path = os.path.join(language_data_output_path,"language") # /workspace/asr/brev.nemo.curator.20260324/language_sep/data/language

    # Fasttext model path
    model_path = language_base_output_path # '/workspace/asr/brev.nemo.curator.20260324/language_sep'

    # Define key in output .jsonl files to store the language information
    language_field = "language"

    #!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P {model_path}
    t0 = time.time()

    # Load dataset 
    multilingual_dataset = DocumentDataset.read_json(multilingual_data_path, add_filename=True) # <nemo_curator.datasets.doc_dataset.DocumentDataset object at 0x7fdbb2ad42e0>

    # Define Language separation pipeline
    lang_filter = FastTextLangId(os.path.join(model_path,'lid.176.bin')) # <nemo_curator.filters.classifier_filter.FastTextLangId object at 0x7fd7e0d947c0>
    language_id_pipeline = ScoreFilter(lang_filter, score_field=language_field, score_type='object')

    # TODO 这个不work, 没有识别出来具体的lang；之前输入的时候，是"language":"ja"; 现在经过了 fast text 之后，language="N/A"了，尴尬了属于是
    #import ipdb; ipdb.set_trace()
    filtered_dataset = language_id_pipeline(multilingual_dataset)

    # The language separation pipeline will produce a result looks like ['JA',0.96873], we only want to keep the 'JA' label and drop the detailed classifier score
    filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(lambda score: score[1],meta = (language_field, 'object'))

    # Split the dataset to corresponding language sub-folders
    language_stats = separate_by_metadata(filtered_dataset.df, language_separated_output_path, metadata_field=language_field).compute()

    print(f"Time taken for splitting language:{time.time()-t0}")

    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
