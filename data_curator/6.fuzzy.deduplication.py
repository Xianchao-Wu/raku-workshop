#%env DASK_DATAFRAME__QUERY_PLANNING=False

import os
import time
import pandas as pd

from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

import dask

def main():
    def pre_imports():
        import cudf 
        
    cur_dir = os.getcwd()
    data_dir = f"{cur_dir}/"

    client = get_client(cluster_type='gpu', set_torch_to_use_rmm=False)
    print(f"Number of dask worker:{get_num_workers(client)}")
    client.run(pre_imports)

    # Input
    fuzzy_dedup_data_path = "./add_id/cleaned"

    # Output
    fuzzy_dedup_base_output_path = os.path.join(data_dir, "fuzzy_wrapper")
    fuzzy_dedup_log_dir = os.path.join(fuzzy_dedup_base_output_path, 'log')
    fuzzy_dedup_cache_dir = os.path.join(fuzzy_dedup_base_output_path, 'cache')
    fuzzy_dedup_output_dir = os.path.join(fuzzy_dedup_base_output_path, 'data')

    # Relevant parameters
    id_field = 'id'
    text_field = 'text'
    filetype = "parquet"

    cmd5 = f"rm -r {fuzzy_dedup_cache_dir}"
    print(cmd5)
    os.system(cmd5)

    #!mkdir -p {fuzzy_dedup_base_output_path}
    #!mkdir -p {fuzzy_dedup_log_dir}
    #!mkdir -p {fuzzy_dedup_cache_dir}
    #!mkdir -p {fuzzy_dedup_output_dir}
    cmd1 = f"mkdir -p {fuzzy_dedup_base_output_path}"
    cmd2 = f"mkdir -p {fuzzy_dedup_log_dir}"
    cmd3 = f"mkdir -p {fuzzy_dedup_cache_dir}"
    cmd4 = f"mkdir -p {fuzzy_dedup_output_dir}"
    for acmd in [cmd1, cmd2, cmd3, cmd4]:
        print(acmd)
        os.system(acmd)

    #cmd5 = "rm -r {fuzzy_dedup_cache_dir}"
    #os.system(cmd5)

    with dask.config.set({"dataframe.backend": 'cudf'}):
            
        t0 = time.time()
            
        input_dataset = DocumentDataset.read_json(fuzzy_dedup_data_path, backend='cudf')
        fuzzy_dedup_config = FuzzyDuplicatesConfig(
            cache_dir=fuzzy_dedup_cache_dir,
            id_field=id_field,
            text_field=text_field,
            seed=42,  # Use the seed set in Minhash section for consistency
            char_ngrams=5,
            num_buckets=20,
            hashes_per_bucket=13,
            use_64_bit_hash=False,
            buckets_per_shuffle=5,
            false_positive_check=True,
            num_anchors=2,
            jaccard_threshold=0.8,
        )
        fuzzy_dup = FuzzyDuplicates(logger=fuzzy_dedup_log_dir, config=fuzzy_dedup_config)
        duplicates = fuzzy_dup(dataset=input_dataset)
            
        duplicates.to_parquet(fuzzy_dedup_output_dir, write_to_filename=False)
           
        print(f"Time taken for Connected Component: {time.time()-t0} s")    
            
    fuzzy_dedup_res = pd.read_parquet(fuzzy_dedup_output_dir)
    fuzzy_dedup_res.head()

if __name__ == "__main__":
    main()

