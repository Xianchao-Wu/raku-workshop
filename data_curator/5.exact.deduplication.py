#%env DASK_DATAFRAME__QUERY_PLANNING=False

import os
import time
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

def main():

    def pre_imports():
        import cudf 


    cur_dir = os.getcwd()
    data_dir = f"{cur_dir}/"


    client = get_client(cluster_type='gpu', set_torch_to_use_rmm=False)
    print(f"Number of dask worker:{get_num_workers(client)}")
    client.run(pre_imports)

    # Input
    exact_dedup_input_dataset_dir = "./add_id/cleaned"

    # Output
    exact_dedup_base_output_path = os.path.join(data_dir, "exact_dedup")
    exact_dedup_log_dir = os.path.join(exact_dedup_base_output_path, 'log')
    exact_dedup_output_dir = os.path.join(exact_dedup_base_output_path, 'data')

    # Parameters for ExactDuplicates()
    exact_dedup_dataset_id_field = "id"
    exact_dedup_dataset_text_field = "text"

    #!mkdir -p {exact_dedup_log_dir}
    #!mkdir -p {exact_dedup_output_dir}

    t0 = time.time()
    # Read input dataset
    input_dataset = DocumentDataset.read_json(exact_dedup_input_dataset_dir, backend='cudf')

    # Run exact deduplication to the input
    exact_dup = ExactDuplicates(
        logger=exact_dedup_log_dir,
        id_field=exact_dedup_dataset_id_field,
        text_field=exact_dedup_dataset_text_field,
        hash_method="md5",
        cache_dir=exact_dedup_output_dir  # Duplicated document ID list is output to the cache_dir
    )
    duplicates = exact_dup(dataset=input_dataset)

    print(f"Number of exact duplicated file:{len(duplicates)}")
    print(f"Time taken for exact duplicate:{time.time()-t0}")

    exact_dedup_res = pd.read_parquet(os.path.join(exact_dedup_output_dir, "_exact_duplicates.parquet"))
    print(f"Number of exact duplicated document:{len(exact_dedup_res)}")
    exact_dedup_res.head()

    exact_dedup_res.groupby('_hashes')['id'].agg(lambda x: ' '.join(x)).reset_index().head()

if __name__ == "__main__":
    main()
