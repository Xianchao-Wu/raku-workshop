import os
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from dask.distributed import Client, LocalCluster


def main():

    cluster = LocalCluster(n_workers=8, processes=True, memory_limit='24GB')
    client = Client(cluster)

    cur_dir = os.getcwd()
    data_dir = f"{cur_dir}/"

    # Input
    dataset_dir = "./add_id/cleaned"
    exact_dedup_output_dir="./exact_dedup/data"
    fuzzy_dedup_output_dir="./fuzzy_wrapper/data"

    # Output
    dudped_output_dir = os.path.join(data_dir, "remove_duplicate/result.parquet")

    # Relevant parameters
    input_id_field = 'id'
    id_prefix = "JA_wiki"

    #!mkdir -p {dudped_output_dir}
    acmd = f"mkdir -p {dudped_output_dir}"
    print(acmd)
    os.system(acmd)

    #Load .jsonl dataset (GPUメモリが足りない場合はbackend='pandas'へ変更してください)
    input_dataset = DocumentDataset.read_json(dataset_dir, backend='cudf')

    # Load exact deduplicate result and extract list of duplicated document ID　(GPUメモリが足りない場合はbackend='pandas'へ変更してください)
    exact_duplicates = DocumentDataset.read_parquet(os.path.join(exact_dedup_output_dir, "_exact_duplicates.parquet"), backend='cudf')
    exact_docs_to_remove = exact_duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )

    # Remove the duplicated document from input dataset
    result = input_dataset.df[
        ~input_dataset.df[input_id_field].isin(exact_docs_to_remove[input_id_field].compute())
    ]

    # Loads result from fuzzy dedup wrapper
    fuzzy_duplicates = pd.read_parquet(fuzzy_dedup_output_dir)

    # Generate list of near duplicate document ID
    fuzzy_docs_to_remove = fuzzy_duplicates.drop_duplicates(subset=['group'], keep='first')

    # Remove near duplicates
    result = result[~result[input_id_field].isin(fuzzy_docs_to_remove[input_id_field])]

    # Save final result to local (backend='pandas'の場合は、write_to_filename=Trueをコメントアウトしてください)
    result.to_parquet(dudped_output_dir, write_to_filename=True)

    res = pd.read_parquet(dudped_output_dir)
    print(f"Length of duplicate removed dataset:{len(res)}")

if __name__ == "__main__":
    main()
