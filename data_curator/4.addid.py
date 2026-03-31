import os
import time

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset

from dask.distributed import Client, LocalCluster

def main():
    # 前の処理でclusterを落としている場合は以下をアンコメントして再度起動してください
    cluster = LocalCluster(n_workers=24, processes=True, memory_limit='24GB')
    client = Client(cluster)

    cur_dir = os.getcwd()
    data_dir = f"{cur_dir}/"

    # Input
    add_id_input_data_dir = "./language_sep/data/cleaned"

    # Output
    added_id_output_path = os.path.join(data_dir, "add_id/cleaned")

    # Format of output ID will be <prefix>_<id>, Define prefix here
    add_ID_id_prefix="JA_wiki"

    t0 = time.time()
    # Read input files
    dataset = DocumentDataset.read_json(add_id_input_data_dir,add_filename=True)

    # Run AddID() on the input dataset
    add_id = AddId(id_field='id',id_prefix=add_ID_id_prefix,start_index=0)
    id_dataset = add_id(dataset)

    # Output files
    id_dataset.to_json(added_id_output_path, write_to_filename=True)

    print(f"Time taken for add ID:{time.time()-t0}")

    client.cluster.close()
    client.shutdown()
    #cluster.close()

if __name__ == "__main__":
    main()
