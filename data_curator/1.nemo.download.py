import os
import multiprocessing as mp
mp.set_start_method('fork', force=True)

from nemo_curator.download import download_wikipedia
from dask.distributed import Client, LocalCluster

def safe_dump_date(language):
    import requests
    from bs4 import BeautifulSoup

    url = f"https://dumps.wikimedia.org/{language}wiki/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    dates = sorted([
        a.get("href").strip("/")
        for a in soup.find_all("a")
        if a.get("href") and a.get("href").strip("/").isdigit()
    ])

    return dates[-1]

def main():
    cur_dir = os.getcwd()
    data_dir = cur_dir

    cluster = LocalCluster(
        n_workers=32,
        threads_per_worker=2,
        processes=True,
        memory_limit='16GB',
        nanny=False
    )
    client = Client(cluster)

    download_output_directory = os.path.join(data_dir, "wiki_downloads", "data")

    dump_date = safe_dump_date("ja")
    print(dump_date)

    res = download_wikipedia(
        download_output_directory,
        language='ja',
        dump_date="20251020", #"20240201",   # �� 固定
        url_limit=1
    ).df.compute()
    # https://dumps.wikimedia.org/jawiki/20251020

    print("Download task submitted")
    print(res)

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

