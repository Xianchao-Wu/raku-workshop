# 📘 Building Japanese Datasets with NeMo Curator

> Based on NVIDIA Technical Blog  
> Source: https://developer.nvidia.com/ja-jp/blog/curating-japanese-data-using-nemo-curator/

---

## 🚀 Overview

This guide explains how to use **NeMo Curator** to build high-quality Japanese datasets for training and fine-tuning large language models (LLMs).

NeMo Curator is a:

- ⚡ High-performance (GPU-accelerated)
- 📦 Scalable (distributed processing)
- 🧩 Modular (pipeline-based)

data curation framework.

### Use Cases

- LLM Pretraining  
- Domain-Adaptive Pretraining (DAPT)  
- Supervised Fine-Tuning (SFT)  
- Preference Optimization / RLHF  

---

## 🧠 What is Data Curation?

Data curation includes:

- Data acquisition  
- Text extraction  
- Cleaning  
- Deduplication  
- Filtering  

👉 It is one of the **most critical steps** in LLM training pipelines.

---

## 🧩 What Can NeMo Curator Do?

Key features:

- 📥 Data ingestion (Common Crawl, Wikipedia, etc.)
- 🌐 Language identification
- 🧹 Text cleaning and normalization
- 🎯 Quality filtering
- 🔁 Deduplication (exact + fuzzy)
- 🔐 PII (personally identifiable information) removal
- 🧠 Data classification (domain / quality)

👉 Supports GPU acceleration and distributed processing via Dask.

---

## 🏗️ Pipeline Overview

```text
Download → Extract → Language Filter → Format → Add ID → Deduplicate → Synthetic Data
```

## ⚙️ Environment Setup

### 🖥️ Recommended Hardware

- GPU: NVIDIA A100  
- CPU: Multi-core (e.g., 64 cores)  
- RAM: ≥ 128GB  

### 🧪 Software

- Ubuntu 22.04  
- Docker  
- NeMo container:

```bash
nvcr.io/nvidia/nemo:24.07

mkdir curator-example
cd curator-example

sudo docker run --rm -it --gpus all \
  --ulimit memlock=-1 \
  --network=host \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:24.07 bash
```

## Step 1: Download Data (Wikipedia)

As mentioned earlier, NeMo Curator provides dedicated functions for Common Crawl, Wikipedia, and arXiv, allowing you to immediately begin downloading and extracting text by providing arguments.

(Optional): You can also download and extract text from your own data sources. This requires defining a download class inheriting from NeMo Curator's DocumentDownloader (the same class used in the dedicated functions for Common Crawl, etc.), an iteration class inheriting from DocumentIterator, and a text extraction class inheriting from DocumentExtractor. Examples of these can be found in "Curating Custom Datasets for LLM Parameter-Efficient Fine-Tuning with NVIDIA NeMo Curator" and "Curating Custom Datasets for LLM Training with NVIDIA NeMo Curator," so please refer to them.

This tutorial will download data from the Japanese Wikipedia and extract text from it.

When targeting the Japanese Wikipedia, use the pre-defined function `download_wikipedia()`, specifying Japanese with `language='ja'` and the date of the snapshot to download with `dump_date`. Here, to save time and resources, we set `url_limit` to limit the number of files downloaded (comment this out if you want to download all files).

Adjust the number of workers and memory limits in the `LocalCluster` arguments to suit your environment. In our test environment and with the following settings, this process took approximately 2 hours (this step is the most time-consuming part of this tutorial).

The following scripts after container startup assume cell execution in Jupyter Notebook. Also, each step can be executed individually by changing the path if the dataset used as input already exists.

```bash
python 1.nemo.download.py
```
Once processing is complete, a file named `jawiki-20251020-pages-articles-multistream1.xml-p1p114794.bz2.jsonl` will be output to the directory wiki_downloads/data/. This file contains 59,609 documents.


## Step 2: Language Detection

In this section, we classify the documents we extracted earlier by language using fasttext's language identification model. This sorts the documents into subfolders created for each language.
```bash
python 2.fasttext.py
```

Once this process is complete (1-2 minutes in the testing environment), the documents will be saved under language_sep/data/language/, categorized by language.

## Step 3: Unicode Processing

Next, the Unicode in the Japanese documents, which were previously sorted by language, is reformatted using UnicodeReformatter (which internally runs ftfy).

```bash
python 3.ftfy.py
# python 3.ftfy.nthreads.py
```

Once this process is complete (approximately 7 minutes in the test environment), a file named `jawiki-20251020-pages-articles-multistream1.xml-p1p114794.bz2.jsonl` will be output to `language_sep/data/cleaned/`. This file contains 59,570 documents.

## Step 4: Adding IDs

Japanese Wikipedia data already has IDs, but assigning a unified ID format like <prefix>_<id> makes it easier to identify which document in which dataset has been deleted when working with multiple datasets. This is useful when performing tasks such as deduplication and filtering.

The function that performs this process is AddID(). The arguments for this function are as follows:
```text
id_field: The field is added to the input JSON file.
    If the key already exists in the JSON file, its value is replaced.

id_prefix: The prefix used for the ID. The default is "doc_id".

start_index: The starting index for the ID. The default is "None".
    Setting it to "None" uses an unordered ID scheme for faster calculation.
    Here, it is set to "0" for easier referencing.
```

```bash
python 4.addid.py
```


## Step 5: Deduplication
Deduplication supports Exact Deduplication, Fuzzy Deduplication, and Semantic Deduplication using embeddings. This section covers Exact Deduplication and Fuzzy Deduplication. For information on Semantic Deduplication, see here.

Note: To perform these operations, the IDs in the corpus must be unique, for example, by using AddID() in the previous section.

### Step 5.1: Exact Deduplication
Exact Deduplication hashes the document text into a unique string using a specific hash algorithm, such as "md5". Documents with strictly identical hash values ​​have identical text.

The function used here is ExactDuplicates(). This function has the following arguments:

```text
id_field: The key in the input file to identify the document ID
text_field: The key in the input file containing the document text
hash_method: The hash algorithm to use. The default is md5.
cache_dir: If specified, duplicate document IDs are output to cache_dir. If not specified, IDs are not saved.
Also, use a GPU Dask cluster to speed up deduplication calculations.
```

```bash
python 5.exact.deduplication.py
```

Once this process is complete, the IDs of the duplicate documents will be saved in `exact_dedup/data/`. There were 2 duplicate documents. In our testing environment, the process took approximately 11.2 seconds using the CPU, but was approximately three times faster using the GPU, at 3.8 seconds.

### Step 5.2: Fuzzy Deduplication
Unlike Exact Deduplication, Fuzzy Deduplication does not find exact duplicates, but rather extracts similar text based on text statistics using a GPU-implemented MinhashLSH algorithm (this differs from semantic similarity). There are several intermediate steps to extract these duplicates; see here for details.

While Fuzzy Deduplication can be performed step-by-step, it can be easily done using `FuzzyDuplicates()`. The `FuzzyDuplicatesConfig()` function takes arguments such as the length of the n-gram, the number of buckets and hash values ​​within those buckets, and the Jaccard similarity threshold for determining duplicates, and these are passed to `FuzzyDuplicates()`.

Note: Fuzzy Deduplication only works with IDs in the format of `AddID()` or integer IDs.
```bash
python 6.fuzzy.deduplication.py
```

Once this process is complete (approximately several tens of seconds in the testing environment), the IDs of documents matching the duplicate criteria specified in the parameters will be saved to fuzzy_wrapper/data/.

Exact Deduplication and Fuzzy Deduplication have outputted the IDs of the duplicate documents. Now, we will remove the duplicate documents from the dataset.

## Step 6: Remove Deduplication

```bash
python 7.rm.duplicate.py
```
Once this process is complete (a few seconds in the testing environment), the deduplication-removed dataset will be saved to `remove_duplicate/result.parquet/`. 
The number of documents before processing was 59,609, but after deduplication using Exact Deduplication and Fuzzy Deduplication, 59,478 documents were saved.
