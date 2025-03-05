import os

os.environ["HF_HOME"] = "~/data/hf-home"

import boto3
import gzip
from datasets import load_dataset, concatenate_datasets, load_from_disk
from botocore.exceptions import ClientError


num_proc = 512
s3 = boto3.client(
    "s3"
)
bucket_name = "softwareheritage"


def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise


ds = load_dataset(
    "HuggingFaceTB/smollm-corpus", "python-edu", split="train", num_proc=num_proc
)
# ds = ds.map(download_contents, input_columns="blob_id", num_proc=num_proc)

batch_size = 100000

for start_idx in range(2 * batch_size, len(ds), batch_size):
    end_idx = min(start_idx + batch_size, len(ds)) 
    print(f"Processing batch: {start_idx} to {end_idx}")
    batch_ds = ds.select(range(start_idx, end_idx))
    batch_ds = batch_ds.map(
        download_contents, input_columns="blob_id", num_proc=num_proc
    )
    batch_ds = batch_ds.filter(lambda x: x["download_success"])
    batch_ds = batch_ds.save_to_disk(
        f"./python_edu_batches/python_edu_{start_idx}_{end_idx}"
    )

processed_batches = []
for start_idx in range(0, len(ds), batch_size):
    end_idx = min(start_idx + batch_size, len(ds)) 
    batch_ds = load_from_disk(f"./python_edu_batches/python_edu_{start_idx}_{end_idx}")
    processed_batches.append(batch_ds)
merged_ds = concatenate_datasets(processed_batches)
merged_ds.save_to_disk("./python_edu")
print(merged_ds[0])
# merged_ds=load_from_disk("./python_edu_batches/python_edu_0_100000")
# print(merged_ds.column_names)
