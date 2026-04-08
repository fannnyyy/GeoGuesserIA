from huggingface_hub import snapshot_download
import zipfile
import os

snapshot_download(
    repo_id="osv5m/osv5m", 
    local_dir="datasets/osv5m", 
    repo_type='dataset'
)


for root, dirs, files in os.walk("datasets/osv5m"):
    for file in files:
        if file.endswith(".zip"):
            filepath = os.path.join(root, file)
            print(f"  Extraction de {file}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(root)
            os.remove(filepath)  # Supprimer le zip après extraction

print("Dataset ok")