import kagglehub

# Download latest version
path = kagglehub.dataset_download("thienkhonghoc/affectnet")

print("Path to dataset files:", path)