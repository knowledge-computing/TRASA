import boto3
from boto3.s3.transfer import TransferConfig
import os

access_key = os.environ["MSI_S3_ACCESS_KEY"]
secret_key = os.environ["MSI_S3_SECRET_KEY"]

bucket_name = "kirsa002-ca-area"
local_file = "/projects/standard/yaoyi/shared/IMOLA/CA_area.zip"
object_name = os.path.basename(local_file)

s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    endpoint_url="https://s3.msi.umn.edu",
)

# force NO multipart upload
config = TransferConfig(multipart_threshold=1024 * 1024 * 1024)  # 1GB threshold

s3.upload_file(local_file, bucket_name, object_name, Config=config)

print("Uploaded successfully")