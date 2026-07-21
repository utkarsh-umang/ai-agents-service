import os
import uuid
import boto3
from botocore.exceptions import ClientError

def upload_to_s3(file_path: str, bucket_name: str) -> str:
    """
    Upload a file to S3 and return its public URL.

    Args:
        file_path: Local file path.
        bucket_name: S3 bucket name.

    Returns:
        Public URL of the uploaded file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    s3 = boto3.client("s3")

    file_name = os.path.basename(file_path)
    
    if not bucket_name:
        raise ValueError("AWS_S3_BUCKET environment variable is not set.")

    # Give every upload a unique name
    object_key = f"{uuid.uuid4()}_{file_name}"

    try:
        s3.upload_file(
            Filename=file_path,
            Bucket=bucket_name,
            Key=object_key,
            # ExtraArgs={
            #     "ACL": "public-read"
            # }
        )

    except ClientError as e:
        raise RuntimeError(f"Upload failed: {e}")

    # Public url needs to be configured in AWS
    # region = s3.meta.region_name
    # url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"
    
    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket_name,
            "Key": object_key,
        },
        ExpiresIn=86400,  # Valid for 24 hour
    )

    return url
