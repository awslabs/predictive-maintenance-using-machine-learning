import boto3
import sys

sys.path.append('./site-packages')
from crhelper import CfnResource

helper = CfnResource()


@helper.create
def on_create(_, __):
    pass

@helper.update
def on_update(_, __):
    pass


def delete_s3_objects(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).objects.all().delete()
        print(
            "Successfully deleted objects in bucket "
            "called '{}'.".format(bucket_name)
        )
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )

def delete_sagemaker_model(model_name):
    sagemaker_client = boto3.client("sagemaker")
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print("Successfully deleted model called '{}'.".format(model_name))
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            print(
                "Could not find model called '{}'. "
                "Skipping delete.".format(model_name)
            )
        else:
            raise e


def delete_s3_bucket(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).delete()
        print(
            "Successfully deleted bucket "
            "called '{}'.".format(bucket_name)
        )
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )


@helper.delete
def on_delete(event, __):
    
    # delete model
    model_name = event["ResourceProperties"]["SageMakerModelName"]
    delete_sagemaker_model(model_name)

    # remove files in s3
    solution_bucket = event["ResourceProperties"]["SolutionS3BucketName"]
    log_bucket = event["ResourceProperties"]["LogBucketName"]

    delete_s3_objects(solution_bucket)
    delete_s3_objects(log_bucket)

    # delete buckets
    delete_s3_bucket(solution_bucket)
    delete_s3_bucket(log_bucket)


def handler(event, context):
    helper(event, context)
