import boto3
from time import strftime, gmtime
import pandas as pd
import numpy as np
import json
import io
import os


def get_transform_input(bucket, solution_prefix, s3_test_key, s3_transform_input):
    s3_client = boto3.client('s3')
    s3_response = s3_client.get_object(Bucket=bucket, Key=s3_test_key)
    test_file = s3_response["Body"].read()

    test_df_entry = pd.read_csv(io.BytesIO(test_file))
    test_data = test_df_entry[test_df_entry['id'] == 0 + 1][test_df_entry.columns[2:-1]].values
    test_data = test_data[0:test_data.shape[0] - 1, :].astype('float32')
    data_payload = {'input': np.expand_dims(test_data, axis=0).tolist()}

    job_name = '{}-batch-transform-{}'.format(solution_prefix, strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    s3_batch_transform_input_key = os.path.join(s3_transform_input, job_name)

    s3_client.put_object(Body=json.dumps(data_payload),
                         Bucket=bucket,
                         Key=s3_batch_transform_input_key)
    return job_name, 's3://{}/{}'.format(bucket, s3_batch_transform_input_key)

def get_transform_output(bucket, prefix, job_name):
    s3_client = boto3.client('s3')
    s3_response = s3_client.get_object(Bucket=bucket, Key=os.path.join(prefix,
                                                                       'batch-inference',
                                                                       job_name+'.out'))
    transform_out = np.array(eval(s3_response["Body"].read()))
    return transform_out
