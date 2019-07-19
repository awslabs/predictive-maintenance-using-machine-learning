##############################################################################
#  Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.   #
#                                                                            #
#  Licensed under the Amazon Software License (the "License"). You may not   #
#  use this file except in compliance with the License. A copy of the        #
#  License is located at                                                     #
#                                                                            #
#      http://aws.amazon.com/asl/                                            #
#                                                                            #
#  or in the "license" file accompanying this file. This file is distributed #
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,        #
#  express or implied. See the License for the specific language governing   #
#  permissions and limitations under the License.                            #
##############################################################################

import json
import os
import io
import re
import boto3
import time
import datetime

import numpy as np
import pandas as pd

def lambda_handler(event, context):
    transform_input = get_transform_input()
    print("Running SageMaker batch transform on input file at {}".format(transform_input))
    batch_transform_response = run_batch_transform(transform_input)
    print("Batch Transform results stored at: {}".format(batch_transform_response))
    return batch_transform_response
    
def get_transform_input():
    s3_client = boto3.client('s3')
    print("Using file at S3 Location: {}".format(os.path.join(os.environ['s3_bucket'], 
                                                              os.environ['s3_test_key'])))
    s3_response = s3_client.get_object(Bucket=os.environ['s3_bucket'],
                                    Key=os.environ['s3_test_key'])
    test_file = s3_response["Body"].read()

    test_df_entry = pd.read_csv(io.BytesIO(test_file))
    test_data = test_df_entry[test_df_entry['id']==0+1][test_df_entry.columns[2:-1]].values
    test_data = test_data[0:test_data.shape[0]-1,:].astype('float32')
    data_payload = {'input':np.expand_dims(test_data, axis=0).tolist()}
    
    s3_batch_transform_input_key = os.path.join(os.environ['s3_transform_input'],
                                                get_batch_transform_name(file_name=True))
    
    s3_client.put_object(Body=json.dumps(data_payload),
                         Bucket=os.environ['s3_bucket'], 
                         Key=s3_batch_transform_input_key)
    return s3_batch_transform_input_key


def get_batch_transform_name(file_name=False):
    millisecond_regex = r'\.\d+'
    timestamp = re.sub(millisecond_regex, '', str(datetime.datetime.now()))
    timestamp = timestamp.replace(" ", "-").replace(":", "-")
    if file_name:
        return 'batch-transform-input-{}.json'.format(timestamp)
    return 'predictive-maintenance-batch-transform-job-{}'.format(timestamp)

def run_batch_transform(input_file_location):
    batch_job_name = get_batch_transform_name()
    batch_input = 's3://{}/{}'.format(os.environ['s3_bucket'], input_file_location)
    batch_output = 's3://{}/{}'.format(os.environ['s3_bucket'], os.environ['s3_transform_output'])
    print("SageMaker batch transform results will be stored at {}".format(batch_output))
    
    sm = boto3.client('sagemaker')
    model_name = os.environ['sm_model_name']
    
    print("Using SageMaker model: {} for batch transform".format(model_name))
    
    payload = get_batch_transform_payload(batch_job_name, batch_input, batch_output, model_name)
    sm.create_transform_job(**payload)
    
    # Monitor transform job
    while(True):
        response = sm.describe_transform_job(TransformJobName=batch_job_name)
        status = response['TransformJobStatus']
        if  status == 'Completed':
            print("Transform job ended with status: " + status)
            break
        if status == 'Failed':
            message = response['FailureReason']
            print('Transform failed with the following error: {}'.format(message))
            raise Exception('Transform job failed') 
        print("Transform job is still in status: " + status)
        time.sleep(30)
    return response['TransformOutput']
    

def get_batch_transform_payload(batch_job_name, batch_input, batch_output, model_name):
    request = \
    {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "BatchStrategy": "SingleRecord",
        "TransformOutput": {
            "S3OutputPath": batch_output
        },
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": batch_input 
                }
            },
            "ContentType": "text/json",
            "SplitType": "Line",
            "CompressionType": "None"
        },
        "TransformResources": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1
        }
    }
    return request