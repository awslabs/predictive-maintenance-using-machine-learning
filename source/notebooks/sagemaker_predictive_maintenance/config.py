import json
import os
import boto3
import sagemaker
from pathlib import Path


def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd())
    return current_folder

default_bucket = sagemaker.session.Session(boto3.session.Session()).default_bucket()
default_role = sagemaker.get_execution_role()

cfn_stack_outputs = {}
current_folder = get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, '../stack_outputs.json').resolve()

if os.path.exists(cfn_stack_outputs_filepath):
    with open(cfn_stack_outputs_filepath) as f:
        cfn_stack_outputs = json.load(f)


solution_prefix = cfn_stack_outputs.get('SolutionPrefix', 'sm-soln-pred-maint')
solution_bucket = cfn_stack_outputs.get('SolutionS3Bucket', default_bucket)
s3_prefix = cfn_stack_outputs.get('SolutionS3Prefix', 'pred-maintenance-artifacts')

training_job_name = cfn_stack_outputs.get('SageMakerTrainingJobName', 'sm-soln-pred-maint-model')
role = cfn_stack_outputs.get('NotebookInstanceExecutionRoleArn', default_role)