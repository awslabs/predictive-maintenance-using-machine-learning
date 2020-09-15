cd /home/ec2-user/SageMaker
export PIP_DISABLE_PIP_VERSION_CHECK=1

pip install -r ./sagemaker_predictive_maintenance/requirements.txt -q
pip install -e ./sagemaker_predictive_maintenance/