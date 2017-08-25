import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu-cuda80"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://thanard-tmp/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://thanard-tmp/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "ap-northeast-1": "ami-c42689a5",
    "ap-northeast-2": "ami-865b8fe8",
    "ap-south-1": "ami-ea9feb85",
    "ap-southeast-1": "ami-c74aeaa4",
    "ap-southeast-2": "ami-0792ae64",
    "eu-central-1": "ami-f652a999",
    "eu-west-1": "ami-8c0a5dff",
    "sa-east-1": "ami-3f2cb053",
    "us-east-1": "ami-de5171c9",
    "us-east-2": "ami-e0481285",
    "us-west-1": "ami-efb5ff8f",
    "us-west-2": "ami-53903033",
}

ALL_SUBNET_INFO = {
    "ap-south-1a": {
        "Groups": "sg-bd4b4ed4",
        "SubnetID": "subnet-376b025e"
    },
    "ap-northeast-2c": {
        "Groups": "sg-b59e1fdd",
        "SubnetID": "subnet-58166515"
    },
    "eu-central-1a": {
        "Groups": "sg-9203b1f9",
        "SubnetID": "subnet-12acfb7a"
    },
    "us-west-2b": {
        "Groups": "sg-be48f5c5",
        "SubnetID": "subnet-52f46d1b"
    },
    "us-east-1b": {
        "Groups": "sg-48b38937",
        "SubnetID": "subnet-0f73a847"
    },
    "us-west-1b": {
        "Groups": "sg-57e5ad30",
        "SubnetID": "subnet-e5cb8381"
    },
    "us-west-2c": {
        "Groups": "sg-be48f5c5",
        "SubnetID": "subnet-95d63cce"
    },
    "eu-west-1a": {
        "Groups": "sg-65cc5f1c",
        "SubnetID": "subnet-ff15e298"
    },
    "eu-west-1b": {
        "Groups": "sg-65cc5f1c",
        "SubnetID": "subnet-ec9565a5"
    },
    "us-east-1e": {
        "Groups": "sg-48b38937",
        "SubnetID": "subnet-e043aacc"
    },
    "eu-central-1b": {
        "Groups": "sg-9203b1f9",
        "SubnetID": "subnet-a5168bdf"
    },
    "sa-east-1c": {
        "Groups": "sg-59b6523e",
        "SubnetID": "subnet-a151acf9"
    },
    "us-east-2c": {
        "Groups": "sg-c8a4e9a1",
        "SubnetID": "subnet-d73cce9a"
    },
    "ap-south-1b": {
        "Groups": "sg-bd4b4ed4",
        "SubnetID": "subnet-7544e638"
    },
    "us-west-2a": {
        "Groups": "sg-be48f5c5",
        "SubnetID": "subnet-2bc6814c"
    },
    "us-east-1d": {
        "Groups": "sg-48b38937",
        "SubnetID": "subnet-39164e5c"
    },
    "ap-southeast-2c": {
        "Groups": "sg-4b92e32c",
        "SubnetID": "subnet-6230bc3b"
    },
    "ap-southeast-2a": {
        "Groups": "sg-4b92e32c",
        "SubnetID": "subnet-087d366c"
    },
    "ap-northeast-1c": {
        "Groups": "sg-886c7cef",
        "SubnetID": "subnet-98412dc0"
    },
    "us-east-1c": {
        "Groups": "sg-48b38937",
        "SubnetID": "subnet-ad31d6f7"
    },
    "ap-southeast-2b": {
        "Groups": "sg-4b92e32c",
        "SubnetID": "subnet-b8d2aece"
    },
    "sa-east-1b": {
        "Groups": "sg-59b6523e",
        "SubnetID": "subnet-e572d393"
    },
    "ap-northeast-2a": {
        "Groups": "sg-b59e1fdd",
        "SubnetID": "subnet-4f083326"
    },
    "sa-east-1a": {
        "Groups": "sg-59b6523e",
        "SubnetID": "subnet-9b1d08ff"
    },
    "us-east-2b": {
        "Groups": "sg-c8a4e9a1",
        "SubnetID": "subnet-ace640d7"
    },
    "us-east-1a": {
        "Groups": "sg-48b38937",
        "SubnetID": "subnet-5d420261"
    },
    "ap-southeast-1a": {
        "Groups": "sg-982f63ff",
        "SubnetID": "subnet-05416a61"
    },
    "eu-west-1c": {
        "Groups": "sg-65cc5f1c",
        "SubnetID": "subnet-b92714e1"
    },
    "us-west-1c": {
        "Groups": "sg-57e5ad30",
        "SubnetID": "subnet-ba1c79e2"
    },
    "ap-southeast-1b": {
        "Groups": "sg-982f63ff",
        "SubnetID": "subnet-d30f53a5"
    },
    "ap-northeast-1b": {
        "Groups": "sg-886c7cef",
        "SubnetID": "subnet-806e43f6"
    },
    "us-east-2a": {
        "Groups": "sg-c8a4e9a1",
        "SubnetID": "subnet-33a9365a"
    }
}

INSTANCE_TYPE_INFO = {  # this prices are orientative.
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "c4.8xlarge": dict(price=1.00, vCPU=36),
    "m4.large": dict(price=0.1, vCPU=2),
    "m4.xlarge": dict(price=0.5, vCPU=4),
    "m4.2xlarge": dict(price=0.5, vCPU=8),
    "m4.4xlarge": dict(price=0.8, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    "ap-southeast-2": "thanard-rllab-ap-southeast-2",
    "ap-northeast-1": "thanard-rllab-ap-northeast-1",
    "us-east-1": "thanard-rllab-us-east-1",
    "us-west-1": "thanard-rllab-us-west-1",
    "ap-southeast-1": "thanard-rllab-ap-southeast-1",
    "eu-west-1": "thanard-rllab-eu-west-1",
    "us-west-2": "thanard-rllab-us-west-2",
    "ap-south-1": "thanard-rllab-ap-south-1",
    "sa-east-1": "thanard-rllab-sa-east-1",
    "eu-central-1": "thanard-rllab-eu-central-1",
    "ap-northeast-2": "thanard-rllab-ap-northeast-2",
    "us-east-2": "thanard-rllab-us-east-2"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "thanard-rllab"

AWS_SECURITY_GROUPS = ["thanard-rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "ap-southeast-2": [
        "sg-4b92e32c"
    ],
    "ap-northeast-1": [
        "sg-886c7cef"
    ],
    "us-east-1": [
        "sg-48b38937"
    ],
    "us-west-1": [
        "sg-57e5ad30"
    ],
    "ap-southeast-1": [
        "sg-982f63ff"
    ],
    "eu-west-1": [
        "sg-65cc5f1c"
    ],
    "us-west-2": [
        "sg-be48f5c5"
    ],
    "ap-south-1": [
        "sg-bd4b4ed4"
    ],
    "sa-east-1": [
        "sg-59b6523e"
    ],
    "eu-central-1": [
        "sg-9203b1f9"
    ],
    "ap-northeast-2": [
        "sg-b59e1fdd"
    ],
    "us-east-2": [
        "sg-c8a4e9a1"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data",
    "src",
    ".idea",
    ".pods",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
]

FAST_CODE_SYNC = True
