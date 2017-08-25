
import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-east-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu-cuda80"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://thanard-temp/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://thanard-temp/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "ap-northeast-1": "ami-002f0167",
    "ap-northeast-2": "ami-590bd937",
    "ap-south-1": "ami-77314318",
    "ap-southeast-1": "ami-1610a975",
    "ap-southeast-2": "ami-9dd4ddfe",
    "eu-central-1": "ami-63af720c",
    "eu-west-1": "ami-41484f27",
    "sa-east-1": "ami-b7234edb",
    "us-east-1": "ami-83f26195",
    "us-east-2": "ami-66614603",
    "us-west-1": "ami-576f4b37",
    "us-west-2": "ami-b8b62bd8",

#    "ap-northeast-1": "ami-c42689a5",
#    "ap-northeast-2": "ami-865b8fe8",
#    "ap-south-1": "ami-ea9feb85",
#    "ap-southeast-1": "ami-c74aeaa4",
#    "ap-southeast-2": "ami-0792ae64",
#    "eu-central-1": "ami-f652a999",
#    "eu-west-1": "ami-8c0a5dff",
#    "sa-east-1": "ami-3f2cb053",
#    "us-east-1": "ami-de5171c9",
#    "us-east-2": "ami-e0481285",
#    "us-west-1": "ami-efb5ff8f",
#    "us-west-2": "ami-53903033",
}

ALL_SUBNET_INFO = {
    "ap-south-1b": {
        "Groups": "sg-926c57fb",
        "SubnetID": "subnet-0e7edd43"
    },
    "ap-northeast-1a": {
        "Groups": "sg-66243301",
        "SubnetID": "subnet-a3fdded5"
    },
    "ap-south-1a": {
        "Groups": "sg-926c57fb",
        "SubnetID": "subnet-8d056be4"
    },
    "us-west-2a": {
        "Groups": "sg-4654fd3d",
        "SubnetID": "subnet-3e56c477"
    },
    "sa-east-1c": {
        "Groups": "sg-7c485218",
        "SubnetID": "subnet-98be43c0"
    },
    "eu-central-1b": {
        "Groups": "sg-278f3f4c",
        "SubnetID": "subnet-d55fcdaf"
    },
    "ap-northeast-2a": {
        "Groups": "sg-6a901702",
        "SubnetID": "subnet-bb1b23d2"
    },
    "ap-southeast-2a": {
        "Groups": "sg-6b54230c",
        "SubnetID": "subnet-979de0e1"
    },
    "ap-southeast-1a": {
        "Groups": "sg-23c68544",
        "SubnetID": "subnet-b8c092ce"
    },
    "eu-central-1a": {
        "Groups": "sg-278f3f4c",
        "SubnetID": "subnet-c27622aa"
    },
    "us-east-1a": {
        "Groups": "sg-04f9d57b",
        "SubnetID": "subnet-d0e13d98"
    },
    "ap-southeast-2b": {
        "Groups": "sg-6b54230c",
        "SubnetID": "subnet-7a4a021e"
    },
    "eu-west-1c": {
        "Groups": "sg-6ac74d13",
        "SubnetID": "subnet-263cca6f"
    },
    "eu-west-1a": {
        "Groups": "sg-6ac74d13",
        "SubnetID": "subnet-9794a5cf"
    },
    "us-east-1e": {
        "Groups": "sg-04f9d57b",
        "SubnetID": "subnet-ab044397"
    },
    "us-east-1b": {
        "Groups": "sg-04f9d57b",
        "SubnetID": "subnet-59150d02"
    },
    "us-east-1c": {
        "Groups": "sg-04f9d57b",
        "SubnetID": "subnet-9b98c4fe"
    },
    "us-west-1c": {
        "Groups": "sg-762e6d11",
        "SubnetID": "subnet-1dca8679"
    },
    "sa-east-1a": {
        "Groups": "sg-7c485218",
        "SubnetID": "subnet-cc7973a8"
    },
    "ap-northeast-1c": {
        "Groups": "sg-66243301",
        "SubnetID": "subnet-550f620d"
    },
    "ap-northeast-2c": {
        "Groups": "sg-6a901702",
        "SubnetID": "subnet-fb1565b6"
    },
    "us-east-2a": {
        "Groups": "sg-ffa8eb96",
        "SubnetID": "subnet-04861a6d"
    },
    "ap-southeast-2c": {
        "Groups": "sg-6b54230c",
        "SubnetID": "subnet-cbfd7192"
    },
    "us-east-2b": {
        "Groups": "sg-ffa8eb96",
        "SubnetID": "subnet-d919bda2"
    },
    "us-west-1b": {
        "Groups": "sg-762e6d11",
        "SubnetID": "subnet-c232ac9a"
    },
    "us-west-2b": {
        "Groups": "sg-4654fd3d",
        "SubnetID": "subnet-17cfb770"
    },
    "us-west-2c": {
        "Groups": "sg-4654fd3d",
        "SubnetID": "subnet-e8608eb3"
    },
    "ap-southeast-1b": {
        "Groups": "sg-23c68544",
        "SubnetID": "subnet-6089a004"
    },
    "eu-west-1b": {
        "Groups": "sg-6ac74d13",
        "SubnetID": "subnet-648e7a03"
    },
    "us-east-2c": {
        "Groups": "sg-ffa8eb96",
        "SubnetID": "subnet-c07e8d8d"
    },
    "us-east-1d": {
        "Groups": "sg-04f9d57b",
        "SubnetID": "subnet-7116f45d"
    }
}

INSTANCE_TYPE_INFO = {  #this prices are orientative.
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
    "us-east-1": "thanard-rllab-us-east-1",
    "ap-southeast-1": "thanard-rllab-ap-southeast-1",
    "ap-south-1": "thanard-rllab-ap-south-1",
    "eu-west-1": "thanard-rllab-eu-west-1",
    "ap-northeast-1": "thanard-rllab-ap-northeast-1",
    "eu-central-1": "thanard-rllab-eu-central-1",
    "sa-east-1": "thanard-rllab-sa-east-1",
    "us-west-2": "thanard-rllab-us-west-2",
    "us-east-2": "thanard-rllab-us-east-2",
    "ap-southeast-2": "thanard-rllab-ap-southeast-2",
    "us-west-1": "thanard-rllab-us-west-1",
    "ap-northeast-2": "thanard-rllab-ap-northeast-2"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "thanard-rllab"

AWS_SECURITY_GROUPS = ["thanard-rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-1": [
        "sg-2f7b4150"
    ],
    "ap-southeast-1": [
        "sg-2e175b49"
    ],
    "ap-south-1": [
        "sg-704d4819"
    ],
    "eu-west-1": [
        "sg-f3de4d8a"
    ],
    "ap-northeast-1": [
        "sg-70607017"
    ],
    "eu-central-1": [
        "sg-261ba94d"
    ],
    "sa-east-1": [
        "sg-69b1550e"
    ],
    "us-west-2": [
        "sg-a863ded3"
    ],
    "us-east-2": [
        "sg-aca9e4c5"
    ],
    "ap-southeast-2": [
        "sg-34addc53"
    ],
    "us-west-1": [
        "sg-cee8a0a9"
    ],
    "ap-northeast-2": [
        "sg-01971669"
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

