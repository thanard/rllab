
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

AWS_S3_PATH = "s3://thanard/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://thanard/rllab/code"

AWS_S3_RESOURCE_PATH = "s3://thanard/rllab/resource"

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
    #"ap-northeast-1": "ami-c42689a5",
    #"ap-northeast-2": "ami-865b8fe8",
    #"ap-south-1": "ami-ea9feb85",
    #"ap-southeast-1": "ami-c74aeaa4",
    #"ap-southeast-2": "ami-0792ae64",
    #"eu-central-1": "ami-f652a999",
    #"eu-west-1": "ami-8c0a5dff",
    #"sa-east-1": "ami-3f2cb053",
    #"us-east-1": "ami-de5171c9",
    #"us-east-2": "ami-e0481285",
    #"us-west-1": "ami-efb5ff8f",
    #"us-west-2": "ami-53903033",
}

ALL_SUBNET_INFO = {
    "us-west-1b": {
        "SubnetID": "subnet-c232ac9a",
        "Groups": "sg-762e6d11"
    },
    "us-east-1c": {
        "SubnetID": "subnet-9b98c4fe",
        "Groups": "sg-04f9d57b"
    },
    "eu-west-1c": {
        "SubnetID": "subnet-263cca6f",
        "Groups": "sg-6ac74d13"
    },
    "ap-northeast-2c": {
        "SubnetID": "subnet-fb1565b6",
        "Groups": "sg-6a901702"
    },
    "us-west-2a": {
        "SubnetID": "subnet-3e56c477",
        "Groups": "sg-4654fd3d"
    },
    "ap-southeast-1b": {
        "SubnetID": "subnet-6089a004",
        "Groups": "sg-23c68544"
    },
    "eu-west-1b": {
        "SubnetID": "subnet-648e7a03",
        "Groups": "sg-6ac74d13"
    },
    "ap-northeast-2a": {
        "SubnetID": "subnet-bb1b23d2",
        "Groups": "sg-6a901702"
    },
    "eu-west-1a": {
        "SubnetID": "subnet-9794a5cf",
        "Groups": "sg-6ac74d13"
    },
    "ap-northeast-1a": {
        "SubnetID": "subnet-a3fdded5",
        "Groups": "sg-66243301"
    },
    "us-east-1b": {
        "SubnetID": "subnet-59150d02",
        "Groups": "sg-04f9d57b"
    },
    "us-west-1c": {
        "SubnetID": "subnet-1dca8679",
        "Groups": "sg-762e6d11"
    },
    "us-east-2c": {
        "SubnetID": "subnet-c07e8d8d",
        "Groups": "sg-ffa8eb96"
    },
    "us-west-2c": {
        "SubnetID": "subnet-e8608eb3",
        "Groups": "sg-4654fd3d"
    },
    "us-east-2a": {
        "SubnetID": "subnet-04861a6d",
        "Groups": "sg-ffa8eb96"
    },
    "us-east-1d": {
        "SubnetID": "subnet-7116f45d",
        "Groups": "sg-04f9d57b"
    },
    "ap-southeast-2b": {
        "SubnetID": "subnet-7a4a021e",
        "Groups": "sg-6b54230c"
    },
    "ap-south-1a": {
        "SubnetID": "subnet-8d056be4",
        "Groups": "sg-926c57fb"
    },
    "us-east-2b": {
        "SubnetID": "subnet-d919bda2",
        "Groups": "sg-ffa8eb96"
    },
    "ap-south-1b": {
        "SubnetID": "subnet-0e7edd43",
        "Groups": "sg-926c57fb"
    },
    "sa-east-1a": {
        "SubnetID": "subnet-cc7973a8",
        "Groups": "sg-7c485218"
    },
    "eu-central-1b": {
        "SubnetID": "subnet-d55fcdaf",
        "Groups": "sg-278f3f4c"
    },
    "ap-southeast-1a": {
        "SubnetID": "subnet-b8c092ce",
        "Groups": "sg-23c68544"
    },
    "ap-southeast-2a": {
        "SubnetID": "subnet-979de0e1",
        "Groups": "sg-6b54230c"
    },
    "us-east-1a": {
        "SubnetID": "subnet-d0e13d98",
        "Groups": "sg-04f9d57b"
    },
    "sa-east-1c": {
        "SubnetID": "subnet-98be43c0",
        "Groups": "sg-7c485218"
    },
    "us-east-1e": {
        "SubnetID": "subnet-ab044397",
        "Groups": "sg-04f9d57b"
    },
    "ap-southeast-2c": {
        "SubnetID": "subnet-cbfd7192",
        "Groups": "sg-6b54230c"
    },
    "eu-central-1a": {
        "SubnetID": "subnet-c27622aa",
        "Groups": "sg-278f3f4c"
    },
    "ap-northeast-1c": {
        "SubnetID": "subnet-550f620d",
        "Groups": "sg-66243301"
    },
    "us-west-2b": {
        "SubnetID": "subnet-17cfb770",
        "Groups": "sg-4654fd3d"
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
    "us-east-2": "rllab-us-east-2",
    "sa-east-1": "rllab-sa-east-1",
    "us-west-2": "rllab-us-west-2",
    "us-east-1": "rllab-us-east-1",
    "ap-south-1": "rllab-ap-south-1",
    "ap-southeast-2": "rllab-ap-southeast-2",
    "eu-central-1": "rllab-eu-central-1",
    "ap-northeast-1": "rllab-ap-northeast-1",
    "ap-northeast-2": "rllab-ap-northeast-2",
    "ap-southeast-1": "rllab-ap-southeast-1",
    "us-west-1": "rllab-us-west-1",
    "eu-west-1": "rllab-eu-west-1"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-2": [
        "sg-abaeedc2"
    ],
    "sa-east-1": [
        "sg-724f5516"
    ],
    "us-west-2": [
        "sg-7f298004"
    ],
    "us-east-1": [
        "sg-57f9d528"
    ],
    "ap-south-1": [
        "sg-526c573b"
    ],
    "ap-southeast-2": [
        "sg-1853247f"
    ],
    "eu-central-1": [
        "sg-b98d3dd2"
    ],
    "ap-northeast-1": [
        "sg-892334ee"
    ],
    "ap-northeast-2": [
        "sg-6b901703"
    ],
    "ap-southeast-1": [
        "sg-0bc0836c"
    ],
    "us-west-1": [
        "sg-82a4e9e5"
    ],
    "eu-west-1": [
        "sg-01c14b78"
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

