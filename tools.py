import json
import datetime
import os
from pathlib import Path

import boto3
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(override=True)

CHROMA_PATH = str(Path(__file__).resolve().parent / "chroma_db")

_openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _chroma_client.get_or_create_collection(name="runbooks", embedding_function=_openai_ef)


def get_aws_cost(days: int = 7) -> str:
    ce = boto3.client("ce")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
    )
    return json.dumps(resp["ResultsByTime"])


def get_cost_by_service(days: int = 7) -> str:
    ce = boto3.client("ce")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    return json.dumps(resp["ResultsByTime"])


def get_s3_storage_cost(days: int = 7) -> str:
    ce = boto3.client("ce")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Simple Storage Service"]}},
        GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
    )
    return json.dumps(resp["ResultsByTime"])


def list_ec2_instances() -> str:
    ec2 = boto3.client("ec2")
    resp = ec2.describe_instances()
    instances = [
        {"id": i["InstanceId"], "type": i["InstanceType"], "state": i["State"]["Name"]}
        for r in resp["Reservations"] for i in r["Instances"]
    ]
    return json.dumps(instances)


def search_runbooks(query: str, n_results: int = 2) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    return json.dumps(docs)


TOOL_FUNCTIONS = {
    "get_aws_cost": get_aws_cost,
    "get_cost_by_service": get_cost_by_service,
    "get_s3_storage_cost": get_s3_storage_cost,
    "list_ec2_instances": list_ec2_instances,
    "search_runbooks": search_runbooks,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_aws_cost",
            "description": "Get total AWS unblended cost for the last N days from Cost Explorer",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "days to look back"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cost_by_service",
            "description": "Get AWS unblended cost for the last N days, broken down by service",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "days to look back"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_s3_storage_cost",
            "description": "Get S3 storage cost breakdown for the last N days",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "days to look back"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_ec2_instances",
            "description": "List all EC2 instances in the account with their type and current state",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_runbooks",
            "description": "Search past incident runbooks for similar issues and how they were resolved",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "description of the issue to search for"}},
                "required": ["query"],
            },
        },
    },
]
