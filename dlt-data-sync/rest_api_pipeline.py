from os import environ
from typing import Any, List

import dlt
import requests
from dlt.common.pendulum import pendulum
from dlt.sources.rest_api import (
    RESTAPIConfig,
    rest_api_resources,
    ClientConfig,
    EndpointResourceBase,
    EndpointResource,
)

nursing_home_api_url_base = "http://127.0.0.1:8000/api"

environ["RUNTIME__LOG_LEVEL"] = "DEBUG"


def get_resource(endpoint_type: str, endpoint: str):
    types_endpoint = nursing_home_api_url_base + f'/{endpoint_type}/' + endpoint
    print(f"Getting resource for endpoint: {types_endpoint}")
    json: dict[str, Any] = requests.get(types_endpoint).json()
    yield json


@dlt.source
def nursing_home_static_sources():
    endpoints = {
        "types": ["injuries", "treatments"],
        "people": ["nurses", "residents"],
        "facilities": ["facilities", "floors", "rooms", "beds"],
    }

    for endpoint_name, endpoint_group in endpoints.items():
        for endpoint in endpoint_group:
            yield dlt.resource(
                get_resource(endpoint_name, endpoint),
                name=f"{endpoint_name}/{endpoint}",
                primary_key="id",
                write_disposition="replace"
            )


@dlt.source
def nursing_home_events():
    endpoint_name = "events"
    endpoints = [
        "resident-injury-events",
        "resident-treatment-events",
        "nurse-room-events"
    ]

    client_config: ClientConfig = {
        "base_url": nursing_home_api_url_base
    }

    resource_defaults: EndpointResourceBase = {
        "primary_key": "id",
        "write_disposition": "merge",
        "endpoint": {
            "params": {
                "from_date": {
                    "type": "incremental",
                    "cursor_path": "timestamp",
                    "initial_value": pendulum.date(2025, 1, 1).isoformat(),
                    "convert": lambda x: pendulum.parse(x).to_date_string(),
                },
                "to_date": pendulum.today().to_date_string(),
            },
            "paginator": "single_page"
        }
    }

    resources: List[EndpointResource] = [
        {
            "name": endpoint,
            "endpoint": {
                "path": f"{endpoint_name}/{endpoint}",
            },
        }
        for endpoint in endpoints
    ]

    rest_api_config: RESTAPIConfig = {
        "client": client_config,
        "resource_defaults": resource_defaults,
        "resources": resources
    }

    yield from rest_api_resources(rest_api_config)


def load_nursing_home_data():
    pipeline = dlt.pipeline(
        pipeline_name="rest_api_nursing_home",
        destination='duckdb',
        dataset_name="nursing_home_data",
    )

    nursing_home_static_sources_ = nursing_home_static_sources()
    nursing_home_events_ = nursing_home_events()

    load_info = pipeline.run([
        nursing_home_static_sources_,
        nursing_home_events_
    ])
    print(load_info)


if __name__ == "__main__":
    load_nursing_home_data()
