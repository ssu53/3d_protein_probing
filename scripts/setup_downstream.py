"""Set up a downstream dataset (assumes it has already been downloaded and partially processed)."""
import json
from functools import partial
from pathlib import Path
from typing import Literal

import requests
from tqdm.contrib.concurrent import thread_map


def pdb_request(url: str, query: dict) -> dict | None:
    """Send a query to the RCSB PDB search API.

    :param url: The URL of the RCSB PDB search API.
    :param query: The query to send.
    """
    response = requests.post(url, data=json.dumps(query), headers={'Content-Type': 'application/json'})

    if response.status_code == 204:
        return None

    try:
        return response.json()
    except json.decoder.JSONDecodeError:
        print(response.text)
        return None


def search_pdb_experimental(sequence: str) -> dict | None:
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 0.1,
                        "identity_cutoff": 1,
                        "sequence_type": "protein",
                        "value": sequence
                    }
                },
                {
                    "type": "group",
                    "nodes": [
                        {
                            "type": "group",
                            "nodes": [
                                {
                                    "type": "terminal",
                                    "service": "text",
                                    "parameters": {
                                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                                        "value": "Protein",
                                        "operator": "exact_match"
                                    }
                                }
                            ],
                            "logical_operator": "or",
                            "label": "entity_poly.rcsb_entity_polymer_type"
                        },
                        {
                            "type": "group",
                            "nodes": [
                                {
                                    "type": "terminal",
                                    "service": "text",
                                    "parameters": {
                                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                                        "value": "experimental",
                                        "operator": "exact_match"
                                    }
                                },
                                {
                                    "type": "group",
                                    "nodes": [
                                        {
                                            "type": "terminal",
                                            "service": "text",
                                            "parameters": {
                                                "attribute": "rcsb_entry_info.resolution_combined",
                                                "value": {
                                                    "from": 0.0,
                                                    "to": 3.0,
                                                    "include_lower": True,
                                                    "include_upper": False
                                                },
                                                "operator": "range"
                                            }
                                        }
                                    ],
                                    "logical_operator": "or",
                                    "label": "rcsb_entry_info.resolution_combined"
                                }
                            ],
                            "logical_operator": "and"
                        }
                    ],
                    "logical_operator": "and",
                    "label": "text"
                }
            ]
        },
        "return_type": "polymer_instance",
        "request_options": {
            "results_content_type": [
                "experimental"
            ],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ],
            "scoring_strategy": "combined"
        }
    }

    return pdb_request(
        url=url,
        query=query
    )


def search_pdb_computational(sequence: str) -> dict | None:
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 0.1,
                        "identity_cutoff": 1,
                        "sequence_type": "protein",
                        "value": sequence
                    }
                },
                {
                    "type": "group",
                    "nodes": [
                        {
                            "type": "group",
                            "nodes": [
                                {
                                    "type": "group",
                                    "logical_operator": "and",
                                    "nodes": [
                                        {
                                            "type": "terminal",
                                            "service": "text",
                                            "parameters": {
                                                "attribute": "rcsb_ma_qa_metric_global.ma_qa_metric_global.value",
                                                "value": 90,
                                                "operator": "greater_or_equal"
                                            }
                                        },
                                        {
                                            "type": "terminal",
                                            "service": "text",
                                            "parameters": {
                                                "attribute": "rcsb_ma_qa_metric_global.ma_qa_metric_global.type",
                                                "operator": "exact_match",
                                                "value": "pLDDT",
                                                "negation": False
                                            }
                                        }
                                    ],
                                    "label": "nested-attribute"
                                },
                                {
                                    "type": "terminal",
                                    "service": "text",
                                    "parameters": {
                                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                                        "value": "Protein",
                                        "operator": "exact_match"
                                    }
                                }
                            ],
                            "logical_operator": "and"
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_entry_info.structure_determination_methodology",
                                "value": "computational",
                                "operator": "exact_match"
                            }
                        }
                    ],
                    "logical_operator": "and",
                    "label": "text"
                }
            ]
        },
        "return_type": "polymer_instance",
        "request_options": {
            "results_content_type": [
                "computational"
            ],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ],
            "scoring_strategy": "combined"
        }
    }

    return pdb_request(
        url=url,
        query=query
    )


def extract_item(results: dict, value: float, sequence: str) -> dict | None:
    for result in results:
        if result["score"] >= 1.0:
            return {
                "value": value,
                "sequence": sequence,
                "pdb_id": result["identifier"]
            }

    return None


def search_pdb(
        sequence_value: tuple[str, float],
        structure_type: Literal['experimental', 'computational']
) -> dict | None:
    sequence, value = sequence_value

    if structure_type == 'experimental':
        item = search_pdb_experimental(sequence)
    elif structure_type == 'computational':
        item = search_pdb_computational(sequence)
    else:
        raise ValueError(f"Invalid structure type: {structure_type}")

    if item is not None and 'result_set' in item:
        item = extract_item(item['result_set'], value, sequence)
    else:
        item = None

    return item


def setup_downstream(
        data_path: Path,
        save_path: Path,
        downstream_task: str,
        structure_type: Literal['experimental', 'computational']
) -> None:
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    sequence_values = [
        (item["sequence"], item[downstream_task])
        for item in data.values()
        if item is not None
    ]

    # Search PDB by sequence
    data = [
        item
        for item in thread_map(
            partial(search_pdb, structure_type=structure_type),
            sequence_values,
            max_workers=8
        )
        if item is not None
    ]

    # Save data
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    from tap import tapify

    tapify(setup_downstream)
