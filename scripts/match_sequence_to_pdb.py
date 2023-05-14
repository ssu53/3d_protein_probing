"""Match sequences (with values) to PDB files."""
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
    """Search the RCSB PDB for experimental structures matching a sequence."""
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
    """Search the RCSB PDB for computational structures matching a sequence."""
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


def extract_item(results: dict, sequence: str, value: str) -> dict | None:
    """Extract the first matching PDB structure from a dictionary of results and return it as a dict.

    :param results: A dictionary of results from a PDB search.
    :param sequence: The sequence that is matched.
    :param value: The value for the sequence.
    :return: A dictionary of the first matching PDB structure or None.
    """
    for result in results:
        if result["score"] >= 1.0:
            return {
                "pdb_id": result["identifier"],
                "sequence": sequence,
                "value": float(value)
            }

    return None


def search_pdb(
        sequence_value: str,
        structure_type: Literal['experimental', 'computational']
) -> dict | None:
    """Search the RCSB PDB for structures matching a sequence with a value.

    :param sequence_value: A string containing the sequence and value separated by "=".
    :param structure_type: The type of structure to search for.
    :return: A dictionary of the first matching PDB structure or None.
    """
    sequence, value = sequence_value.split('=')

    if structure_type == 'experimental':
        item = search_pdb_experimental(sequence)
    elif structure_type == 'computational':
        item = search_pdb_computational(sequence)
    else:
        raise ValueError(f"Invalid structure type: {structure_type}")

    if item is not None and 'result_set' in item:
        item = extract_item(item['result_set'], sequence, value)
    else:
        item = None

    return item


def match_sequence_to_pdb(
        data_path: Path,
        save_path: Path,
        structure_type: Literal['experimental', 'computational']
) -> None:
    """Match sequences to PDB structures.

    :param data_path: The path to the data file with sequences and values.
    :param save_path: The path to save the data file with sequences, values, and PDB IDs.
    :param structure_type: The type of structure to search for.
    """
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} proteins with values")

    # Search PDB by sequence
    data = [
        item
        for item in thread_map(
            partial(search_pdb, structure_type=structure_type),
            data,
            max_workers=8
        )
        if item is not None
    ]

    print(f"Found {len(data):,} {structure_type} proteins with values")

    # Save data
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    from tap import tapify

    tapify(match_sequence_to_pdb)
