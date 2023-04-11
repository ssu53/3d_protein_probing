import os
import json
import subprocess
import requests
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from biotite.sequence.io.fasta import FastaFile

data_dir = "/oak/stanford/groups/jamesz/swansonk/3d_protein_probing/data"
thermo_meta_url = "https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/full_dataset.json.zip"
thermo_seq_url = "https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/full_dataset_sequences.fasta.zip"
split_url = "https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/splits.zip"

thermo_seq_file = os.path.join(data_dir, "full_dataset_sequences.fasta")
if not os.path.exists(thermo_seq_file):
    subprocess.call(["wget", thermo_seq_url], cwd=data_dir)
    subprocess.call(["unzip", "full_dataset_sequences.fasta.zip"], cwd=data_dir)
    subprocess.call(["wget", thermo_meta_url], cwd=data_dir)
    subprocess.call(["unzip", "full_dataset.json.zip"], cwd=data_dir)
    subprocess.call(["wget", split_url], cwd=data_dir)
    subprocess.call(["unzip", "splits.zip"], cwd=data_dir)

thermo_meta_file = os.path.join(data_dir, "full_dataset.json")
splits_file = os.path.join(data_dir, "splits/mixed_split.csv")

thermo_meta_data = json.load(open(thermo_meta_file, "r"))
thermo_seq_data = FastaFile.read(thermo_seq_file)
splits_data = pd.read_csv(splits_file)

def search_pdb_experimental(sequence):
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
        "return_type": "polymer_entity",
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
    response = requests.post(url, data=json.dumps(query), headers={'Content-Type': 'application/json'})
    if response.status_code == 204:
        return None
    else:
        return response.json()


def search_pdb_computational(sequence):
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
        "return_type": "polymer_entity",
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
    response = requests.post(url, data=json.dumps(query), headers={'Content-Type': 'application/json'})
    if response.status_code == 204:
        return None
    else:
        return response.json()


def extract_item(results, melting_point, seq):
    for result in results:
        if result["score"] >= 1.0:
            return {
                "melting_point": melting_point,
                "sequence": seq,
                "pdb_id": result["identifier"]
            }
    return None


def search_pdb(args):
    idx, seq = args
    melting_point = float(idx.split("=")[-1])
    found_experimental = search_pdb_experimental(seq)
    item = None
    if found_experimental is not None:
        # Attempt searching for experimental data
        item = extract_item(found_experimental["result_set"], melting_point, seq)
    
    if item is None:
        # if not found, check computational data
        found_computational = search_pdb_computational(seq)
        if found_computational is not None:
            item = extract_item(found_computational["result_set"], melting_point, seq)
    return item


save_file = os.path.join(data_dir, "pdb_thermostability", "seq_melting_data.json")
all_items = thread_map(search_pdb, thermo_seq_data.items(), max_workers=8)
all_items = [item for item in all_items if item is not None]
with open(save_file, "w") as f:
    json.dump(all_items, f)
