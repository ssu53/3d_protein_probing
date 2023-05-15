"""Process torchdrug targets for GeneOntology or EnzymeComission datasets.

Assumes zip folder with data is already downloaded.

GeneOntology: https://torchdrug.ai/docs/_modules/torchdrug/datasets/gene_ontology.html#GeneOntology
GeneOntology: https://zenodo.org/record/6622158/files/GeneOntology.zip

EnzymeCommission: https://torchdrug.ai/docs/_modules/torchdrug/datasets/enzyme_commission.html#EnzymeCommission
EnzymeCommission: https://zenodo.org/record/6622158/files/EnzymeCommission.zip
"""
import csv
from typing import Literal

import torch


GO_BRANCHES = ['MF', 'BP', 'CC']


def load_annotation_ec(tsv_file: str) -> dict[str, torch.Tensor]:
    with open(tsv_file) as fin:
        reader = csv.reader(fin, delimiter="\t")
        _ = next(reader)
        tasks = next(reader)
        task2id = {task: i for i, task in enumerate(tasks)}
        _ = next(reader)
        pos_targets = {}
        for pdb_id, pos_target in reader:
            pos_target = [task2id[t] for t in pos_target.split(",")]
            pos_target = torch.tensor(pos_target)
            pos_targets[pdb_id] = pos_target

    for key, value in pos_targets.items():
        targets = torch.zeros(len(task2id)).long()
        targets[value] = 1
        pos_targets[key] = targets

    return pos_targets


def load_annotation_go(tsv_file: str, branch: str) -> dict[str, torch.Tensor]:
    idx = GO_BRANCHES.index(branch)
    with open(tsv_file) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i in range(12):
            _ = next(reader)
            if i == idx * 4 + 1:
                tasks = _
        task2id = {task: i for i, task in enumerate(tasks)}
        _ = next(reader)
        pos_targets = {}
        for line in reader:
            pdb_id, pos_target = line[0], line[idx + 1] if idx + 1 < len(line) else None
            pos_target = [task2id[t] for t in pos_target.split(",")] if pos_target else []
            pos_target = torch.LongTensor(pos_target)
            pos_targets[pdb_id] = pos_target

    for key, value in pos_targets.items():
        targets = torch.zeros(len(task2id)).long()
        targets[value] = 1
        pos_targets[key] = targets

    return pos_targets


# TODO: clean this up
def torchdrug_process_targets(dataset: Literal['ec', 'go']) -> None:
    if dataset == 'ec':
        pos_targets = load_annotation_ec('nrPDB-EC_annot.tsv')
    elif dataset == 'go':
        pos_targets = {}
        for branch in GO_BRANCHES:
            pos_targets_branch = load_annotation_go('nrPDB-GO_annot.tsv', branch)

            for key, value in pos_targets_branch.items():
                if key in pos_targets:
                    pos_targets[key] = torch.cat([pos_targets[key], value])
                else:
                    pos_targets[key] = value
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    torch.save(pos_targets, 'targets.pt')


if __name__ == '__main__':
    torchdrug_process_targets('ec')
    torchdrug_process_targets('go')
