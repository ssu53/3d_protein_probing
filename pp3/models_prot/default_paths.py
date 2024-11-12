from pathlib import Path

path_local = Path(__file__).parent

def get_valid_pdb_ids_train_path():
    return path_local / '../../data/embed_for_retrieval/valid_pdb_ids_train.csv'

def get_valid_pdb_ids_val_path():
    return path_local / '../../data/embed_for_retrieval/valid_pdb_ids_val.csv'

def get_pairfile_train_path():
    return path_local / '../../data/embed_for_retrieval/pairfile_train.out'

def get_pairfile_val_path():
    return path_local / '../../data/embed_for_retrieval/pairfile_val.out'

def get_tmaln_data_train_path():
    return path_local / '../../data/embed_for_retrieval/tmaln_data_train.csv'
    # return path_local / '../../data/embed_for_retrieval/tmaln-06_data_train.csv'

def get_tmaln_data_val_path():
    return path_local / '../../data/embed_for_retrieval/tmaln_data_val.csv'
    # return path_local / '../../data/embed_for_retrieval/tmaln-06_data_val.csv'