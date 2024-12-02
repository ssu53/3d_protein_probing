from pathlib import Path

path_local = Path(__file__).parent

def get_proteins_path():
    return path_local / '../../data/scope40_foldseek_compatible/proteins.pt'

def get_valid_pdb_ids_train_path():
    return path_local / '../../data/embed_for_retrieval/valid_pdb_ids_train.csv'

def get_valid_pdb_ids_val_path():
    return path_local / '../../data/embed_for_retrieval/valid_pdb_ids_val.csv'

def get_pairfile_train_path():
    
    # aligned pairs from foldseek-analysis for contrastive learning
    return path_local / '../../data/embed_for_retrieval/train_data/pairfile_train.out'

def get_pairfile_val_path():
    
    # aligned pairs from foldseek-analysis for contrastive learning
    return path_local / '../../data/embed_for_retrieval/train_data/pairfile_val.out'

def get_tmaln_data_train_path():

    # from foldseek-analysis, proteins with tm-score > 0.6 (20k pairs)
    # return path_local / '../../data/embed_for_retrieval/train_data/tmaln-06_data_train.csv'
    
    # the above, supplemented with another 20k random samples (40k pairs)
    # return path_local / '../../data/embed_for_retrieval/train_data/tmaln_data_train.csv'

    # 400k all within-fold pairs and 600k random samples outside-fold (1mil pairs)
    return path_local / '../../data/embed_for_retrieval/train_data/tmaln_data_train_1.csv'

def get_tmaln_data_val_path():

    # from foldseek-analysis, proteins with tm-score > 0.6 (2k pairs)
    # return path_local / '../../data/embed_for_retrieval/train_data/tmaln-06_data_val.csv'

    # the above, supplemented with another 2k random samples (4k pairs)
    return path_local / '../../data/embed_for_retrieval/train_data/tmaln_data_val.csv'