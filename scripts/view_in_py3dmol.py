# %%

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import py3Dmol


# %%

view = py3Dmol.view(query='pdb:1ubq')
view.setStyle({'cartoon':{'color':'spectrum'}})
view

# %%

pdb_dir = '/scratch/groups/jamesz/shiye/scope40'
pdb_id = 'd2y88a_'

with open(f"{pdb_dir}/{pdb_id}") as ifile:
    system = "".join([x for x in ifile])

view = py3Dmol.view(width=400, height=300)
view.addModelsAsFrames(system)
view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
view.zoomTo()
view.show()

# %%

class Atom(dict):
    def __init__(self, line):
        self["type"] = line[0:6].strip()
        self["idx"] = line[6:11].strip()
        self["name"] = line[12:16].strip()
        self["resname"] = line[17:20].strip()
        self["resid"] = int(int(line[22:26]))
        self["x"] = float(line[30:38])
        self["y"] = float(line[38:46])
        self["z"] = float(line[46:54])
        self["sym"] = line[76:78].strip()

    def __str__(self):
        line = list(" " * 80)

        line[0:6] = self["type"].ljust(6)
        line[6:11] = self["idx"].ljust(5)
        line[12:16] = self["name"].ljust(4)
        line[17:20] = self["resname"].ljust(3)
        line[22:26] = str(self["resid"]).ljust(4)
        line[30:38] = str(self["x"]).rjust(8)
        line[38:46] = str(self["y"]).rjust(8)
        line[46:54] = str(self["z"]).rjust(8)
        line[76:78] = self["sym"].rjust(2)
        return "".join(line) + "\n"


class Molecule(list):
    def __init__(self, file):
        for line in file:
            if "ATOM" in line or "HETATM" in line:
                self.append(Atom(line))
    
    def transform(self, translation, rotation):
        for atom in self:
            curr_pos = np.array([atom['x'], atom['y'], atom['z']])
            new_pos = translation + (rotation @ curr_pos.T).T
            atom['x'] = new_pos[0]
            atom['y'] = new_pos[1]
            atom['z'] = new_pos[2]

    def __str__(self):
        outstr = ""
        for at in self:
            outstr += str(at)

        return outstr

# %%


def read_tmalign_transformation(
    dir: Path,
    fn: str,
):
    if isinstance(dir, str): dir = Path(dir)
    tf = np.genfromtxt(dir / fn, skip_header=2, skip_footer=7, usecols=(1,2,3,4))
    translation = tf[:,0]
    rotation = tf[:,1:4]

    return translation, rotation



def get_data(path):

    data = pd.read_csv(path, sep='\t')
    data['tms'] = data.loc[:, ['tms_1', 'tms_2']].min(axis=1)
    data = data[[
        'prot_1',
        'prot_2',
        'tms',
        'tms_1',
        'tms_2',
        'rmsd',
        'len_1',
        'len_2',
        'len_aln',
        'lddt',
        'chamfer',
        'emd',
        'cigar',
    ]]

    return data

# %%

pdb_dir = Path('/scratch/groups/jamesz/shiye/scope40')

# mode = 'random_pairs_train'
# mode = 'high_pairs_train'
mode = None

metrics_dir_random = Path('/home/groups/jamesz/shiye/foldseek-analysis/metrics/outputs_2024-11-06-00-16-43')
data_random = get_data(metrics_dir_random / 'tmalign_extra.csv')

metrics_dir_high = Path('/home/groups/jamesz/shiye/foldseek-analysis/metrics/outputs_2024-10-30-13-40-30')
data_high = get_data(metrics_dir_high / 'tmalign_extra.csv')

if mode == 'random_pairs_train':
    data = data_random
elif mode == 'high_pairs_train':
    data = data_high
else:
    data = pd.concat((data_high, data_random))
    data = data.drop_duplicates(subset=['prot_1', 'prot_2'], keep='first')
    data.reset_index(inplace=True)

# %%

plt.figure()
plt.scatter(data.tms, data.chamfer, s=1, alpha=0.2)
plt.ylim(0,500)
# plt.ylim(0,100)
plt.show()

# %%

print(data.tms.corr(data.rmsd, method='spearman'))
print(data.tms.corr(data.lddt, method='spearman'))
print(data.tms.corr(data.chamfer, method='spearman'))

# %%

print(data.tms.corr(data.len_1, method='spearman'))
print(data.tms.corr(data.len_2, method='spearman'))
print(data.tms.corr(data.len_aln, method='spearman'))

print(data.chamfer.corr(data.len_1, method='spearman'))
print(data.chamfer.corr(data.len_2, method='spearman'))
print(data.chamfer.corr(data.len_aln, method='spearman'))

# %%

"""
top right: high TMS, high chamfer
not much in this quadrant
happens when there is an unmatched strand 
"""
# data_subset = data[(data.tms > 0.8) & (data.chamfer > 15)]
# data_subset = data[(data.tms > 0.3) & (data.chamfer > 100)]

"""
bottom left: low TMS, low chamfer
low range of random pairs: proteins is clumpy, as in most residues close to centre of mass...
low range of high pairs: at mid TMS, low Chamfer, these look like pretty good alignments to me...
"""
# data_subset = data[(data.tms < 0.2) & (data.chamfer < 20)]
# data_subset = data[(data.tms < 0.65) & (data.chamfer < 5)]

"""
very high chamfer
happens when there is an unmatched strand 
"""
# data_subset = data[(data.chamfer > 100)]
# data_subset = data[(data.chamfer > 1000)]


"""
not a match per both metrics
"""
data_subset = data[(data.tms < 0.4) & (data.chamfer > 50)]


print(f"{len(data_subset)} in subset")

row_sampled = data_subset.sample(1).squeeze()
pdb_id_1 = row_sampled.prot_1
pdb_id_2 = row_sampled.prot_2 

print(row_sampled)

with open(pdb_dir / pdb_id_1) as ifile:
    system1 = Molecule(ifile)

with open(pdb_dir / pdb_id_2) as ifile:
    system2 = Molecule(ifile)

try:
    rot_mats_dir = metrics_dir_random / 'rot_mats'
    translation, rotation = read_tmalign_transformation(dir=rot_mats_dir, fn=f"{pdb_id_1}-{pdb_id_2}.txt")
except FileNotFoundError:
    rot_mats_dir = metrics_dir_high / 'rot_mats'
    translation, rotation = read_tmalign_transformation(dir=rot_mats_dir, fn=f"{pdb_id_1}-{pdb_id_2}.txt")

system1.transform(translation, rotation)

view = py3Dmol.view(width=400, height=300)
view.addModelsAsFrames(str(system1))
view.setStyle({'model': -1}, {"cartoon": {'color': 'pink'}})
view.addModelsAsFrames(str(system2))
view.setStyle({'model': -1}, {"cartoon": {'color': 'blue'}})
view.zoomTo()
view.show()

# view.write_html('foo.html') # not working


# %%

plt.figure()
plt.hist(data.chamfer, bins=np.arange(0,200,5))
plt.show()
# %%
