from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from ase.io import read

import numpy as np
import lightning.pytorch as pl
from functools import partial
from dgl.data.utils import split_dataset
#from mp_api.client import MPRester
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.ext.ase import PESCalculator
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch

from prettytable import PrettyTable

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

model_type='m3gnet'

def compute_element_refs_dict(energies,structures,elements):

    element_encoder = np.zeros((len(energies),len(elements)))

    for io,atoms in enumerate(structures):
        for jo,el in enumerate(elements):
            if el in atoms.get_chemical_symbols():
                element_encoder[io,jo] = len((np.array(atoms.get_chemical_symbols()) == el).nonzero()[0])

    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(element_encoder,energies)

    element_refs_lin_reg = lin_reg.coef_

    element_ref_dict = dict(zip(elements,element_refs_lin_reg))

    return element_ref_dict

path = '/home/p.zanineli/work/from-scratch/Dataset_ZrO2'
files_structures = os.listdir(path)
if '.ipynb_checkpoints' in files_structures: files_structures.remove('.ipynb_checkpoints')

strucs,atoms_list,energies,forces,stresses = [],[],[],[],[]

for file in files_structures:

    structures = read(f'{path}/{file}',index=':')

    for structure in structures:

        struc_forces = structure.get_forces()
        energy = structure.get_potential_energy()

        append = True

        for force in struc_forces:
            if append:
                for atom_force in force:
                    if atom_force > 35:
                        append = False
                        break

        if append:

            atoms_list.append(structure)

            structure = AseAtomsAdaptor.get_structure(structure)

            strucs.append(structure)
            forces.append(struc_forces)
            energies.append(energy)
            stresses.append(np.zeros(9))


train_structures, test_structures, train_atoms_list, test_atoms_list, train_energies, test_energies, train_forces, test_forces, train_stresses, test_stresses = train_test_split(
    strucs,atoms_list,energies,forces,stresses,
    test_size=0.1,
    shuffle=True,
    random_state=42,
)


train_labels = {
    "energies": train_energies,
    "forces": train_forces,
    "stresses": train_stresses,
}

test_labels = {
    "energies": test_energies,
    "forces": test_forces,
    "stresses": test_stresses,
}

# get element types in the dataset
elem_list = get_element_list(strucs)

print('Considering following elements:')
print(elem_list)

element_ref_dict = compute_element_refs_dict(energies,atoms_list,elem_list)

print('Considering the following elemental ref energies:')
print(element_ref_dict)

data_mean = np.mean(energies)
data_std = np.std(energies)

# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

train_dataset = MGLDataset(
    threebody_cutoff=4.0, structures=train_structures, converter=converter, labels=train_labels, include_line_graph=True,save_cache=False,
)
test_data = MGLDataset(
    threebody_cutoff=4.0, structures=test_structures, converter=converter, labels=test_labels, include_line_graph=True,save_cache=False,
)

train_data, val_data = split_dataset(
    train_dataset,
    frac_list=[0.9, 0.1],
    shuffle=True,
    random_state=42,
)

my_collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=False)

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=64,
    num_workers=0,
)

element_refs = [element_ref_dict[el] for el in elem_list]

if model_type.lower()=='m3gnet-direct':
    params = {"dim_state_embedding": 0,
              "max_n": 3,
              "max_l": 3,
              "nblocks": 3,
              "rbf_type": "SphericalBessel",
              "is_intensive": False,
              "readout_type": "weighted_atom",
              "task_type": "regression",
              "cutoff": 5.0,
              "threebody_cutoff": 4.0,
              "ntargets": 1,
              "use_smooth": False,
              "use_phi": False,
              "niters_set2set": 3,
              "nlayers_set2set": 3,
              "field": "node_feat",
              "activation_type": "swish",
              "dim_edge_embedding": 128,
              "dim_node_embedding": 128,
              "include_state": False,
              "units": 128
               }
else:
    params = {"dim_state_embedding": 0,
              "max_n": 3,
              "max_l": 3,
              "nblocks": 3,
              "rbf_type": "SphericalBessel",
              "is_intensive": False,
              "readout_type": "weighted_atom",
              "task_type": "regression",
              "cutoff": 5.0,
              "threebody_cutoff": 4.0,
              "ntargets": 1,
              "use_smooth": False,
              "use_phi": False,
              "niters_set2set": 3,
              "nlayers_set2set": 3,
              "field": "node_feat",
              "activation_type": "swish",
              "dim_edge_embedding": 64,
              "dim_node_embedding": 64,
              "include_state": False,
              "units": 64
               }

model = M3GNet(
    element_types=elem_list,
    **params,
)

# setup the M3GNetTrainer
lit_module = PotentialLightningModule(model=model, lr=1e-3, include_line_graph=True,force_weight=1,stress_weight=0,element_refs=element_refs,data_mean=0.,data_std=data_std,decay_steps=100,decay_alpha=0.01)

early_stop_callback = EarlyStopping(monitor="val_Total_Loss", min_delta=0.00, patience=200, verbose=True, mode="min")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_Total_Loss", mode="min",dirpath='./checkpoints')

logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(max_epochs=10000, accelerator="cuda", logger=logger,inference_mode=False,callbacks=[early_stop_callback,checkpoint_callback],enable_checkpointing=True)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
print(checkpoint_callback.best_model_score) # and prints it score
best_model = PotentialLightningModule.load_from_checkpoint(checkpoint_callback.best_model_path,model=model)

trainer.test(model=best_model,dataloaders=test_loader)

model_export_path = "./best_model/"
# lit_module.model.save(model_export_path)
best_model.model.save(model_export_path)

# # load trained model
# trained_model = matgl.load_model(path=model_export_path)

train_metrics_dict = trainer.test(model=best_model,dataloaders=train_loader)
val_metrics_dict = trainer.test(model=best_model,dataloaders=val_loader)
test_metrics_dict = trainer.test(model=best_model,dataloaders=test_loader)

headers = ['Dataset','Energy MAE eV/atom','Energy RMSE eV/atom', 'Force MAE eV/Ang','Force RMSE eV/Ang', 'Stress MAE eV/Ang^2', 'Stress RMSE eV/Ang^2']
keys = ['test_Energy_MAE','test_Energy_RMSE','test_Force_MAE','test_Force_RMSE','test_Stress_MAE','test_Stress_RMSE']

train_metrics = ['Train',*[train_metrics_dict[0][k] for k in keys]]
val_metrics = ['Val',*[val_metrics_dict[0][k] for k in keys]]
test_metrics = ['Test',*[test_metrics_dict[0][k] for k in keys]]

metrics_table = [headers,train_metrics,val_metrics,test_metrics]

tab = PrettyTable(metrics_table[0])
tab.add_rows(metrics_table[1:])
tab.float_format = "7.4"
print(tab,flush=True)

test_structures_ase = [AseAtomsAdaptor.get_atoms(struct) for struct in test_structures]

pot_ft = matgl.load_model(model_export_path)
m3gnet_ft = PESCalculator(pot_ft)

test_energy_per_atom = []
test_forces_flat = []
ml_energies = []
ml_energies_ft = []
ml_forces = []
ml_forces_ft = []
for io,atoms in enumerate(test_structures_ase):

    test_energy_per_atom.append(test_energies[io]/len(atoms))
    for fat in test_forces[io]:
        for f in fat:
            test_forces_flat.append(f)

    atoms.calc = m3gnet_ft
    ml_energies_ft.append(atoms.get_total_energy()/len(atoms))
    forces = atoms.get_forces().ravel()
    for f in forces:
        ml_forces_ft.append(f)

rmse_energy_ft = metrics.root_mean_squared_error(test_energy_per_atom,ml_energies_ft)

mae_energy_ft = metrics.mean_absolute_error(test_energy_per_atom,ml_energies_ft)

rmse_forces_ft = metrics.root_mean_squared_error(test_forces_flat,ml_forces_ft)

mae_forces_ft = metrics.mean_absolute_error(test_forces_flat,ml_forces_ft)

print('\nPerformance on the test set:')

headers = ['Model','Energy MAE meV/atom', 'Energy RMSE meV/atom', 'Force MAE eV/Ang', 'Force RMSE eV/Ang']
m3gnet_ft_metrics = ['M3GNET-TRAIN',mae_energy_ft*1000,rmse_energy_ft*1000,mae_forces_ft,rmse_forces_ft]

model_metrics_table = [headers,m3gnet_ft_metrics]

tab2 = PrettyTable(model_metrics_table[0])
tab2.add_rows(model_metrics_table[1:])
tab2.float_format = "7.4"
print(tab2)

