from fasma.gaussian import parse_gaussian as pg
from fasma.core import file_reader as fr
from fasma.core import boxes as bx
#import parse_chronus as pc
import numpy as np
import pandas as pd
import pickle

def parse(filename):
    key_trie_list, file_lines_list, file_type = fr.read(filename)
    if file_type == "Gaussian":
        parse_meth = pg.parse
    #elif file_type == "ChronusQ":
        #parse_meth = pc.parse
    box_list = []
    for current_key_trie, current_file_lines in zip(key_trie_list, file_lines_list):
        current_box = pg.parse(current_key_trie, current_file_lines)
        box_list.append(current_box)
    #elif file_type == "ChronusQ":
        #current_box = pc.parse(file_key_trie, file_lines, file_type)
    if len(box_list) == 1:
        box_list = box_list[0]
    return box_list


# Function for merging two box object containing separate excited state and pop calculation
def merge(box1, box2):
    if box1.spectra_data is None:
        spectra_data = box2.spectra_data
        pop_data = box1.pop_data
    else:
        spectra_data = box1.spectra_data
        pop_data = box2.pop_data
    return bx.Box(box1.basic_data, spectra_data=spectra_data, pop_data=pop_data)


# Function for merging multiple box objects containing separate chunks of spectra for the same molecule
def merge_td(box_list):
    basic_data = box_list[0].basic_data
    pop_data = None
    spectra_data_list = []
    duplicate_columns = ["transition energy", "oscillator strength"]
    for current_box in box_list:
        if pop_data is None:
            pop_data = current_box.pop_data
        if basic_data.scf_type == "UHF":
            spectra_data_list.append(current_box.generate_merged_mo_transition_analysis())
        else:
            spectra_data_list.append(current_box.generate_mo_transition_analysis())
    mo_columns = [i for i in list(spectra_data_list[0]) if 'MO' in i]
    round_dict = {key: 2 for key in mo_columns}
    duplicate_columns += mo_columns
    merged_mo_transition_analysis = pd.concat(spectra_data_list, sort=False)
    merged_mo_transition_analysis.sort_values(by=["transition energy"], inplace=True)
    merged_mo_transition_analysis = merged_mo_transition_analysis[~merged_mo_transition_analysis.round(round_dict).duplicated(subset=duplicate_columns)]
    merged_mo_transition_analysis.reset_index(inplace=True)
    merged_mo_transition_analysis['Ending State'] = np.arange(2, merged_mo_transition_analysis.shape[0] + 2)
    excitation_index = merged_mo_transition_analysis.columns.get_indexer(['Starting State', 'rotatory strength (length)'])
    mo_index = merged_mo_transition_analysis.columns.get_indexer(['AS MO 1', 'AS MO ' + str(basic_data.n_mo)])

    df_matrix = merged_mo_transition_analysis.to_numpy()
    excitation_matrix = df_matrix[:, excitation_index[0]: excitation_index[1] + 1]
    delta_diagonal_matrix = df_matrix[:, mo_index[0]: mo_index[1] + 1]
    spectra_data = bx.TDData(n_excited_state=merged_mo_transition_analysis.shape[0], n_active_space_mo=basic_data.n_mo, n_active_space_electron=basic_data.n_electron, excitation_matrix=excitation_matrix, delta_diagonal_matrix=delta_diagonal_matrix)
    if basic_data.scf_type == "UHF":
        beta_mo_index = merged_mo_transition_analysis.columns.get_indexer(
            ['Beta AS MO 1', 'Beta AS MO ' + str(basic_data.n_mo)])
        beta_delta_diagonal_matrix = df_matrix[:, beta_mo_index[0]: beta_mo_index[1] + 1]
        spectra_data.add_beta_delta_diagonal_matrix(beta_delta_diagonal_matrix)
    return bx.Box(basic_data, spectra_data=spectra_data, pop_data=pop_data)


def save(item, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(item, output, pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
