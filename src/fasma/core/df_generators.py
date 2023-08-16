from fasma.core import spectrum as sp
import pandas as pd
import numpy as np
import warnings


def get_plotting_dataframe(dataframe, root="oscillator strength", state_breakdown=False, mo_breakdown=False):
    dataframe = dataframe[dataframe[root] != 0]
    index_list = dataframe.index.names.copy()
    index_list.remove("Ending State")
    columns = ["transition energy"]
    if len(index_list) == 1 and "Starting State" in index_list and not state_breakdown and not mo_breakdown:
        columns += [root]
    else:
        dataframe = dataframe.set_index(["transition energy"], append=True)
        if mo_breakdown:
            if "core" in dataframe.columns:
                column_index = dataframe.columns.get_indexer(['core', 'valence'])
                multiply_columns = list(dataframe.columns[column_index[0]: column_index[1] + 1].values)
            else:
                multiply_columns = [i for i in list(dataframe) if 'MO' in i]
        else:
            multiply_columns = ["total sum"]
        dataframe = dataframe[multiply_columns].multiply(dataframe[root], axis="index")
        columns += multiply_columns
        dataframe.reset_index(level=["transition energy"], inplace=True)
    if not state_breakdown:
        index_list.remove("Starting State")
    if len(index_list) == 0:
        index_list = np.arange(len(dataframe)) // len(dataframe)
    return dataframe.groupby(index_list, sort=False)[columns].agg(list)


def get_spectra_dict(dataframe, spectra_name="", keep_all=False):
    absorption = False
    if "core" in dataframe.columns:
        column_index = dataframe.columns.get_indexer(['core', 'valence'])
        mo_columns = list(dataframe.columns[column_index[0]: column_index[1] + 1].values)
    else:
        mo_columns = [i for i in list(dataframe) if 'sum' in i or 'MO' in i]
        if len(mo_columns) == 0:
            mo_columns = [list(dataframe)[1]]
            absorption = True
    if "Subshell" in dataframe.index.names and "Atomic Orbital" in dataframe.index.names:
        warnings.warn("The provided dataframe is broken down by Atomic Orbital and not Subshell. If a subshell breakdown was desired, drop 'Atomic Orbital' from the index list when calling extract_dataframe_plotting data().")
        dataframe = dataframe.reset_index(level='Subshell', drop=True)
    label = ""
    if "Starting State" in dataframe.index.names:
            label += " State {} "
    if "total sum" not in mo_columns and not absorption:
        label += "{} "
    if "Atom Number" in dataframe.index.names:
        if "Atom Type" not in dataframe.index.names:
            label += "Atom "
        label += "{}"
    if "Atom Type" in dataframe.index.names:
        label += "{} "
    if "Principal Quantum Number" in dataframe.index.names:
        if "Subshell" not in dataframe.index.names and "Atomic Orbital" not in dataframe.index.names:
            label += "PQN "
        label += "{}"
    if "Subshell" in dataframe.index.names or "Atomic Orbital" in dataframe.index.names:
        label += "{}"
    spectra_dict = {}
    index_list = list(dataframe.index)
    for current_row in range(len(dataframe)):
        try:
            current_index = list(index_list[current_row])
        except TypeError:
            current_index = [index_list[current_row]]
        energies = np.array(dataframe.iloc[current_row]['transition energy'])
        if "Starting State" in dataframe.index.names:
            state_parameter_list = current_index[0:1]
            ao_parameter_list = current_index[1:]
        else:
            state_parameter_list = []
            ao_parameter_list = current_index
        for current_mo in mo_columns:
            sticks = np.array(dataframe.iloc[current_row][current_mo])
            xy = np.column_stack((energies, sticks))
            xy = xy[np.concatenate([[True], xy[1:-1, 1] != 0, [True]]), :]
            if "total sum" in mo_columns:
                mo_parameter_list = []
            else:
                mo_parameter_list = [current_mo]
            parameter_list = state_parameter_list + mo_parameter_list + ao_parameter_list
            current_spectra_name = (spectra_name + label.format(*parameter_list)).strip()
            new_x = xy[:, 0]
            new_y = xy[:, 1]
            if new_y.any() or keep_all:
                spectra_dict[current_spectra_name] = sp.SimulatedSpectrum(new_x, new_y)
    return spectra_dict


def get_excitations_dataframe(methodology, excitation_matrix):
    data_dict = {"Starting State": excitation_matrix[:, 0], "Ending State": excitation_matrix[:, 1],
                 "transition energy": excitation_matrix[:, 2], "oscillator strength": excitation_matrix[:, 3]}
    if methodology == "TD":
        data_dict.update({"rotatory strength (velocity)": excitation_matrix[:, 4], "rotatory strength (length)": excitation_matrix[:, 5]})
    df = pd.DataFrame(data_dict)
    df = df.astype({"Starting State": int, "Ending State": int})
    return df


def get_ao_dataframe(ao_matrix) -> np.array:
    """
    Returns a numpy matrix containing the oscillation strength of every excited state from each ground state.
    :raise ValueError: incorrect number of oscillation strengths found
    :return: a numpy array containing the oscillation strength of every excited state from each ground state.
    """
    data_dict = {"Atom Number": ao_matrix[:, 0], "Atom Type": ao_matrix[:, 1],
                 "Principal Quantum Number": ao_matrix[:, 2], "Subshell": ao_matrix[:, 3],
                 "Atomic Orbital": ao_matrix[:, 4]}

    df = pd.DataFrame(data_dict)

    return df


def get_summary_dataframe(summary_matrix) -> np.array:
    data_dict = {"total sum": summary_matrix[:, 0], "particle sum": summary_matrix[:, 1], "hole sum": summary_matrix[:, 2]}
    df = pd.DataFrame(data_dict)

    return df


def get_mo_dataframe(delta_diagonal_matrix, title):
    mo_transition_df = pd.DataFrame(delta_diagonal_matrix)
    mo_transition_df.columns += 1
    df = mo_transition_df.add_prefix(title)
    return df
