import pandas as pd


def filter_mo_analysis(dataframe: pd.DataFrame, index: list = [], mo_list: list = []):
    columns = [i for i in list(dataframe) if 'sum' in i or 'MO' in i]
    if index:
        dataframe = dataframe.groupby(index, sort=False)[columns].sum()
    if mo_list:
        kept_mo = ['MO ' + str(number) for number in mo_list]
        dataframe = dataframe[kept_mo]
    return dataframe


def filter_transition_rows(dataframe: pd.DataFrame, index: list = [], attributes: list = ["transition energy", "oscillator strength"], energy_range: tuple = None):
    if energy_range is not None:
        dataframe = dataframe[(dataframe['transition energy'] >= energy_range[0]) & (dataframe['transition energy'] <= energy_range[1])]
    columns = [i for i in list(dataframe) if 'sum' in i or 'MO' in i]
    group_list = ["Starting State", "Ending State"] + index + attributes
    dataframe = dataframe.groupby(group_list, sort=False)[columns].sum()
    dataframe.reset_index(level=attributes, inplace=True)
    return dataframe


def filter_transition_columns(dataframe, custom_mo_dict, basic_data):
    new_df = dataframe.copy()
    mo_dict = {}
    if isinstance(custom_mo_dict, list):
        mo_list = custom_mo_dict
        custom_mo_dict = {}
    elif isinstance(custom_mo_dict, dict):
        mo_list = []
        for label, mo in custom_mo_dict.items():
            mo_list += mo
            custom_mo_dict[label] = ['AS MO ' + str(number) for number in mo]
    core_column = ['AS MO ' + str(mo) for mo in range(1, basic_data.homo + 1) if mo not in mo_list]
    mo_dict["core"] = core_column
    mo_dict.update(custom_mo_dict)
    valence_column = ['AS MO ' + str(mo) for mo in range(basic_data.homo + 1, basic_data.n_mo + 1) if mo not in mo_list]
    mo_dict["valence"] = valence_column
    for label, column_list in mo_dict.items():
        combine_mo_columns(new_df, label, column_list)
    return new_df


def show_nonzero_mo(dataframe):
    new_df = dataframe.copy()
    if "core" in dataframe.columns:
        columns = ['core', 'valence']
    else:
        columns = [i for i in list(dataframe) if 'MO' in i]
    column_index = new_df.columns.get_indexer([columns[0], columns[-1]])
    cols = new_df.columns.values[column_index[0]:]
    mask = new_df.ne(0).values[:, column_index[0]:]
    out = [cols[x].tolist() for x in mask]
    new_df.insert(new_df.columns.get_loc(columns[0]), "non-zero MOs", out)
    return new_df


def get_nonzero_mo(dataframe):
    non_zero_mo_per_excitation = dataframe["non-zero MOs"].to_list()
    mo_list = set()
    for current_mo_list in non_zero_mo_per_excitation:
        mo_list.update(current_mo_list)
    mo_list = [int(mo.strip("AS MO")) for mo in mo_list]
    mo_list.sort()
    return mo_list


def combine_mo_columns(dataframe, label, column_list):
    sum_column = dataframe.loc[:, column_list].sum(axis=1)
    if label == "core":
        dataframe.insert(dataframe.columns.get_loc("AS MO 1"), label, sum_column)
    else:
        dataframe[label] = sum_column
    dataframe.drop(column_list, axis=1, inplace=True)
