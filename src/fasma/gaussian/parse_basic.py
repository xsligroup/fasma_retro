from fasma.core import boxes as bx
from fasma.gaussian import parse_functions


def get_basic(file_keyword_trie, file_lines):
    """
    Initializes a BasicData object containing basic information about this .log calculation.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :return:
    """
    atom_list = get_atom_list(file_keyword_trie, file_lines)
    scf_type = find_scf_type(file_keyword_trie, file_lines)
    temp = find_basic(file_keyword_trie, file_lines)

    n_basis = temp[0]
    n_primitive_gaussian = temp[1]
    n_alpha_electron = temp[2]
    n_beta_electron = temp[3]
    n_electron = n_alpha_electron + n_beta_electron
    n_mo = n_basis
    if scf_type == "GHF":
        n_mo *= 2

    homo = n_electron
    if scf_type == "ROHF" or scf_type == "RHF":
        homo = homo // 2
    lumo = homo + 1

    basic_data = bx.BasicData(atom_list=atom_list, scf_type=scf_type, n_basis=n_basis,
                              n_primitive_gaussian=n_primitive_gaussian, n_alpha_electron=n_alpha_electron,
                              n_beta_electron=n_beta_electron, n_electron=n_electron, n_mo=n_mo, homo=homo, lumo=lumo)
    return basic_data


def get_atom_list(file_keyword_trie, file_lines) -> list:
    """
    Finds all the atoms present in the molecule of the .log file and returns them as a list.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :return: the list of all atoms in the molecule of the .log file
    """
    start = file_keyword_trie.find("Symbolic Z-matrix")[0] + 2

    atom_line = file_lines[file_keyword_trie.find("NAtoms")[0] - 1].split()
    num_of_atoms = int(atom_line[1])
    atom_list = []

    # Check for file containing read-in coordinates
    try:
        value = parse_functions.find_iop(file_keyword_trie, file_lines, "1", ["29"])
        if value[0] in [6, 7]:
            start -= 1
            parse_functions.replace_d(file_lines, start, start + num_of_atoms, ",", " ")
    except ValueError:
        pass

    for i in range(num_of_atoms):
        line = file_lines[start + i - 1].split()
        atom_list.append(line[0])
    return atom_list


def find_scf_type(file_keyword_trie, file_lines) -> str:
    """
    Find and return the SCF type of the .log file.
    :param file_lines: all the lines of this current file
    :return: a string containing the SCF type of the given .log file
    """
    try:
        retrieval = parse_functions.find_iop(file_keyword_trie, file_lines, "3", ["116"])[0]
    except ValueError:
        return "RHF"
    if retrieval == -2:
        line_num = file_keyword_trie.find("SCF Done")[0]
        line = file_lines[line_num - 1]
        type = line.split()[2][2]
        if type == "R" and line.split()[2][3] == "O":
            type = "RO"
        type_dict = {"R": 1, "RO": 101, "U": 2, "G": 7}
        retrieval = type_dict.get(type)
        print(type)
    key_dict = {1: "RHF", 101: "ROHF", 2: "UHF", 7: "GHF"}
    return key_dict.get(retrieval)


def find_basic(file_keyword_trie, file_lines) -> list:
    """
    Find and parse basic attributes of a .log file and return them in a list.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :return: a list containing the basic attributes in each .log file.
    """
    word_conveyor = ["basis", "primitive", "alpha", "beta"]
    variable_conveyor = []

    current_line_num = file_keyword_trie.find("alpha")[0] - 2

    for i, x in enumerate(word_conveyor):
        if i == 2:
            current_line_num += 1
        if i % 2 == 0:
            current_line = file_lines[current_line_num]
            list_of_words = current_line.split()
        index = list_of_words.index(x)
        variable_conveyor.append(int(list_of_words[index - 1]))

    return variable_conveyor
