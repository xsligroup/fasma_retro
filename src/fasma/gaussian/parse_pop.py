from fasma.core import boxes as bx
from fasma.core import messages as msg
from fasma.gaussian import parse_functions
from fasma.gaussian import parse_matrices
import pandas as pd
import numpy as np
import math as m


def check_pop(basic, file_keyword_trie, file_lines, cas_status: bool) -> bool:
    """
    Tests if .log contains a pop calculation.
    If yes, initializes necessary pop-related attributes with SCF type in consideration.
    :return: true if this .log contains a population calculation, false otherwise
    """
    try:
        temp_check = parse_functions.find_iop(file_keyword_trie, file_lines, "3", ["33"])
    except IndexError:
        pass
    else:
        if temp_check[0] >= 1:
            ao_matrix = get_ao_matrix(basic, file_keyword_trie, file_lines)
            overlap_matrix = get_overlap_matrix(basic, file_keyword_trie, file_lines)
            electron_data = get_alpha_electron_data(basic, file_keyword_trie, file_lines, cas_status)
            calculate_ao_projection(overlap_matrix, electron_data)
            pop_data = bx.PopData(ao_matrix=ao_matrix, overlap_matrix=overlap_matrix, electron_data=electron_data)
            if (basic.scf_type == "ROHF" and cas_status) or basic.scf_type == "UHF":
                beta_electron_data = get_beta_electron_data(basic, file_keyword_trie, file_lines, cas_status)
                calculate_ao_projection(overlap_matrix, beta_electron_data)
                pop_data.add_beta_electron_data(beta_electron_data)
            return pop_data
    return


def calculate_ao_projection(overlap_matrix, electron_data):
    ao_projection_matrix = np.multiply(np.dot(overlap_matrix, electron_data.mo_coefficient_matrix), np.conjugate(electron_data.mo_coefficient_matrix))
    electron_data.add_ao_projection_matrix(ao_projection_matrix)


def get_alpha_electron_data(basic, file_keyword_trie, file_lines, cas_status):
    density_matrix = get_density_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, cas_status=cas_status)
    mo_coefficient_matrix = get_mo_coefficient_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines)
    eigenvalues = get_eigenvalues(file_keyword_trie=file_keyword_trie, file_lines=file_lines, n_mo=basic.n_mo)
    return bx.ElectronData(density_matrix, mo_coefficient_matrix, eigenvalues)


def get_beta_electron_data(basic, file_keyword_trie, file_lines, cas_status):
    density_matrix = get_density_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, cas_status=cas_status, beta=True)
    beta_electron_data = bx.BetaData(density_matrix=density_matrix)
    if basic.scf_type == "UHF":
        mo_coefficient_matrix = get_mo_coefficient_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, beta=True)
        eigenvalues = get_eigenvalues(file_keyword_trie=file_keyword_trie, file_lines=file_lines, n_mo=basic.n_mo, beta=True)
    beta_electron_data.add_beta_mo_coefficient_matrix(mo_coefficient_matrix)
    beta_electron_data.add_beta_eigenvalues(eigenvalues)
    return beta_electron_data


def get_eigenvalues(file_keyword_trie, file_lines, n_mo, n_col=5, beta=False):
    cycles = m.ceil(n_mo / n_col)
    n_col_block = n_col
    n_last_block_col = n_mo % n_col
    eigenvalue_type = "alpha"

    if beta:
        eigenvalue_lines = file_keyword_trie.find("Eigenvalues")[cycles:]
        eigenvalue_type = "beta"
    else:
        eigenvalue_lines = file_keyword_trie.find("Eigenvalues")[0: cycles]

    if len(eigenvalue_lines) != cycles:
        raise ValueError(str(n_mo) + " " + eigenvalue_type + " eigenvalues" + msg.gaussian_missing_msg())

    skip_amount = file_lines[eigenvalue_lines[0] - 1].find("-") + 2
    eigenvalues = np.zeros(n_mo, dtype=float)

    for current_block in range(cycles):
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        line = file_lines[eigenvalue_lines[current_block] - 1]
        line_values = parse_matrices.parse_matrix_line(line, skip_amount)
        eigenvalues[(n_col * current_block): (n_col * current_block) + n_col_block] = line_values

    return eigenvalues


def parse_ao_line(current_line, subshell_position):
    ao_dict = {"XX": "D", "YY": "D", "ZZ": "D", "XY": "D", "XZ": "D", "YZ": "D", "XXX": "F", "YYY": "F", "ZZZ": "F",
               "XYY": "F", "XXY": "F", "XXZ": "F", "XZZ": "F", "YZZ": "F", "YYZ": "F", "XYZ": "F"}
    subshell = current_line[subshell_position]
    atomic_orbital = current_line[subshell_position: subshell_position + 3].strip()
    if atomic_orbital in ao_dict:
        subshell = ao_dict.get(atomic_orbital)
        atomic_orbital = subshell+atomic_orbital
    line = (current_line[0:subshell_position] + " " + subshell + " " + atomic_orbital).replace(" 0", "0")
    return line.split()[1:]


def get_ao_matrix(basic, file_keyword_trie, file_lines):
    start = file_keyword_trie.find("Eigenvalues")[0] + 1
    ao_matrix = np.empty((basic.n_mo, 5), dtype='<U12')
    subshell_position = file_lines[start - 1].rfind('S')
    counter = 0
    n_lines = basic.n_mo
    if basic.scf_type == "GHF":
        n_lines *= 2
    for current_ao in range(n_lines):
        if basic.scf_type == "GHF" and current_ao % 2 != 0:
            continue
        line = file_lines[start - 1 + current_ao]
        ao_row = parse_ao_line(line, subshell_position)
        if len(ao_row) == 5:
            atom_info = ao_row[0:2]
        else:
            ao_row = atom_info + ao_row
        ao_row = np.array(ao_row, dtype='<U12')
        ao_matrix[counter, :] = ao_row
        counter += 1
    return ao_matrix

def get_overlap_matrix(basic, file_keyword_trie, file_lines) -> np.array:
    """
    Builds and returns a numpy array containing the Overlap matrix contained in the given .log file
    and converts it to square matrix form
    :return: a numpy array containing the Overlap matrix contained in the given .log file in square matrix form
    """
    overlap_matrix_state_line_num = file_keyword_trie.find("Overlap")[0] + 2
    triangular_overlap_matrix = parse_matrices.parse_matrix(file_lines, start=overlap_matrix_state_line_num, n_mo=basic.n_basis, triangular=True)
    overlap_matrix = parse_matrices.square_trig_matrix(triangular_overlap_matrix)

    if basic.scf_type == "GHF":
        identity = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        overlap_matrix = np.kron(overlap_matrix, identity)

    return overlap_matrix


def get_ghf_density_matrix(basic, file_keyword_trie, file_lines):
    temp = file_keyword_trie.find("Density matrix")[0:2]
    real_matrix_start = temp[0] + 2
    imaginary_matrix_start = temp[1] + 2
    real_matrix = parse_matrices.parse_matrix(file_lines, start=real_matrix_start, n_mo=basic.n_mo, triangular=True)
    imaginary_matrix = parse_matrices.parse_matrix(file_lines, start=imaginary_matrix_start, n_mo=basic.n_mo, triangular=True)
    return parse_matrices.combine_complex_matrix(real_matrix, imaginary_matrix)


def get_density_matrix(basic, file_keyword_trie, file_lines, cas_status=False, beta=False):
    if basic.scf_type == "GHF":
        return get_ghf_density_matrix(basic, file_keyword_trie, file_lines)
    elif beta:
        keyword = "Beta Density Matrix"
    elif basic.scf_type == "RHF" or (basic.scf_type == "ROHF" and cas_status):
        keyword = "Density Matrix"
    else:
        keyword = "Alpha Density Matrix"
    start = file_keyword_trie.find(keyword)[-1] + 2
    return parse_matrices.parse_matrix(file_lines, start=start, n_mo=basic.n_mo, last_string="S", space_skip=6, triangular=True)


def get_mo_coefficient_matrix(basic, file_keyword_trie, file_lines, n_col=5, beta=False):
    if beta:
        index = m.ceil(basic.n_mo / n_col)
    else:
        index = 0
    start = file_keyword_trie.find("Eigenvalues")[index] + 1
    mo_coefficient_matrix = parse_matrices.parse_mo_coefficient_matrix(basic, file_lines, start=start)
    if basic.scf_type == "GHF":
        real_matrix = mo_coefficient_matrix[0::2]
        imaginary_matrix = mo_coefficient_matrix[1::2]
        mo_coefficient_matrix = parse_matrices.combine_complex_matrix(real_matrix, imaginary_matrix)
    return mo_coefficient_matrix
