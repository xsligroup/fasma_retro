import numpy as np
import math as m


def parse_mo_coefficient_matrix(basic, file_lines, start: int, last_string="S", space_skip=6, n_col=5, block_skip=3):
    n_row_block = basic.n_mo
    if basic.scf_type == "GHF":
        n_row_block *= 2
        block_skip = 2
    matrix = np.zeros((n_row_block, basic.n_mo))
    cycles = m.ceil(basic.n_mo / n_col)
    n_col_block = n_col
    n_last_block_col = basic.n_mo % n_col
    skip_amount = file_lines[start - 1].rfind(last_string) + space_skip

    for current_block in range(cycles):  # Number of separated matrix blocks
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        end = start + n_row_block
        current_block_matrix = parse_matrix_block(file_lines, start, n_row_block, n_col_block, skip_amount)
        matrix[:, (n_col * current_block): (n_col * current_block) + n_col_block] = current_block_matrix
        start = end + block_skip
    return matrix


def square_trig_matrix(current_matrix: np.array) -> np.array:
    """
    Returns numpy array containing a square matrix transformed from a given triangular matrix.
    :param current_matrix: the given triangular matrix to be converted
    :return: a numpy array containing a square matrix transformed from a given triangular matrix
    """
    triangular_matrix = current_matrix
    output = triangular_matrix.transpose() + triangular_matrix
    np.fill_diagonal(output, np.diag(triangular_matrix))
    return output


def parse_matrix_line(current_line, skip_amount) -> np.array:
    line = current_line[skip_amount:].replace("**********", "   nan")
    if "D" in current_line:
        line = line.replace("D", "E").split()
    else:
        line = line.replace("-", " -").split()
    try:
        value_array = np.asarray(line, dtype=float)
    except ValueError:
        print(current_line)
    return value_array


def parse_matrix_block(file_lines, start, n_row, n_col, skip_amount):
    block_matrix = np.zeros((n_row, n_col))
    for current_row in range(n_row):  # Number of rows
        line = file_lines[start - 1 + current_row]
        line_values = parse_matrix_line(line, skip_amount)
        block_matrix[current_row, : len(line_values)] = line_values
    return block_matrix


def parse_matrix(file_lines, start: int, n_mo, last_string="1", space_skip=2, n_col=5, block_skip=1, triangular=False):
    # Parse cartesian_axes, last_string="s", space_skip=6
    matrix = np.zeros((n_mo, n_mo))
    cycles = m.ceil(n_mo / n_col)
    n_row_block = n_mo
    n_col_block = n_col
    n_last_block_col = n_mo % n_col
    if last_string == "S":
        skip_amount = file_lines[start - 1].rfind(last_string) + space_skip
    else:
        skip_amount = file_lines[start - 1].find(last_string) + space_skip

    for current_block in range(cycles):  # Number of separated matrix blocks
        if triangular:
            n_row_block = n_mo - (current_block * n_col)
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        end = start + n_row_block
        current_block_matrix = parse_matrix_block(file_lines, start, n_row_block, n_col_block, skip_amount)
        matrix[n_mo - n_row_block:, (n_col * current_block): (n_col * current_block) + n_col_block] = current_block_matrix
        start = end + block_skip
    return matrix


def combine_complex_matrix(real_matrix, imaginary_matrix):
    return real_matrix + 1j * imaginary_matrix


def convert_ao_projection_to_mo_transition(n_excitation, data):
    return np.tile(data, (n_excitation, 1))


def convert_mo_transition_to_ao_projection(n_mo, n_excitation, data):
    return np.tile(data, (1, n_mo)).reshape((n_excitation * n_mo, data.shape[1]))


def summarize_matrix(matrix):
    particle_diagonal_matrix = np.where(matrix[:, :] > 0, matrix[:, :], 0)
    hole_diagonal_matrix = np.where(matrix[:, :] < 0, matrix[:, :], 0)

    summary_matrix = np.ndarray((matrix.shape[0], 3))
    summary_matrix[:, 0] = np.sum(matrix, axis=1)
    summary_matrix[:, 1] = np.sum(particle_diagonal_matrix, axis=1)
    summary_matrix[:, 2] = np.sum(hole_diagonal_matrix, axis=1)

    return summary_matrix


def swap_ao_projection_orbitals(ao_projection_matrix, swapped_orbitals):
    swapped_matrix = ao_projection_matrix
    for i in range(swapped_orbitals.shape[0]):
        col_1 = swapped_orbitals[i, 0]
        col_2 = swapped_orbitals[i, 1]
        swapped_matrix[:, [col_1, col_2]] = swapped_matrix[:, [col_2, col_1]]
    return swapped_matrix
