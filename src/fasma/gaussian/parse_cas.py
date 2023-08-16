from fasma.core import boxes as bx
from fasma.core import messages as msg
from fasma.gaussian import parse_excitation
from fasma.gaussian import parse_functions
from fasma.gaussian import parse_matrices
import numpy as np


def check_cas(basic, file_keyword_trie, file_lines) -> bool:
    """
    Checks if .log contains a CAS calculation.
    If yes, initializes and returns a CASData object.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :param basic: the BasicData object for this current file
    :return: CASData object if this .log file contains a CAS calculation, none otherwise
    """
    try:
        temp_list = parse_functions.find_iop(file_keyword_trie, file_lines, "9", ["6", "7", "13", "17", "19"])
    except ValueError:
        pass
    else:
        n_active_space_electron = temp_list[0]
        n_active_space_mo = temp_list[1]
        de_min_status = temp_list[2]
        n_root = temp_list[3]
        n_ground_state = temp_list[4]
        n_slater_determinant = int(file_lines[file_keyword_trie.find("NDet=")[0] - 1].split("NDet=")[1])
        final_state_full = n_slater_determinant
        active_space_start = basic.homo - n_active_space_electron

        if de_min_status == 1:
            # Check n_root over 9999 (Davidson iterative diagonalization DEMin case)
            if n_root > 9999:
                temp = str(n_root)
                front = int(temp[:-4])
                back = int(temp[-4:])
                n_root = front + back
            final_state_full = n_root

        final_state = n_root
        n_excitation_full = int((final_state_full * n_ground_state) - (n_ground_state * (n_ground_state + 1) / 2))

        cas_data = bx.CASCentricData(n_root=n_root, n_slater_determinant=n_slater_determinant, n_excitation_full=n_excitation_full)
        try:
            # Remember to swap orbitals later
            switched_orbitals = find_switched_orbitals(file_keyword_trie, file_lines)
        except ValueError:
            pass
        else:
            cas_data.add_switched_orbitals(switched_orbitals)
        excitation_data = bx.CASData(n_ground_state=n_ground_state, final_state=final_state,
                                     n_active_space_mo=n_active_space_mo,
                                     n_active_space_electron=n_active_space_electron,
                                     active_space_start=active_space_start, methodology_data=cas_data)
        excitation_matrix, delta_diagonal_matrix = parse_excitation.get_excitation_matrix(*get_excitations_cas(file_keyword_trie, file_lines, excitation_data))
        excitation_data.add_excitation_matrix(excitation_matrix)
        excitation_data.add_delta_diagonal_matrix(delta_diagonal_matrix)
        return excitation_data


def get_excitations_cas(file_keyword_trie, file_lines, excitation_data):
    ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list = parse_excitation.initialize_excitation_fields(
        excitation_data.n_excitation)
    num_of_results = 0
    diag_lines, energy_lines, osc_lines = verify_cas_completeness(file_keyword_trie, excitation_data.methodology_data.n_root,
                                                                  excitation_data.methodology_data.n_excitation_full)
    for x in range(excitation_data.n_ground_state):
        current_degenerate_state = x + 1
        current_degenerate_diag = dm_get_diag(file_lines, diag_lines[x], excitation_data.n_active_space_mo)
        current_degenerate_energy = dm_get_hartree(file_lines, energy_lines[x])
        for y in range(excitation_data.final_state - 1 - x):
            current_excited_state = x + y + 2
            ground_state_list[num_of_results] = current_degenerate_state
            excited_state_list[num_of_results] = current_excited_state
            delta_diagonal_list[num_of_results], delta_energy_list[num_of_results], oscillations[
                num_of_results] = cas_get_excited_state(file_lines, current_degenerate_diag, current_degenerate_energy,
                                                        diag_lines[current_excited_state - 1],
                                                        energy_lines[current_excited_state - 1],
                                                        osc_lines[num_of_results], excitation_data.n_active_space_mo)
            num_of_results += 1
    return ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list


def verify_cas_completeness(file_keyword_trie, n_root, n_excitation):
    diag_lines = file_keyword_trie.find("diagonals of 1PDM for State: ")
    if len(diag_lines) != n_root:
        raise ValueError(
            str(n_root) + " PDM diagonals" + msg.gaussian_missing_msg())

    energy_lines = file_keyword_trie.find("Energy (Hartree)")
    if len(energy_lines) != n_root:
        raise ValueError(str(n_root) + " state energies" + msg.gaussian_missing_msg())

    osc_lines = file_keyword_trie.find("Oscillator Strength For States")
    if len(osc_lines) != n_excitation:
        raise ValueError(str(n_excitation) + " oscillations" + msg.gaussian_missing_msg())
    return diag_lines, energy_lines, osc_lines


def cas_get_excited_state(file_lines, ground_diag, ground_energy, diag_line_num, energy_line_num, osc_line_num, n_active_space_mo):
    delta_diagonal = dm_get_diag(file_lines, diag_line_num, n_active_space_mo) - ground_diag
    energy_value = dm_get_hartree(file_lines, energy_line_num) - ground_energy
    osc_value = float(file_lines[osc_line_num - 1].split()[8])

    return delta_diagonal, energy_value, osc_value


def find_switched_orbitals(file_keyword_trie, file_lines) -> np.array:
    """
    Find and checks if there are switched orbitals. If so, returns a matrix containing the switched orbital pairs.
    :raise ValueError: No orbitals were switched
    :return: a numpy array containing matrix with the switched orbital pairs
    """
    try:
        start = file_keyword_trie.find("orbitals")[0] + 1
        end = file_keyword_trie.find("MCSCF")[0] - 4
    except TypeError:
        raise ValueError("No orbitals were switched in this CAS calculation.")

    amt_switched = end - start
    matrix = np.zeros((amt_switched, 2), dtype=int)

    for x in range(start - 1, end - 1):
        line = file_lines[x].split()
        matrix[x - (start - 1), :] = line
    return matrix - 1


def dm_get_hartree(file_lines, line_num):
    current_line = file_lines[line_num - 1].split()
    return float(current_line[4]) * 27.2114


def dm_get_diag(file_lines, line_num, n_active_space_mo):
        start = line_num + 2
        skip_amount = file_lines[start - 1].find("1") + 2
        diag = (parse_matrices.parse_matrix_block(file_lines, start, n_active_space_mo, 1, skip_amount)).reshape((n_active_space_mo,))
        return diag
