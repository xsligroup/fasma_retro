import numpy as np


def initialize_excitation_fields(n_excitation):
    ground_state_list = np.zeros(n_excitation, dtype=int)
    excited_state_list = np.zeros(n_excitation, dtype=int)
    delta_energy_list = np.zeros(n_excitation)
    oscillations = np.zeros(n_excitation)
    delta_diagonal_list = np.empty(n_excitation, dtype=np.ndarray)
    return ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list


def get_excitation_matrix(excitation_func) -> np.array:
    """
    Returns a numpy matrix containing the oscillation strength of every excited state from each ground state.
    :raise ValueError: incorrect number of oscillation strengths found
    :return: a numpy array containing the oscillation strength of every excited state from each ground state.
    """
    ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list = excitation_func
    return get_excitation_matrix(ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list)


def get_excitation_matrix(ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list) -> np.array:
    """
    Returns a numpy matrix containing the oscillation strength of every excited state from each ground state.
    :raise ValueError: incorrect number of oscillation strengths found
    :return: a numpy array containing the oscillation strength of every excited state from each ground state.
    """
    delta_diagonal_matrix = np.vstack(delta_diagonal_list)
    excitation_matrix = np.column_stack((ground_state_list, excited_state_list, delta_energy_list, oscillations))
    return excitation_matrix, delta_diagonal_matrix


