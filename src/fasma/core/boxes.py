from fasma.gaussian import parse_matrices
from fasma.core import df_generators as dfg
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class BasicData:
    atom_list: list
    scf_type: str
    n_basis: int
    n_primitive_gaussian: int
    n_alpha_electron: int
    n_beta_electron: int
    n_electron: int
    n_mo: int
    homo: int
    lumo: int


class SpectraData(ABC):
    "Spectra Data"


@dataclass
class ElectronData:
    density_matrix: np.ndarray
    mo_coefficient_matrix: np.ndarray
    eigenvalues: np.ndarray
    ao_projection_matrix: Optional[np.ndarray] = None

    def add_ao_projection_matrix(self, data: np.ndarray):
        if self.ao_projection_matrix is None:
            self.ao_projection_matrix = data


@dataclass
class BetaData(ElectronData):
    mo_coefficient_matrix: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None

    def add_beta_mo_coefficient_matrix(self, data: np.ndarray):
        if self.mo_coefficient_matrix is None:
            self.mo_coefficient_matrix = data

    def add_beta_eigenvalues(self, data: np.ndarray):
        if self.eigenvalues is None:
            self.eigenvalues = data


@dataclass
class PopData:
    ao_matrix: np.ndarray
    overlap_matrix: np.ndarray
    electron_data: ElectronData
    beta_electron_data: Optional[BetaData] = None

    def add_beta_electron_data(self, data: BetaData):
        if self.beta_electron_data is None:
            self.beta_electron_data = data


class MethodologyData(ABC):
    "Methodology Data"


@dataclass
class CASCentricData(MethodologyData):
    n_root: int
    n_slater_determinant: int
    n_excitation_full: int
    switched_orbitals: Optional[np.ndarray] = None

    def add_switched_orbitals(self, data: np.ndarray):
        if self.switched_orbitals is None:
            self.switched_orbitals = data


@dataclass
class TDCentricData(MethodologyData):
    davidson_threshold: int
    excited_state_transitions: dict


@dataclass
class ExcitationData(SpectraData):
    n_ground_state: int
    n_excited_state: int
    final_state: int
    n_excitation: int
    n_active_space_mo: int
    n_active_space_electron: int
    active_space_start: int
    active_space_end: int
    active_space: np.ndarray
    delta_diagonal_matrix: np.ndarray = None
    beta_delta_diagonal_matrix: Optional[np.ndarray] = None
    excitation_matrix: Optional[np.ndarray] = None
    methodology_data: Optional[MethodologyData] = None
    methodology: Optional[str] = None

    def initialize_active_space(self):
        self.active_space_end = self.active_space_start + self.n_active_space_mo
        self.active_space = np.array(range(self.active_space_start, self.active_space_end), dtype=int)

    def add_excitation_matrix(self, data: np.ndarray):
        if self.excitation_matrix is None:
            self.excitation_matrix = data

    def add_delta_diagonal_matrix(self, data: np.ndarray):
        if self.delta_diagonal_matrix is None:
            self.delta_diagonal_matrix = data

    def add_beta_delta_diagonal_matrix(self, data: np.ndarray):
        if self.beta_delta_diagonal_matrix is None:
            self.beta_delta_diagonal_matrix = data


@dataclass
class RealTimeData(SpectraData):
    n_step: int
    steps: pd.DataFrame


@dataclass
class TDData(ExcitationData):
    n_ground_state: int = field(init=False)
    final_state: int = field(init=False)
    n_excitation: int = field(init=False)
    active_space_start: int = field(init=False)
    active_space_end: int = field(init=False)
    active_space: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n_ground_state = 1
        self.final_state = self.n_ground_state + self.n_excited_state
        self.n_excitation = self.n_excited_state
        self.active_space_start = 0
        self.initialize_active_space()
        self.methodology = "TD"


@dataclass
class CASData(ExcitationData):
    n_excited_state: int = field(init=False)
    n_excitation: int = field(init=False)
    active_space_end: int = field(init=False)
    active_space: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n_excited_state = self.methodology_data.n_root - self.n_ground_state
        self.n_excitation = int((self.final_state * self.n_ground_state) - (self.n_ground_state * (self.n_ground_state + 1) / 2))
        self.initialize_active_space()
        self.methodology = "CAS"


@dataclass
class Box:
    basic_data: BasicData
    spectra_data: Optional[SpectraData] = None
    pop_data: Optional[PopData] = None

    def add_spectra_data(self, data: SpectraData):
        if self.spectra_data is None:
            self.spectra_data = data

    def add_pop_data(self, data: PopData):
        if self.pop_data is None:
            self.pop_data = data

    def generate_mo_analysis(self, electron: str = "alpha"):
        if self.pop_data is None:
            raise ValueError(
                "Cannot perform an MO Analysis. Please check that this object has population calculation.")
        if electron == "beta" and self.basic_data.scf_type == "UHF":
            ao_projection_matrix = self.pop_data.beta_electron_data.ao_projection_matrix
        else:
            ao_projection_matrix = self.pop_data.electron_data.ao_projection_matrix
        ao_matrix = self.pop_data.ao_matrix
        ao_df = dfg.get_ao_dataframe(ao_matrix)
        df = dfg.get_mo_dataframe(ao_projection_matrix, 'MO ')
        df = pd.concat([ao_df, df], axis=1)
        df.set_index(
            ['Atom Number', 'Atom Type', 'Principal Quantum Number', 'Subshell', 'Atomic Orbital'], inplace=True)
        return df

    def generate_mo_transition_analysis(self, electron: str = "alpha"):
        if self.spectra_data is None:
            raise ValueError(
                "Cannot perform an MO Transition Analysis. Please check that this object has an excited state calculation.")
        if electron == "beta":
            delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        else:
            delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)
        summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(delta_diagonal_matrix))
        df = dfg.get_mo_dataframe(delta_diagonal_matrix, 'AS MO ')
        df = pd.concat([excitation_df, summary_df, df], axis=1)
        df.set_index(['Starting State', 'Ending State'], inplace=True)
        return df

    def generate_merged_mo_transition_analysis(self):
        if self.spectra_data is None:
            raise ValueError(
                "Cannot perform an MO Transition Analysis. Please check that this object has an excited state calculation.")
        alpha_delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        alpha_delta_mo_transition_df = dfg.get_mo_dataframe(alpha_delta_diagonal_matrix, 'AS MO ')
        #alpha_summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(alpha_delta_diagonal_matrix))

        beta_delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        beta_mo_transition_df = dfg.get_mo_dataframe(beta_delta_diagonal_matrix, 'Beta AS MO ')
        #beta_summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(beta_delta_diagonal_matrix))

        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)

        df = pd.concat([excitation_df, alpha_delta_mo_transition_df, beta_mo_transition_df], axis=1)
        df.set_index(['Starting State', 'Ending State'], inplace=True)
        return df

    def generate_ao_transition_analysis(self, electron: str = "alpha"):
        if self.pop_data is None and self.spectra_data is None:
            raise ValueError(
                "Cannot perform an AO Projection Transition Analysis. Please check this object has both a population calculation and an excited state calculation.")
        ao_matrix = self.pop_data.ao_matrix
        ao_matrix = parse_matrices.convert_ao_projection_to_mo_transition(self.spectra_data.n_excitation, ao_matrix)
        ao_df = dfg.get_ao_dataframe(ao_matrix)

        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_matrix = parse_matrices.convert_mo_transition_to_ao_projection(self.basic_data.n_mo, self.spectra_data.n_excitation,  excitation_matrix)
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)
        ao_transition_matrix = self.generate_ao_transition_matrix(electron)
        summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(ao_transition_matrix))
        df = dfg.get_mo_dataframe(ao_transition_matrix, 'AS MO ')
        df = pd.concat([excitation_df, ao_df, summary_df, df], axis=1)
        index_list = ['Starting State', 'Ending State', 'Atom Number', 'Atom Type', 'Principal Quantum Number',
             'Subshell', 'Atomic Orbital']
        df.set_index(index_list, inplace=True)
        return df

    def generate_ao_transition_matrix(self, electron: str = "alpha", swap_orbitals: bool = False):
        if self.pop_data is None and self.spectra_data is None:
            raise ValueError("Cannot perform an AO Projection Transition Analysis. Please check this object has both a population calculation and an excited state calculation.")
        if electron == "beta":
            ao_projection_matrix = np.copy(self.pop_data.beta_electron_data.ao_projection_matrix).real
            delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        else:
            ao_projection_matrix = np.copy(self.pop_data.electron_data.ao_projection_matrix).real
            delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        if self.spectra_data.methodology == "CAS" and self.spectra_data.methodology_data.switched_orbitals is not None and swap_orbitals:
            ao_projection_matrix = parse_matrices.swap_ao_projection_orbitals(ao_projection_matrix, self.spectra_data.methodology_data.switched_orbitals)
        ao_projection_matrix = ao_projection_matrix[:, self.spectra_data.active_space_start: self.spectra_data.active_space_end]
        ao_projection_matrix = parse_matrices.convert_ao_projection_to_mo_transition(self.spectra_data.n_excitation, ao_projection_matrix)
        delta_diagonal_matrix = parse_matrices.convert_mo_transition_to_ao_projection(self.basic_data.n_mo, self.spectra_data.n_excitation, delta_diagonal_matrix)
        return np.multiply(ao_projection_matrix, delta_diagonal_matrix)





