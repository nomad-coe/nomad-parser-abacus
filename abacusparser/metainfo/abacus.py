#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.metainfo.legacy import LegacyDefinition

from nomad.datamodel.metainfo import public

m_package = Package(
    name='abacus_nomadmetainfo_json',
    description='None',
    a_legacy=LegacyDefinition(name='abacus.nomadmetainfo.json'))


class x_abacus_input_settings(MCategory):
    '''
    Parameters of INPUT.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_abacus_input_settings'))


class x_abacus_output_settings(MCategory):
    '''
    Parameters of ABACUS output of parsed INPUT.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_abacus_output_settings'))


class x_abacus_section_parallel(MSection):
    '''
    section for run-time parallization options of ABACUS
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(
        name='x_abacus_section_parallel'))

    x_abacus_nproc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of processors
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_nproc'))

    x_abacus_kpar = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Devide all processors into kpar groups, and k points will be distributed among each group. 
        ''',
        categories=[public.settings_run, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_kpar'))

    x_abacus_bndpar = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Devide all processors into bndpar groups, and bands (only stochastic orbitals now) will be distributed among each group
        ''',
        categories=[public.settings_run, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_bndpar'))

    x_abacus_diago_proc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        If set to a positive number, then it specifies the number of threads used for carrying out diagonalization.
        ''',
        categories=[public.settings_run, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_diago_proc'))

class x_abacus_section_pseudopotential(MSection):
    '''
    pseudo-section for collecting pseudopotential data
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_abacus_section_pseudopotential'))

    x_abacus_pp_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Filename of pseudopotential
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_filename'))

    x_abacus_pp_xc = Quantity(
        type=str,
        shape=[],
        description='''
        Exchange-correlation functional of pseudopotential, e.g. 'PBE' or 'PZ'
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_xc'))

    x_abacus_pp_type = Quantity(
        type=str,
        shape=[],
        description='''
        Type of pseudopotential, e.g. 'Norm-conserving' or 'Ultrasoft'
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_type'))

    x_abacus_pp_valence = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of Valence electrons in pseudopotential
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_valence'))

    x_abacus_pp_lmax = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Maximum angular momentum component in pseudopotential
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_lmax'))

    x_abacus_pp_nzeta = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of wavefunctions in pseudopotential
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_nzeta'))

    x_abacus_pp_nprojectors = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of projectors in pseudopotential
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pp_nprojectors'))

    x_abacus_pao_radial_cutoff = Quantity(
        type=np.dtype(np.float64),
        unit='bohr',
        shape=[],
        description='''
        Radial cut-off of pseudo atomic orbital
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pao_radial_cutoff'))

    x_abacus_npao = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of pseudo atomic orbital
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_npao'))

class section_single_configuration_calculation(public.section_single_configuration_calculation):

    x_abacus_md_step_input = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[public.settings_molecular_dynamics,
                    x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_md_step_input'))

    x_abacus_md_step_output = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[public.settings_molecular_dynamics,
                    x_abacus_output_settings],
        a_legacy=LegacyDefinition(name='x_abacus_md_step_output'))

    x_abacus_initial_magnetization_total = Quantity(
        type=np.dtype(np.float64),
        unit='bohr_magneton',
        shape=[],
        description='''
        Initial total magnetization of the system set in INPUT.
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_magnetization_total_input'))

    x_abacus_magnetization_total = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Total per-cell magnetization
        ''',
        a_legacy=LegacyDefinition(name='x_qe_magnetization_total'))

    x_abacus_magnetization_absolute = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Absolute per-cell magnetization
        ''',
        a_legacy=LegacyDefinition(name='x_qe_magnetization_absolute'))


class section_run(public.section_run):

    m_def = Section(validate=False, extends_base_section=True,
                    a_legacy=LegacyDefinition(name='section_run'))

    x_abacus_input_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Filename input was read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_input_filename'))

    x_abacus_outdir_suffix = Quantity(
        type=str,
        shape=[],
        description='''
        Suffix of output subdirectory
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_outdir_suffix'))

    x_abacus_program_execution_date = Quantity(
        type=str,
        shape=[],
        description='''
        The date on which the program execution started
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_program_execution_date'))

    x_abacus_program_execution_time = Quantity(
        type=str,
        shape=[],
        description='''
        The duration of the program execution
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_program_execution_time'))

    x_abacus_stru_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where initial atom_positions and lattice_vectors were read from
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_stru_filename'))

    x_abacus_kpt_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where k-points were read from
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_kpt_filename'))

    x_abacus_basis_set_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where basis set were read from
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_basis_set_dirname'))

    x_abacus_pseudopotential_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where pseudopotential were read from
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_pseudopotential_dirname'))

    x_abacus_read_file_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where files such as electron density were read from
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_read_file_dirname'))

    x_abacus_geometry_optimization_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Determines whether a geoemtry optimization is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_geometry_optimization_converged'))

    x_abacus_section_parallel = SubSection(
        sub_section=SectionProxy('x_abacus_section_parallel'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_abacus_section_parallel'))


class section_method(public.section_method):

    m_def = Section(validate=False, extends_base_section=True,
                    a_legacy=LegacyDefinition(name='section_method'))

    x_abacus_diagonalization_algorithm = Quantity(
        type=str,
        shape=[],
        description='''
        Algorithm used in subspace diagonalization
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_diagonalization_algorithm'))

    x_abacus_dispersion_correction_method = Quantity(
        type=str,
        shape=[],
        description='''
        Calculation includes semi-empirical DFT-D dispersion correction
        ''',
        categories=[public.settings_van_der_Waals, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_method'))

    x_abacus_occupations_method = Quantity(
        type=str,
        shape=[],
        description='''
        Specifies how to calculate the occupations of bands.
        ''',
        categories=[public.settings_smearing, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_occupations_method'))

    x_abacus_smearing_method = Quantity(
        type=str,
        shape=[],
        description='''
        It indicates which occupation and smearing method is used in the calculation
        ''',
        categories=[public.settings_smearing, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_smearing_method'))

    x_abacus_smearing_sigma = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='rydberg',
        description='''
        Energy range for smearing
        ''',
        categories=[public.settings_smearing, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_smearing_sigma'))

    x_abacus_mixing_method = Quantity(
        type=str,
        shape=[],
        description='''
        Charge mixing methods
        ''',
        categories=[public.settings_scf, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_mixing_method'))

    x_abacus_mixing_beta = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Mixing method: parameter beta
        ''',
        categories=[public.settings_scf, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_mixing_beta'))

    x_abacus_gamma_algorithms = Quantity(
        type=bool,
        shape=[],
        description='''
        Usage of gamma-only optimized algorithms
        ''',
        categories=[public.settings_scf, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_gamma_algorithms'))

    x_abacus_scf_nmax = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        This variable indicates the maximal iteration number for electronic iterations.
        ''',
        categories=[public.settings_scf, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_scf_nmax'))

    x_abacus_scf_thr = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        The threshold for electronic iteration
        ''',
        categories=[public.settings_scf, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_scf_thr'))

    x_abacus_chg_extrap = Quantity(
        type=str,
        shape=[],
        description='''
        Methods to do extrapolation of density when ABACUS is doing geometry relaxations
        ''',
        categories=[public.settings_molecular_dynamics,
                    public.settings_geometry_optimization, x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_chg_extrap'))

    x_abacus_sto_method = Quantity(
        type=str,
        shape=[],
        description='''
        Methods to do SDFT
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_sto_method'))

    x_abacus_xc_functional = Quantity(
        type=str,
        shape=[],
        description='''
        Type of exchange-correlation functional used in calculation. 
        ''',
        categories=[x_abacus_input_settings, public.settings_XC, public.settings_potential_energy_surface, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_abacus_xc_functional'))

    x_abacus_section_pseudopotential = SubSection(
        sub_section=SectionProxy('x_abacus_section_pseudopotential'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_abacus_section_pseudopotential'))


class section_system(public.section_system):

    m_def = Section(validate=False, extends_base_section=True,
                    a_legacy=LegacyDefinition(name='section_system'))

    x_abacus_number_of_species = Quantity(
        type=str,
        shape=[],
        description='''
        Number of different atom species in this calculations.
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_number_of_species'))

    x_abacus_lattice_name_input = Quantity(
        type=str,
        shape=[],
        description='''
        Specifies the type of Bravias lattice in INPUT.
        ''',
        categories=[x_abacus_input_settings],
        a_legacy=LegacyDefinition(name='x_abacus_lattice_name_input'))

    x_abacus_lattice_name_output = Quantity(
        type=str,
        shape=[],
        description='''
        The type of Bravias lattice checked by ABACUS when open symmetry calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_lattice_name_output'))

    x_abacus_number_of_electrons_input = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        This denotes total number of electrons in the system set in INPUT.
        ''',
        categories=[x_abacus_input_settings, public.configuration_core],
        a_legacy=LegacyDefinition(name='x_abacus_number_of_electrons_input'))

    x_abacus_number_of_electrons_output = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        This denotes total number of electrons in the system calculated by ABACUS.
        ''',
        categories=[x_abacus_output_settings, public.configuration_core],
        a_legacy=LegacyDefinition(name='x_abacus_number_of_electrons_output'))


m_package.__init_metainfo__()
