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

class x_abacus_input_method(MCategory):
    '''
    Parameters of INPUT belonging to section method.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_abacus_input_method'))

class x_abacus_output_method(MCategory):
    '''
    Parameters of ABACUS output of parsed INPUT belonging to section method.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_abacus_output_method'))

class x_abacus_input_run(MCategory):
    '''
    Parameters of INPUT belonging to settings run.
    '''

    m_def = Category(
        categories=[public.settings_run],
        a_legacy=LegacyDefinition(name='x_abacus_input_run'))

class x_abacus_output_run(MCategory):
    '''
    Parameters of ABACUS output of parsed INPUT belonging to settings run.
    '''

    m_def = Category(
        categories=[public.settings_run],
        a_legacy=LegacyDefinition(name='x_abacus_output_run'))

class x_abacus_section_parallel(MSection):
    '''
    section for run-time parallization options of ABACUS
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_abacus_section_parallel'))

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
        categories=[public.settings_run, x_abacus_input_run],
        a_legacy=LegacyDefinition(name='x_abacus_kpar'))

    x_abacus_bndpar = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Devide all processors into bndpar groups, and bands (only stochastic orbitals now) will be distributed among each group
        ''',
        categories=[public.settings_run, x_abacus_input_run],
        a_legacy=LegacyDefinition(name='x_abacus_bndpar'))

    x_abacus_diago_proc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        If set to a positive number, then it specifies the number of threads used for carrying out diagonalization.
        ''',
        categories=[public.settings_run, x_abacus_input_run],
        a_legacy=LegacyDefinition(name='x_abacus_diago_proc'))

class section_run(public.section_run):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_run'))

    x_abacus_input_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Filename input was read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_input_filename'))

    x_abacus_output_suffix = Quantity(
        type=str,
        shape=[],
        description='''
        Suffix of output subdirectory
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_output_suffix'))

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
        a_legacy=LegacyDefinition(name='x_abacus_stru_filename'))

    x_abacus_kpt_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where k-points were read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_kpt_filename'))

    x_abacus_basis_set_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where basis set were read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_basis_set_dirname'))

    x_abacus_pseudopotential_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where pseudopotential were read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_pseudopotential_dirname'))

    x_abacus_read_file_dirname = Quantity(
        type=str,
        shape=[],
        description='''
        Directory where files such as electron density were read from
        ''',
        a_legacy=LegacyDefinition(name='x_abacus_read_file_dirname'))

    x_abacus_input_md_time_step = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        -
        ''',
        categories=[public.settings_run, x_abacus_input_run],
        a_legacy=LegacyDefinition(name='x_abacus_input_md_time_step'))

    x_abacus_output_md_time_step = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        -
        ''',
        categories=[public.settings_run, x_abacus_output_run],
        a_legacy=LegacyDefinition(name='x_abacus_output_md_time_step'))

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

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_method'))

    x_abacus_diagonalization_algorithm = Quantity(
        type=str,
        shape=[],
        description='''
        Algorithm used in subspace diagonalization
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_diagonalization_algorithm'))

    x_abacus_dispersion_correction_method = Quantity(
        type=str,
        shape=[],
        description='''
        Calculation includes semi-empirical DFT-D dispersion correction
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_method'))

    x_abacus_dispersion_correction_s6 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        This scale factor is to optimize the interaction energy deviations
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_s6'))
    
    x_abacus_dispersion_correction_s8 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        This scale factor is only the parameter of DFTD3 approachs including D3(0) and D3(BJ).
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_s8'))
    
    x_abacus_dispersion_correction_a1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        This damping function parameter is only the parameter of DFT-D3 approachs including D3(0) and D3(BJ)
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_a1'))

    x_abacus_dispersion_correction_a2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        This damping function arameter is only the parameter of DFT-D3(BJ) approach
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_a2'))

    x_abacus_dispersion_correction_d = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        The variable is to control the dumping speed of dumping function of DFT-D2.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_d'))

    x_abacus_dispersion_correction_abc = Quantity(
        type=bool,
        shape=[],
        description='''
        The variable is to control the calculation of three-body term of DFT-D3 approachs, including D3(0) and D3(BJ).
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_abc'))

    x_abacus_dispersion_correction_C6_file = Quantity(
        type=str,
        shape=[],
        description='''
        This variable which is useful only when set vdw_method to d2 specifies the name of each elemetent's C6 parameters file
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_C6_file'))

    x_abacus_dispersion_correction_C6_unit = Quantity(
        type=str,
        shape=[],
        description='''
        This variable which is useful only when set vdw_method to d2 specifies unit of C6 parameters.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_C6_unit'))

    x_abacus_dispersion_correction_R0_file = Quantity(
        type=str,
        shape=[],
        description='''
        This variable which is useful only when set vdw_method to d2 specifies the name of each elemetent's R0 parameters file
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_R0_file'))

    x_abacus_dispersion_correction_R0_unit = Quantity(
        type=str,
        shape=[],
        description='''
        This variable which is useful only when set vdw_method to d2 specifies unit of R0 parameters.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_R0_unit'))

    x_abacus_dispersion_correction_model = Quantity(
        type=str,
        shape=[],
        description='''
        To calculate the periodic structure, you can assign the number of lattice cells calculated.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_model'))

    x_abacus_dispersion_correction_radius = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        If vdw_model is set to radius, this variable specifies the radius of the calculated sphere.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_radius'))

    x_abacus_dispersion_correction_radius_unit = Quantity(
        type=str,
        shape=[],
        description='''
        If vdw_model is set to radius, this variable specifies the unit of vdw_radius.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_radius_unit'))

    x_abacus_dispersion_correction_cn_radius = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        This cutoff is chosen for the calculation of the coordination number (CN) in DFT-D3 approachs,
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_cn_radius'))

    x_abacus_dispersion_correction_cn_radius_unit = Quantity(
        type=str,
        shape=[],
        description='''
        This variable specifies the unit of vdw_cn_radius.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_cn_radius_unit'))

    x_abacus_dispersion_correction_period = Quantity(
        type=list,
        shape=[],
        description='''
        If vdw_model is set to period, these variables specify the number of x, y and z periodic.
        ''',
        categories=[x_abacus_input_method],
        a_legacy=LegacyDefinition(name='x_abacus_dispersion_correction_period'))

