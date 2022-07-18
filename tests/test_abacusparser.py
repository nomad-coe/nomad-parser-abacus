#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest

from nomad.datamodel import EntryArchive
from abacusparser import ABACUSParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return ABACUSParser()


def test_band(parser):
    archive = EntryArchive()
    parser.parse(r'data\Si_band\running_nscf.log', archive, None)

    sec_run = archive.section_run[0]
    assert sec_run.program_version == 'Parallel, in development'
    assert sec_run.program_basis_set_type == 'Numeric AOs'
    assert sec_run.time_run_date_start.magnitude == approx(1657036247.0)
    sec_parallel = sec_run.x_abacus_section_parallel[0]
    assert sec_parallel.x_abacus_nproc == 8
    assert sec_parallel.x_abacus_allocation_method == '2D block'
    assert sec_parallel.x_abacus_allocation_nb2d == 1
    assert sec_parallel.x_abacus_allocation_trace_loc_row == 26
    assert sec_parallel.x_abacus_allocation_trace_loc_col == 26
    assert sec_parallel.x_abacus_allocation_nloc == 91
    assert sec_run.x_abacus_input_filename == 'INPUT'
    assert sec_run.x_abacus_kpt_filename == 'KLINES'
    assert sec_run.section_basis_set_cell_dependent[0].basis_set_cell_dependent_name == 'PW_50.0'
    assert sec_run.section_basis_set_cell_dependent[1].basis_set_planewave_cutoff.magnitude == approx(4.35974472220717e-16)
    assert sec_run.section_sampling_method[0].sampling_method == 'geometry_optimization'
    assert sec_run.time_run_date_end.magnitude == approx(1657036249.0)
    assert sec_run.run_clean_end

    sec_method = sec_run.section_method[0]
    assert sec_method.x_abacus_number_of_pw_for_wavefunction == 2085
    assert sec_method.x_abacus_number_of_sticks_for_density == 721
    sec_basis_sets = sec_method.x_abacus_section_basis_sets[0]
    assert sec_basis_sets.x_abacus_basis_sets_delta_k.magnitude == 0.01
    assert sec_basis_sets.x_abacus_basis_sets_delta_r.magnitude == 0.01
    assert sec_basis_sets.x_abacus_basis_sets_dr_uniform.magnitude == 0.001
    assert sec_basis_sets.x_abacus_basis_sets_rmax.magnitude == 30
    assert sec_basis_sets.x_abacus_basis_sets_kmesh.magnitude == 711
    sec_specie_basis_set = sec_basis_sets.x_abacus_section_specie_basis_set
    assert len(sec_specie_basis_set) == 1
    assert sec_specie_basis_set[0].x_abacus_specie_basis_set_filename == 'Si_lda_8.0au_50Ry_2s2p1d'
    assert sec_specie_basis_set[0].x_abacus_specie_basis_set_ln == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
    assert sec_specie_basis_set[0].x_abacus_specie_basis_set_rcutoff.magnitude == 8
    assert sec_specie_basis_set[0].x_abacus_specie_basis_set_rmesh == 801
    assert sec_method.number_of_spin_channels == 1
    assert sec_method.x_abacus_spin_orbit == False
    assert sec_method.relativity_method == 'scalar relativistic'
    sec_method_atom_kind = sec_method.section_method_atom_kind
    assert len(sec_method_atom_kind) == 1
    assert sec_method_atom_kind[0].method_atom_kind_label == 'Si'
    assert sec_method_atom_kind[0].method_atom_kind_explicit_electrons == 4
    assert sec_method_atom_kind[0].method_atom_kind_pseudopotential_name == 'Si.pz-vbc.UPF'
    assert sec_method_atom_kind[0].x_abacus_pp_type == 'NC'
    assert sec_method_atom_kind[0].x_abacus_pp_xc == 'PZ'
    assert sec_method_atom_kind[0].x_abacus_pp_lmax == 1
    assert sec_method_atom_kind[0].x_abacus_pp_nzeta == 2
    assert sec_method_atom_kind[0].x_abacus_pp_nprojectors == 2
    assert len(sec_method.section_XC_functionals) == 2
    assert sec_method.section_XC_functionals[0].XC_functional_name == 'LDA_C_PZ'

    sec_system = sec_run.section_system[0]
    assert sec_system.x_abacus_alat.magnitude == 10.2
    assert sec_system.lattice_vectors[0][1].magnitude == approx(2.69880378e-10)
    assert sec_system.atom_labels == ['Si']
    assert sec_system.atom_positions[1][0].magnitude == approx(1.34940189e-10)
    assert sec_system.x_abacus_cell_volume.magnitude == 265.302
    assert sec_system.x_abacus_reciprocal_vectors[2][0].magnitude == approx(-1.16406857e+10)
    assert sec_system.number_of_atoms == 1
    assert sec_system.number_of_species == 1
    assert sec_system.number_of_electrons_out == [8]

    sec_scc = sec_run.section_single_configuration_calculation[0]
    assert sec_scc.x_abacus_longest_orb_rcut.magnitude == 8
    assert sec_scc.x_abacus_longest_nonlocal_projector_rcut.magnitude == 5.01
    assert sec_scc.x_abacus_searching_radius.magnitude == 26
    assert sec_scc.x_abacus_searching_radius_unit.magnitude == 10.2
    assert sec_scc.x_abacus_read_space_grid == [36, 36, 36]
    assert sec_scc.x_abacus_big_cell_numbers_in_grid == [18, 18, 18]
    assert sec_scc.x_abacus_meshcell_numbers_in_big_cell == [2, 2, 2]
    assert sec_scc.x_abacus_extended_fft_grid == [25, 25, 25]
    assert sec_scc.x_abacus_extended_fft_grid_dim == [69, 69, 69]
    sec_k_band = sec_scc.section_k_band[0]
    assert sec_k_band.band_structure_kind == 'electronic'
    assert sec_k_band.reciprocal_cell[0][0].magnitude == approx(1.16406857e+10)
    sec_k_band_segment = sec_k_band.section_k_band_segment[0]
    assert sec_k_band_segment.band_k_points[3][2] == 0.425
    assert sec_k_band_segment.band_energies[0][4][4].magnitude == approx(1.14715847e-18)

if __name__ == '__main__':
    test_parser = parser()
    test_band(test_parser)