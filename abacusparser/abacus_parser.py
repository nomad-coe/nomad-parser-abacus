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

import re
import numpy as np
from collections import namedtuple

from nomad.units import ureg
from nomad.parsing import FairdiParser
from nomad.parsing.file_parser import TextParser, Quantity

units_mapping = {'Ha': ureg.hartree, 'Ry': ureg.rydberg, 'eV': ureg.eV,
                 'bohr': ureg.bohr, 'A': ureg.angstrom, 'fs': ureg.fs, 'polar': ureg.C/ureg.meter**2}

class ABACUSInputParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):

        self._quantities = [

        ]

class ABACUSOutParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def str_to_sites(val_in):
            data = dict()
            val = [v.strip().split() for v in val_in.split('\n')][0]
            if len(val) == 5:
                labels, x, y, z, mag = val
            elif len(val) == 8:
                labels, x, y, z, mag, vx, vy, vz = val
                data['velocities'] = np.array(
                    [vx, vy, vz], dtype=float) * units_mapping['A'] / units_mapping['fs']
            data['labels'] = labels
            data['positions'] = np.array([x, y, z], dtype=float)
            data['magnetic_moments'] = float(mag)
            return data

        def str_to_coordclass(val_in):
            if val_in == 'DIRECT':
                name = 'direct'
            elif val_in == 'CARTESIAN':
                name = 'cartesian'
            return name

        def str_to_matrix(val_in):
            val = [v.strip().split() for v in val_in.split('\n')]
            return np.reshape(val, (3, 3)).astype(float)

        def str_to_dict(val_in):
            data = dict()
            val = val_in.split()
            data[val[0]] = int(val[1])
            return data

        def str_to_kpoints(val_in):
            lines = re.search(
                rf'KPOINTS\s*DIRECT_X\s*DIRECT_Y\s*DIRECT_Z\s*WEIGHT([\s\S]+?)DONE', val_in).group(1).strip().split('\n')
            data = []
            for line in lines:
                data.append(line.strip().split()[1:5])
            kpoints, weights, _ = np.split(
                np.array(data, dtype=float), [3, 4], axis=1)
            return kpoints, weights.flatten()

        def str_to_sticks(val_in):
            Data = namedtuple('PW', ['proc', 'columns', 'pw'])
            val = re.findall(
                rf'\s+({re_float})\s+({re_float})\s+({re_float})\n', val_in)
            data = []
            for v in val:
                data.append(
                    Data(proc=int(v[0]), columns=int(v[1]), pw=int(v[2])))
            return data

        def str_to_orbital(val_in):
            Data = namedtuple(
                'Orbital', ['index', 'l', 'n', 'nr', 'dr', 'rcut', 'check_unit', 'new_unit'])
            val = re.findall(
                rf'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\n', val_in)
            data = []
            for v in val:
                data.append(Data(index=int(v[0]), l=int(v[1]), n=int(v[2]), nr=int(v[3]), dr=float(
                    v[4]), rcut=float(v[5]), check_unit=float(v[6]), new_unit=float(v[7])))
            return data

        def str_to_energy_occupation(val_in):
            def extract_data(val_in, nks):
                State = namedtuple(
                    'State', ['kpoint', 'energies', 'occupations', 'npws'])
                data = []
                for i in range(nks):
                    kx, ky, kz, npws = re.search(
                        rf'{i+1}/{nks} kpoint \(Cartesian\)\s*=\s*({re_float})\s*({re_float})\s*({re_float})\s*\((\d+)\s*pws\)', val_in).groups()
                    _, energies, occupations = np.split(np.array(list(map(lambda x: x.strip().split(), re.search(
                        rf'{i+1}/{nks} kpoint \(Cartesian\)\s*=.*\n([\s\S]+?)\n\n', val_in).group(1).split('\n'))), dtype=float), 3, axis=1)
                    state = State(kpoint=np.array([kx, ky, kz], dtype=float), energies=energies.flatten().astype(
                        float)*units_mapping['eV'], occupations=occupations.flatten().astype(float), npws=int(npws))
                    data.append(state)
                return data

            nspin = int(re.search(
                r'STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*(\d+)', val_in).group(1))
            nks = int(
                re.search(r'\d+/(\d+) kpoint \(Cartesian\)', val_in).group(1))
            data = dict()
            if nspin in [1, 4]:
                data['up'] = extract_data(val_in, nks)
            elif nspin == 2:
                val_up = re.search(
                    r'SPIN UP :([\s\S]+?)\n\nSPIN', val_in).group()
                data['up'] = extract_data(val_up, nks)
                val_dw = re.search(
                    r'SPIN DOWN :([\s\S]+?)(?:\n\n\s*EFERMI|\n\n\n)', val_in).group()
                data['down'] = extract_data(val_dw, nks)
            return data

        def str_to_bandstructure(val_in):
            def extract_data(val_in, nks):
                State = namedtuple('State', ['kpoint', 'energies'])
                data = []
                for i in range(nks):
                    kx, ky, kz = re.search(
                        rf'k\-points{i+1}\(\d+\):\s*({re_float})\s*({re_float})\s*({re_float})', val_in).groups()
                    _, _, energies, _ = np.split(np.array(list(map(lambda x: x.strip().split(), re.search(
                        rf'k\-points{i+1}\(\d+\):.*\n([\s\S]+?)\n\n', val_in).group(1).split('\n')))), 4, axis=1)
                    state = State(kpoint=np.array([kx, ky, kz], dtype=float), energies=energies.flatten(
                    ).astype(float)*units_mapping['eV'])
                    data.append(state)
                return data

            nks = int(re.search(r'k\-points\d+\((\d+)\)', val_in).group(1))
            data = dict()
            if re.search('spin up', val_in) and re.search('spin down', val_in):
                val = re.search(r'spin up :\n([\s\S]+?)\n\n\n', val_in).group()
                val_new = extract_data(val, nks)
                data['up'] = val_new[:int(nks/2)]
                data['down'] = val_new[int(nks/2):]
            else:
                data['up'] = extract_data(val_in, nks)
            return data

        def str_to_force(val_in):
            data = dict()
            val = [v.strip().split() for v in val_in.split('\n')]
            for v in val:
                data[v[0]] = np.array(v[1:], dtype=float) * \
                    units_mapping['eV'] / units_mapping['A']
            return data

        def str_to_polarization(val_in):
            P = namedtuple('P', ['value', 'mod', 'vector'])
            data = np.array(val_in.split(), dtype=float)
            return P(value=data[0]*units_mapping['polar'], mod=data[1], vector=data[2:]*units_mapping['polar'])

        re_float = r'[\d\.\-\+Ee]+'

        atom_quantities = [
            Quantity(
                'label', r'atom label\s*=\s*(\w+)'
            ),
            Quantity(
                'orbital', r'L=\d+, number of zeta\s*=\s*(\d+)', repeats=True
            ),
            Quantity(
                'natoms', r'number of atom for this type\s*=\s*(\d+)',
            ),
            Quantity(
                'start_magnetization', r'start magnetization\s*=\s*(\w+)', repeats=True,
                str_operation=lambda x: True if x == 'TRUE' else False
            )
        ]

        structure_quantities = [
            Quantity(
                'sites', rf'tau[cd]_([a-zA-Z]+)\d+\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})|tau[cd]_([a-zA-Z]+)\d+\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+',
                dtype=float, repeats=True,  str_operation=str_to_sites
            ),
            Quantity(
                'units', rf'UNIT = ({re_float}) Bohr',
                dtype=float, repeats=False, unit='bohr'
            ),
            Quantity(
                'coord_class', r'(DIRECT) COORDINATES|(CARTESIAN) COORDINATES',
                repeat=False, str_operation=str_to_coordclass
            )
        ]

        symmetry_quantities = [
            Quantity(
                'lattice_vectors',
                rf'LATTICE VECTORS: \(CARTESIAN COORDINATE: IN UNIT OF A0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, convert=False, repeats=False,
            ),
            Quantity(
                'right_hand_lattice',
                r'right hand lattice\s*=\s*(\d+)'
            ),
            Quantity(
                'norm_a',
                rf'NORM_A\s*=\s*({re_float})',
            ),
            Quantity(
                'norm_b',
                rf'NORM_B\s*=\s*({re_float})',
            ),
            Quantity(
                'norm_c',
                rf'NORM_C\s*=\s*({re_float})',
            ),
            Quantity(
                'alpha',
                rf'ALPHA\s*\(DEGREE\)\s*=\s*({re_float})', unit='degree'
            ),
            Quantity(
                'beta',
                rf'BETA\s*\(DEGREE\)\s*=\s*({re_float})', unit='degree'
            ),
            Quantity(
                'gamma',
                rf'GAMMA\s*\(DEGREE\)\s*=\s*({re_float})', unit='degree'
            ),
            Quantity(
                'bravais_name',
                r'BRAVAIS\s*=\s*([\w ]+)', str_operation=lambda x:x
            ),
            Quantity(
                'number_of_rotation_matrices',
                r'ROTATION MATRICES\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_point_group_operations',
                r'PURE POINT GROUP OPERATIONS\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_space_group_operations',
                r'SPACE GROUP OPERATIONS\s*=\s*(\d+)'
            ),
            Quantity(
                'point_group',
                r'POINT GROUP\s*=\s*([\w\_]+)', convert=False
            )
        ]

        orbital_quantities = [
            Quantity(
                'delta_k',
                rf'delta k\s*\(1/Bohr\)\s*=\s*({re_float})', unit='1/bohr', dtype=float
            ),
            Quantity(
                'delta_r',
                rf'delta r\s*\(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float
            ),
            Quantity(
                'dr_uniform',
                rf'dr_uniform\s*\(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float
            ),
            Quantity(
                'rmax',
                rf'rmax\s*\(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float
            ),
            Quantity(
                'kmesh',
                rf'kmesh\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'orbital_information',
                r'ORBITAL\s*L\s*N\s*nr\s*dr\s*RCUT\s*CHECK_UNIT\s*NEW_UNIT\n([\s\S]+)SET NONLOCAL PSEUDOPOTENTIAL PROJECTORS',
                convert=False, str_operation=str_to_orbital
            )
        ]

        header_quantities = [
            Quantity(
                'number_of_species',
                r'ntype\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'alat',
                rf'lattice constant \(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float
            ),
            Quantity(
                'atom_data',
                r'READING ATOM TYPE\s*\d+([\s\S]+?)\n\n', repeats=True,
                sub_parser=TextParser(quantities=atom_quantities), convert=False
            ),
            Quantity(
                'number_of_atoms',
                r'TOTAL ATOM NUMBER\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'positions',
                rf'(CARTESIAN COORDINATES \( UNIT = {re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag\s*vx\s*vy\s*vz\s*\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag\s*vx\s*vy\s*vz\s*\n[\s\S]+?)\n\n',
                sub_parser=TextParser(quantities=structure_quantities), convert=False, repeats=False,
            ),
            Quantity(
                'orbital_files',
                r'orbital file:\s*(\S+)', repeats=True
            ),
            Quantity(
                'cell_volume',
                rf'Volume \(Bohr\^3\)\s*=\s*({re_float})', unit='bohr**3', dtype=float
            ),
            Quantity(
                'units',
                r'Lattice vectors: \(Cartesian coordinate: in unit of (\w+)\)'
            ),
            Quantity(
                'lattice_vectors',
                rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, convert=False, repeats=False,
            ),
            Quantity(
                'reciprocal_units',
                r'Reciprocal vectors: \(Cartesian coordinate: in unit of ([\w\/ ]+)\)', flatten=False
            ),
            Quantity(
                'reciprocal_vectors',
                rf'Reciprocal vectors: \(Cartesian coordinate: in unit of 2 pi/a_0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, convert=False, repeats=False,
            ),
            Quantity(
                'pseudopotential',
                r'(Read in pseudopotential file is [\s\S]+?\n\n)', repeats=True,
                sub_parser=TextParser(quantities=[
                    Quantity('filename', r'file is\s*(\S+)'),
                    Quantity(
                        'type', r'pseudopotential type\s*=\s*(\w+)', flatten=False),
                    Quantity(
                        'xc', r'exchange-correlation functional\s*=\s*(\w+)', flatten=False),
                    Quantity(
                        'valence', rf'valence electrons\s*=\s*({re_float})'),
                    Quantity('lmax', rf'lmax\s*=\s*({re_float})'),
                    Quantity('number_of_zeta',
                             rf'number of zeta\s*=\s*({re_float})', dtype=int),
                    Quantity('number_of_projectors',
                             rf'number of projectors\s*=\s*({re_float})', dtype=int),
                    Quantity(
                        'l_of_projector', rf'L of projector\s*=\s*({re_float})', repeats=True, dtype=int),
                    Quantity('pao_radial_cut_off',
                             rf'PAO radial cut off \(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float),
                    Quantity('initial_pseudo_atomic_orbital_number',
                             rf'initial pseudo atomic orbital number\s*=\s*({re_float})'),
                    Quantity('nlocal', rf'NLOCAL\s*=\s*({re_float})'),
                ]
                )
            ),
            Quantity(
                'number_of_electrons',
                rf'total electron number of element (\w+)\s*=\s*(\d+)', repeats=True,
                str_operation=str_to_dict, convert=False
            ),
            Quantity(
                'occupied_bands',
                rf'occupied bands\s*=\s*({re_float})'
            ),
            Quantity(
                'nbands',
                rf'NBANDS\s*=\s*({re_float})', repeats=False
            ),
            Quantity(
                'symmetry',
                r'(LATTICE VECTORS: \(CARTESIAN COORDINATE: IN UNIT OF A0\)\s*[\s\S]+?)\n\n', repeats=False,
                sub_parser=TextParser(quantities=symmetry_quantities)
            ),
            Quantity(
                'nspin',
                r'nspin\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'sampling_method',
                r'Input type of k points\s*=\s*([\w\-\(\) ]+)', str_operation=lambda x:x
            ),
            Quantity(
                'nkstot',
                r'nkstot\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'nkstot_ibz',
                r'nkstot_ibz\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'k_points',
                r'minimum distributed K point number\s*=\s*\d+([\s\S]+?DONE : INIT K-POINTS Time)',
                str_operation=str_to_kpoints,
            ),
            Quantity(
                'wavefunction_cutoff',
                rf'energy cutoff for wavefunc \(unit:Ry\)\s*=\s*({re_float})', unit='rydberg', dtype=float
            ),
            Quantity(
                'wavefunction_fft_grid',
                r'\[fft grid for wave functions\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'
            ),
            Quantity(
                'charge_fft_grid',
                r'\[fft grid for charge/potential\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'
            ),
            Quantity(
                'fft_grid_division',
                r'\[fft grid division\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'
            ),
            Quantity(
                'nbxx',
                r'nbxx\s*=\s*(\d+)'
            ),
            Quantity(
                'nrxx',
                r'nrxx\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_pw_for_charge',
                r'SETUP PLANE WAVES FOR CHARGE/POTENTIAL\n\s*number of plane waves\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_sticks_for_charge',
                r'SETUP PLANE WAVES FOR CHARGE/POTENTIAL\n.*\n\s*number of sticks\s*=\s*(\d+)\n'
            ),
            Quantity(
                'number_of_pw_for_wavefunction',
                r'SETUP PLANE WAVES FOR WAVE FUNCTIONS\n\s*number of plane waves\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_sticks_for_wavefunction',
                r'SETUP PLANE WAVES FOR WAVE FUNCTIONS\n.*\n\s*number of sticks\s*=\s*(\d+)\n'
            ),
            Quantity(
                'parallel_pw_for_charge',
                r'PARALLEL PW FOR CHARGE/POTENTIAL\n\s*PROC\s*COLUMNS\(POT\)\s*PW\n([\s\S]+?)\-+',
                str_operation=str_to_sticks, convert=False
            ),
            Quantity(
                'parallel_pw_for_wavefunction',
                r'PARALLEL PW FOR WAVE FUNCTIONS\n\s*PROC\s*COLUMNS\(W\)\s*PW\n([\s\S]+?)\-+',
                str_operation=str_to_sticks, convert=False
            ),
            Quantity(
                'number_of_total_pw',
                r'number of total plane waves\s*=\s*(\d+)'
            ),
            Quantity(
                'number_of_g',
                r'number of \|g\|\s*=\s*(\d+)'
            ),
            Quantity(
                'max_g',
                rf'max \|g\|\s*=\s*({re_float})'
            ),
            Quantity(
                'min_g',
                rf'min \|g\|\s*=\s*({re_float})'
            ),
            Quantity(
                'total_number_of_nlocal_projectors',
                r'TOTAL NUMBER OF NONLOCAL PROJECTORS\s*=\s*(\d+)'
            ),
            Quantity(
                'init_chg',
                rf'init_chg\s*=\s*(\w+)'
            ),
            Quantity(
                'max_mesh_in_pp',
                rf'max mesh points in Pseudopotential\s*=\s*(\d+)'
            ),
            Quantity(
                'dq',
                rf'dq\(describe PAO in reciprocal space\)\s*=\s*({re_float})'
            ),
            Quantity(
                'max_q',
                rf'max q\s*=\s*({re_float})'
            ),
            Quantity(
                'number_of_pseudo_ao',
                r'number of pseudo atomic orbitals for (\w+) is (\d+)', repeats=True,
                str_operation=str_to_dict, convert=False
            ),
            Quantity(
                'orbital_settings',
                r'SETUP ONE DIMENSIONAL ORBITALS/POTENTIAL\s*([\s\S]+?)\-+',
                sub_parser=TextParser(quantities=orbital_quantities), convert=False
            ),
            Quantity(
                'allocation_method',
                r'divide the H&S matrix using [\w ]+ algorithms\.\n([\s\S]+?)\-+',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'nb2d', r'nb2d\s*=\s*(\d+)', dtype=int
                    ),
                    Quantity(
                        'row_dim', r'trace_loc_row dimension\s*=\s*(\d+)', dtype=int
                    ),
                    Quantity(
                        'col_dim', r'trace_loc_row dimension\s*=\s*(\d+)', dtype=int
                    ),
                    Quantity(
                        'nloc', r'nloc\s*=\s*(\d+)', dtype=int
                    ),
                ])
            ),
        ]

        search_quantities = [
            Quantity(
                'search_adjacent_atoms',
                r'SETUP SEARCHING RADIUS FOR PROGRAM TO SEARCH ADJACENT ATOMS\s*([\s\S]+?)\n\n',
                sub_parser=TextParser(quantities=[
                    Quantity('longest_orb_rcut',
                             rf'longest orb rcut \(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float),
                    Quantity('longest_nonlocal_projector_rcut',
                             rf'longest nonlocal projector rcut \(Bohr\)\s*=\s*({re_float})', unit='bohr', dtype=float),
                    Quantity('searching_radius',
                             rf'searching radius is \(Bohr\)\)\s*=\s*({re_float})', unit='bohr', dtype=float),
                    Quantity('searching_radius_unit',
                             rf'searching radius unit is \(Bohr\)\)\s*=\s*({re_float})', unit='bohr', dtype=float),
                ]
                )
            ),
            Quantity(
                'grid_integration',
                r'SETUP EXTENDED REAL SPACE GRID FOR GRID INTEGRATION\s*([\s\S]+?)\n\n',
                sub_parser=TextParser(quantities=[
                    Quantity('read_space_grid',
                             r'\[real space grid\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'),
                    Quantity('big_cell_numbers',
                             r'\[big cell numbers in grid\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'),
                    Quantity('meshcell_numbers',
                             r'\[meshcell numbers in big cell\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'),
                    Quantity('extended_fft_grid',
                             r'\[extended fft grid\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'),
                    Quantity('extended_fft_grid_dim',
                             r'\[dimension of extened grid\]\s*=\s*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*'),
                    Quantity('atom_number',
                             r'Atom number in sub-FFT-grid\s*=\s*(\d+)', dtype=int),
                    Quantity('local_orbitals_number',
                             r'Local orbitals number in sub-FFT-grid\s*=\s*(\d+)', dtype=int),
                    Quantity('nnr', r'ParaV\.nnr\s*=\s*(\d+)', dtype=int),
                    Quantity('nnrg', r'nnrg\s*=\s*(\d+)', dtype=int),
                    Quantity('nnrg_last', r'nnrg_last\s*=\s*(\d+)', dtype=int),
                    Quantity('nnrg_now', r'nnrg_now\s*=\s*(\d+)', dtype=int),
                    Quantity('lgd_last', r'nnrg_now\s*=\s*(\d+)', dtype=int),
                    Quantity('lgd_now', r'nnrg_now\s*=\s*(\d+)', dtype=int),
                ]
                )
            )
        ]

        calculation_quantities = [
            Quantity(
                'energy_occupation',
                r'(STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*\d+[\s\S]+?(?:\n\n\s*EFERMI|\n\n\n))', repeats=True,
                str_operation=str_to_energy_occupation, convert=False
            ),
            Quantity(
                'fermi_energy',
                rf'EFERMI\s*=\s*({re_float})\s*eV', unit='eV', dtype=float
            ),
            Quantity(
                'energies',
                rf'\s*final etot is\s*({re_float})\s*eV', unit='eV', dtype=float
            ),
            Quantity(
                'forces',
                r'TOTAL\-FORCE\s*\(eV/Angstrom\)\n\n.*\s*atom\s*x\s*y\s*z\n([\s\S]+?)\n\n',
                str_operation=str_to_force, convert=False
            ),
            Quantity(
                'stress',
                rf'(?:TOTAL\-|MD\s*)STRESS\s*\(KBAR\)\n\n.*\n\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, unit='kilobar', convert=False
            ),
            Quantity(
                'pressure',
                rf'TOTAL\-PRESSURE:\s*({re_float})\s*KBAR', unit='kilobar', dtype=float
            ),
            Quantity(
                'positions',
                rf'(CARTESIAN COORDINATES \( UNIT = {re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag\s*vx\s*vy\s*vz\s*\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag\s*vx\s*vy\s*vz\s*\n[\s\S]+?)\n\n',
                sub_parser=TextParser(quantities=structure_quantities), convert=False,
            ),
            Quantity(
                'cell_volume',
                rf'Volume \(Bohr\^3\)\s*=\s*({re_float})', unit='bohr**3', dtype=float
            ),
            Quantity(
                'units',
                r'Lattice vectors: \(Cartesian coordinate: in unit of (\w+)\)'
            ),
            Quantity(
                'lattice_vectors',
                rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, convert=False, repeats=False,
            ),
            Quantity(
                'reciprocal_units',
                r'Reciprocal vectors: \(Cartesian coordinate: in unit of ([\w\/ ]+)\)', flatten=False
            ),
            Quantity(
                'reciprocal_vectors',
                rf'Reciprocal vectors: \(Cartesian coordinate: in unit of 2 pi/a_0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n',
                str_operation=str_to_matrix, convert=False, repeats=False,
            ),
            Quantity(
                'nspin',
                r'nspin\s*=\s*(\d+)', dtype=int
            ),
            Quantity(
                'k_points',
                r'minimum distributed K point number\s*=\s*\d+([\s\S]+?DONE : INIT K-POINTS Time)',
                str_operation=str_to_kpoints,
            )
        ]+search_quantities

        scf_quantities = [
            Quantity(
                'iteration',
                r'(ELEC\s*=\s*\d+\s*\-+[\s\S]+?charge density convergence is achieved)', repeats=True,
                sup_parser=TextParser(quantities=[
                    Quantity(
                        'elec_step', r'ELEC\s*=\s*(\d+)', dtype=int
                    ),
                    Quantity(
                        'density_error', rf'Density error is\s*({re_float})', dtype=float
                    ),
                    Quantity(
                        'e_ks', rf'E_KohnSham\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_harris', rf'E_Harris\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_fermi', rf'E_Fermi\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_band', rf'E_band\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_one_elec', rf'E_one_elec\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_hartree', rf'E_Hartree\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_xc', rf'E_xc\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_ewald', rf'E_ewald\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_demet', rf'E_demet\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_descf', rf'E_descf\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_efield', rf'E_efield\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_exx', rf'E_exx\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                    Quantity(
                        'e_vdw', rf'E_vdwD\d+\s*{re_float}\s*({re_float})', dtype=float, unit='eV'
                    ),
                ]
                )
            )
        ]+calculation_quantities

        nscf_quantities = [
            Quantity(
                'band_structure',
                r'(band eigenvalue in this processor \(eV\)\s*:\n[\s\S]+?\n\n\n)',
                str_operation=str_to_bandstructure, convert=False
            ),
            Quantity(
                'min_state_energy',
                rf'min state energy (eV)\s*=\s*({re_float})', unit='eV', dtype=float, repeats=True
            ),
            Quantity(
                'max_state_energy',
                rf'max state energy (eV)\s*=\s*({re_float})', unit='eV', dtype=float, repeats=True
            ),
            Quantity(
                'delta_energy',
                rf'delta energy interval (eV)\s*=\s*({re_float})', unit='eV', dtype=float, repeats=True
            ),
            Quantity(
                'nbands',
                rf'number of bands\s*=\s*(\d+)', dtype=int, repeats=True
            ),
            Quantity(
                'sum_bands',
                rf'sum up the states\s*=\s*(\d+)', dtype=int, repeats=True
            ),
            Quantity(
                'fermi_energy',
                rf'Fermi energy.*is\s*({re_float})\s*Rydberg', unit='rydberg', dtype=float, repeats=True
            ),
            Quantity(
                'ionic_phase',
                rf'The Ionic Phase:\s*({re_float})', dtype=float
            ),
            Quantity(
                'electronic_phase',
                rf'Electronic Phase:\s*({re_float})', dtype=float
            ),
            Quantity(
                'polarization',
                r'(The calculated polarization direction is in \w+ direction[\s\S]+?C/m\^2)',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'direction', r'The calculated polarization direction is in (\w+) direction'),
                    Quantity(
                        'P',
                        rf'P\s*=\s*({re_float})\s*\(mod\s*({re_float})\)\s*\(\s*({re_float}),\s*({re_float}),\s*({re_float})\)\s*C/m\^2',
                        str_operation=str_to_polarization, convert=False)
                ])
            )
        ]+search_quantities

        relax_quantities = [
            Quantity(
                'ion_step', r'(?:STEP OF ION RELAXATION\s*:\s*|RELAX IONS\s*:\s*\d+\s*\(in total:\s*)(\d+)', dtype=int
            ),
            Quantity(
                'self_consistent',
                r'((STEP OF ION RELAXATION\s*:\s*\d+|RELAX IONS\s*:\s*\d+\s*\(in total: \d+\))[\s\S]+?(?:Setup the structure|\!FINAL_ETOT_IS))', sub_parser=TextParser(quantities=scf_quantities)
            )
        ]

        md_quantities = [
            Quantity(
                'md_step', r'STEP OF MOLECULAR DYNAMICS\s*:\s*(\d+)', dtype=int
            ),
            Quantity(
                'self_consistent',
                r'(STEP OF MOLECULAR DYNAMICS\s*:\s*\d+[\s\S]+?Energy\s*Potential\s*Kinetic)', sub_parser=TextParser(quantities=scf_quantities)
            ),
            Quantity(
                'energy',
                rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*({re_float})',
                dtype=float, unit='rydberg'
            ),
            Quantity(
                'potential',
                rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*{re_float}\s*({re_float})',
                dtype=float, unit='rydberg'
            ),
            Quantity(
                'kinetic',
                rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*{re_float}\s*{re_float}\s*({re_float})',
                dtype=float, unit='rydberg'
            ),
            Quantity(
                'temperature',
                rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*{re_float}\s*{re_float}\s*{re_float}\s*({re_float})',
                dtype=float, unit='kelvin'
            ),
            Quantity(
                'pressure',
                rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*{re_float}\s*{re_float}\s*{re_float}\s*{re_float}\s*({re_float})',
                dtype=float, unit='kilobar'
            ),
        ]

        run_quantities = [
            Quantity(
                'program_version',
                r'Version:\s*(.*)\n', str_operation=lambda x: ''.join(x)
            ),
            Quantity(
                'nproc',
                r'Processor Number is\s*(\d+)\n', str_operation=lambda x: ''.join(x), dtype=int
            ),
            Quantity(
                'start_date_time',
                r'Start Time is\s*(.*)\n', str_operation=lambda x: ''.join(x)
            ),
            Quantity(
                'global_out_dir', r'global_out_dir\s*=\s*(.*)\n',
            ),
            Quantity(
                'global_in_card', r'global_in_card\s*=\s*(.*)\n',
            ),
            Quantity(
                'pseudo_dir', r'pseudo_dir\s*=\s*([.\w\-\\ ]*?)',
            ),
            Quantity(
                'orbital_dir', r'orbital_dir\s*=\s*([.\w\-\\ ]*?)',
            ),
            Quantity(
                'drank', r'DRANK\s*=\s*(\d+)\n', dtype=int
            ),
            Quantity(
                'dsize', r'DSIZE\s*=\s*(\d+)\n', dtype=int
            ),
            Quantity(
                'dcolor', r'DCOLOR\s*=\s*(\d+)\n', dtype=int
            ),
            Quantity(
                'grank', r'GRANK\s*=\s*(\d+)\n', dtype=int
            ),
            Quantity(
                'gsize', r'GSIZE\s*=\s*(\d+)\n', dtype=int
            ),
            Quantity(
                'header',
                r'READING GENERAL INFORMATION([\s\S]+?)(?:[NON]*SELF-|STEP OF|RELAX CELL)', sub_parser=TextParser(quantities=header_quantities)
            ),
            Quantity(
                'self_consistent',
                r'SELF-CONSISTENT([\s\S]+?)Total\s*Time', sub_parser=TextParser(quantities=scf_quantities)
            ),
            Quantity(
                'nonself_consistent',
                r'NONSELF-CONSISTENT([\s\S]+?)Total\s*Time', sub_parser=TextParser(quantities=nscf_quantities)
            ),
            Quantity(
                'ion_threshold',
                rf'Ion relaxation is not converged yet \(threshold is\s*({re_float})\)', unit='eV/angstrom', repeats=False
            ),
            Quantity(
                'geometry_optimization',
                r'((?:STEP OF ION RELAXATION|RELAX IONS)\s*:\s*\d+[\s\S]+?(?:Setup the|\!FINAL_ETOT_IS))',
                sub_parser=TextParser(quantities=relax_quantities), repeats=True
            ),
            Quantity(
                'molecular_dynamics',
                r'(STEP OF MOLECULAR DYNAMICS\s*:\s*\d+[\s\S]+?(?:\n{4,5}))',
                sub_parser=TextParser(quantities=md_quantities), repeats=True
            ),
            Quantity(
                'final_energy',
                rf'\!FINAL_ETOT_IS\s*({re_float})', dtype=float, unit='eV'
            ),
            Quantity(
                'finish_date_time',
                r'Finish\s*Time\s*:\s*(.*)\n', str_operation=lambda x: ''.join(x)
            ),
            Quantity(
                'total_time',
                rf'Total\s*Time\s*:\s*(\d+)\s*h\s*(\d+)\s*mins\s*(\d+)\s*secs',
                str_operation=lambda x: int(x.strip().split()[0])*3600+int(x.strip().split()[1])*60+int(x.strip().split()[2]), unit='seconds'
            )
        ]

        self._quantities = [
            Quantity(
                'run',
                r'(WELCOME\s*TO\s*ABACUS[\S\s]+?Total\s*Time\s*:\s*\d+\s*h\s*\d+\s*mins\s*\d+\s*secs)',
                sub_parser=run_quantities, repeats=True)
        ]


class ABACUSParser(FairdiParser):
    def __init__(self):
        super().__init__(name='parsers/abacus', code_name='ABACUS',
                         code_homepage='http://abacus.ustc.edu.cn/',
                         mainfile_contents_re=r'(\n\s*WELCOME TO ABACUS)')
        self.out_parser = ABACUSOutParser()
# TODO: need to convert direct positions to cartesian ones, then thw cartesian ones multiply units bohr
# symmetry, lattice_vector, rep_vector, position, converted
