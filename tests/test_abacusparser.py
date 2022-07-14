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
import numpy as np

from nomad.datamodel import EntryArchive
from nomad.units import ureg
from abacusparser.abacus_parser import ABACUSParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return ABACUSParser()


def test_band(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Si_band/running_nscf.log', archive, None)

    sec_run = archive.section_run[0]
    assert sec_run.program_version == 'Parallel, in development'
    assert sec_run.program_basis_set_type == 'Numeric AOs'
    assert sec_run.time_run_date_start.magnitude == 