# -*- coding: utf-8 -*-
import distutils.text_file
from pathlib import Path

import pkg_resources


BASE_PATH = Path(__file__).parent.parent
REQUIREMENTS_PATH = BASE_PATH.joinpath("reqs/base-requirements.txt")


class TestRequirements:
    def test_requirements(self):
        """Test that each requirement is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        print(dir(Path(__file__).parent.parent))

        requirements = distutils.text_file.TextFile(filename=REQUIREMENTS_PATH).readlines()
        for requirement in requirements:
            pkg_resources.require(requirement)
