autopep8:
	autopep8 --ignore E501,E241,W690 --in-place --recursive --aggressive lifelines/

lint:
	flake8 lifelines

autolint: autopep8 lint

