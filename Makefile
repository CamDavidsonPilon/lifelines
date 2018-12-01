init:
	pip install pipenv
	pipenv install --dev

test:
	pipenv run py.test -s --cov=lifelines -vv --block=False --cov-report term-missing

autopep8:
	autopep8 --ignore E501,E241,W690 --in-place --recursive --aggressive lifelines/

lint:
	flake8 lifelines --statistics --config=setup.cfg

autolint: autopep8 lint


