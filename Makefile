init:
ifeq ($(TRAVIS), true)
		pip install -r reqs/travis-requirements.txt
		pip install pandas==${PANDAS_VERSION}
		pip list --local
else
		pip install -r reqs/dev-requirements.txt
		pre-commit install
endif

test:
	py.test tests/ -rfs --cov=lifelines --block=False --cov-report term-missing

lint:
ifeq ($(TRAVIS_PYTHON_VERSION), 2.7)
		echo "Skip linting for Python2.7"
else
		make black
		prospector --output-format grouped
endif

black:
ifeq ($(TRAVIS_PYTHON_VERSION), 2.7)
		echo "Skip linting for Python2.7"
else
		black lifelines/ -l 120 --fast
		black tests/ -l 120 --fast
endif

check_format:
ifeq ($(TRAVIS_PYTHON_VERSION), 3.6)
		black . --check --line-length 120
else
		echo "Only check format on Python3.6"
endif

pre:
	pre-commit run --all-files
