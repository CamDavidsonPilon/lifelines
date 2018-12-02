init:
	pip install pipenv
	pipenv install --dev --skip-lock

test:
	pipenv run py.test -rf -s --cov=lifelines -vv --block=False --cov-report term-missing

lint:
ifeq ($(TRAVIS_PYTHON_VERSION), 2.7)
		echo "Skip linting for Python2.7"
else
		prospector --output-format grouped
endif	

format:
	black .

check_format:
ifeq ($(TRAVIS_PYTHON_VERSION), 3.6)
		black . --check
else
		echo "Only check format on Python3.6"
endif	

update_reqs:
	pipenv lock -r > requirements.txt
	pipenv lock -r --dev > dev-requirements.txt