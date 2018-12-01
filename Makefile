init:
	pip install pipenv
	pipenv install --dev --skip-lock

test:
	pipenv run py.test -rfs -s --cov=lifelines -vv --block=False --cov-report term-missing

lint:
	prospector --zero-exit --output-format grouped

format:
	black .
