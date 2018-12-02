init:
	pip install pipenv
	pipenv install --dev --skip-lock

test:
	pipenv run py.test -rf -s --cov=lifelines -vv --block=False --cov-report term-missing

lint:
	prospector --output-format grouped

format:
	black .
