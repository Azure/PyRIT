.PHONY: all pre-commit mypy test test-cov-html test-cov-xml

CMD:=python -m
PYMODULE:=pyrit
TESTS:=tests

all: pre-commit

pre-commit:
	$(CMD) isort --multi-line 3 --recursive $(PYMODULE) $(TESTS)
	pre-commit run --all-files

mypy:
	$(CMD) mypy $(PYMODULE) $(TESTS)

docs-build:
	jb build -W -v ./doc

test:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS)

test-cov-html:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html

test-cov-xml:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report xml --junitxml=junit/test-results.xml --doctest-modules

#clean:
#	git clean -Xdf # Delete all files in .gitignore
