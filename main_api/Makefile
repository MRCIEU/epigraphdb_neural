.PHONY: clean data lint requirements docs

# directories that contain python scripts in the codebase
python_dirs = app epigraphdb_common_utils scripts tests

#################################################################################
# Rules
#################################################################################

## ==== entity embeddings ====
write_ent_embeddings:
	python -m scripts.write_ent_embeddings


## ==== running the api ====

## sanity check
check:
	python -m scripts.check

## Start API server, port: 80
run:
	python -m scripts.check
	uvicorn app.main:app --reload \
	--host 0.0.0.0 --port 80

## ==== codebase ====

## Format codebase
fmt:
	python -m autoflake \
		--in-place --remove-all-unused-imports --recursive \
		$(python_dirs)
	python -m isort -rc \
		$(python_dirs)
	python -m black \
		$(python_dirs)

## Lint codebase
lint:
	python -m flake8 \
		$(python_dirs)
	python -m mypy \
		$(python_dirs)

## Unit tests
test:
	python -m pytest -vv

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'
