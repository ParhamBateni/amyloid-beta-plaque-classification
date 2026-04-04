#! /bin/bash

ruff check --select F,I --fix src/
ruff format src/