#!/bin/bash

for filename in ./bundled_pip_dependencies/*.tar.gz; do
    [ -e "$filename" ] || continue
    pip install -U "$filename"
done
