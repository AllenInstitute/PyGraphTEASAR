#!/bin/bash
bumpversion $1
git push && git push --tags
TAG=$(git describe --tags)
poetry build
twine upload dist/graphteasar-${TAG:1}.tar.gz
twine upload dist/graphteasar-${TAG:1}-py3-non-any.whl
