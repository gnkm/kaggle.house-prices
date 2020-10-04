#! /usr/bin/env bash

# print the usage and exit
print_usage_and_exit () {
	cat <<____USAGE 1>&2
Usage   : ${0##*/} <var1> <var2> ...
____USAGE
	exit 1
}

# main script starts here

if [ $1 = 'run' ]; then
    docker run \
        -v $PWD:/tmp/working \
        -w=/tmp/working \
        --rm \
        -it \
        --name kaggle-python \
        gcr.io/kaggle-images/python@sha256:af3e19d6210754b09d3ae97d30210712ff183877ab6d0fa71485a0dd7c1f33b3 \
        python $2
elif [ $1 = 'jp' ]; then
    docker run \
        -v $PWD:/tmp/working \
        -w=/tmp/working \
        -p 8080:8080 \
        --rm \
        -it \
        --name kaggle-jupyter \
        gcr.io/kaggle-images/python@sha256:af3e19d6210754b09d3ae97d30210712ff183877ab6d0fa71485a0dd7c1f33b3 \
        jupyter notebook \
            --no-browser \
            --ip=0.0.0.0 \
            --allow-root \
            --notebook-dir=/tmp/working
elif [ $1 = 'flake' ]; then
    docker run \
        -v $PWD:/tmp/working \
        -w=/tmp/working \
        --rm \
        -it \
        --name kaggle-python \
        gcr.io/kaggle-images/python@sha256:af3e19d6210754b09d3ae97d30210712ff183877ab6d0fa71485a0dd7c1f33b3 \
        flake8 $2
elif [ $1 = 'script' ]; then
    docker run \
        -v $PWD:/tmp/working \
        -w=/tmp/working \
        --rm \
        -it \
        --name kaggle-python \
        gcr.io/kaggle-images/python@sha256:af3e19d6210754b09d3ae97d30210712ff183877ab6d0fa71485a0dd7c1f33b3 \
        python $2 $3
else
    docker run \
        -v $PWD:/tmp/working \
        -w=/tmp/working \
        --rm \
        -it \
        --name kaggle-python \
        gcr.io/kaggle-images/python@sha256:af3e19d6210754b09d3ae97d30210712ff183877ab6d0fa71485a0dd7c1f33b3 \
        $1
fi
