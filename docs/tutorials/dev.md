# ElasticDL: Developer's Guide

## Get the Source Code

You can get the latest source code from Github:

```bash
git clone https://github.com/sql-machine-learning/elasticdl
cd elasticdl
```


## Development Tools in a Docker Image

We prefer to install all building tools in a Docker image.

```bash
docker build --target dev -t elasticdl:dev -f elasticdl/docker/Dockerfile .
```


## Check Code Style

The above Docker image contains pre-commit and hooks.  We can run it as a
container and bind mount the local Git repo into the container.  In this
container, we run the pre-commit command.

```bash
docker run --rm -it -v $PWD:/work -w /work \
  elasticdl:dev /work/scripts/pre-commit.sh
```

If you have all required tools installed, you can run the same script on your
host.

```bash
./scripts/pre-commit.sh
```
