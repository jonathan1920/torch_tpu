
# GKE TPUv7 example

### Set environment variables for later

All of the steps in this tutorial rely on the following variables, set them
accordingly.

```bash
export CLUSTER=
export ZONE_NAME=
export PROJECT=
export ARTIFACT_REGISTRY_URL=us-pkg.docker.dev/...
export DOCKER_IMAGE=test-docker-image:latest
```

### Setup XPK

XPK makes the process of setting up your TPUv7 cluster quite simple, setting up
coordination and job queue management out of the box.

See https://github.com/AI-Hypercomputer/xpk/tree/main

`pip install xpk`

### Create XPK enabled cluster

This step is optional *if* you already have an active cluster. XPK initialized
clusters manage queuing for jobs automatically, so there's no need to have
unique clusters per user.

```bash
xpk cluster create \
  --cluster=${CLUSTER} \
  --tpu-type=tpu7x-8 \
  --num-slices=1 \
  --zone=${ZONE_NAME} \
  --project=${PROJECT} \
  --spot # this sets up your cluster as a spot instance. Not best for long-running jobs.
```

### GCloud and docker setup

You need to login to GCloud CLI for XPK, and we also need to set up our docker
credentials so we can push to artifact registry for GKE to consume.

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker
sudo usermod -aG docker $USER # relaunch the terminal and activate venv
docker run hello-world # Test Docker
```

Follow
[this tutorial](https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)
to initialize your artifact registry to store this new container image for use
in GKE (skip if one already exists).

### Build TorchTPU docker with Tensor Parallel Example

```bash
cd $TORCH_TPU_SRC/docker
./build_image.sh [specify a different source location to build from]
docker tag torch-tpu-local:latest ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
docker push ${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
```

### Enqueuing a workload to the new cluster

```bash
xpk workload create \
  --workload=unique-worker-name-1 \
  --cluster=${CLUSTER} \
  --zone=${ZONE_NAME} \
  --project=${PROJECT} \
  --tpu-type=tpu7x-8 \
  --docker-image=${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE} \
  --command="cd /workspace/examples/distributed/tensor_parallel/ && ./gcp_launch.sh"
```

If the example contains a requirements file, you will need to install them:

```bash
xpk workload create
  --workload=unique-worker-name-1
  --cluster=${CLUSTER}
  --zone=${ZONE_NAME}
  --project=${PROJECT}
  --tpu-type=tpu7x-8
  --docker-image=${ARTIFACT_REGISTRY_URL}/${DOCKER_IMAGE}
  --command="cd /workspace/examples/distributed/fsdp/lora/ && pip install -r requirements.txt && ./gcp_launch.sh"
```

