# syntax=docker/dockerfile:1.7

# This derived image keeps the user's vendor daily A5 image and replaces vLLM.
# vLLM-Ascend is installed after container creation from the host-mounted
# checkout, so the image build never clones a source repository.
ARG BASE_IMAGE=vllm-ascend:dev-26.1.0.day20260817-A5-py311-openEuler24.03-lts-aarch64
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG VLLM_VERSION=0.27.1
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# proxy.sh remains a BuildKit secret and is not copied into an image layer.
# The same RUN instruction performs every operation that needs its exported
# proxy variables.
RUN --mount=type=secret,id=hajimi_proxy,required=false \
    if [[ -s /run/secrets/hajimi_proxy ]]; then source /run/secrets/hajimi_proxy; fi && \
    python3 -m pip install --force-reinstall "vllm==${VLLM_VERSION}" \
        --no-deps \
        --index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PYTORCH_INDEX_URL}" \
        --trusted-host mirrors.aliyun.com \
        --trusted-host download.pytorch.org \
        --trusted-host download-r2.pytorch.org && \
    python3 -c 'import importlib.metadata as m; print("vllm=" + m.version("vllm"))'

# create-container.sh runs this only after /home has been mounted into the
# container, making the host-managed checkout available at its stable path.
COPY --chmod=0755 scripts/install-vllm-ascend.sh /usr/local/bin/install-vllm-ascend

# `/home` is mounted at container start. Interactive root shells source the
# host-managed proxy and land in the project without baking proxy details into
# the image.
RUN printf '%s\n' \
    'if [[ -f /home/hajimi/proxy.sh ]]; then source /home/hajimi/proxy.sh; fi' \
    'if [[ -d /home/hajimi/qwen3.8/scripts ]]; then cd /home/hajimi/qwen3.8/scripts; else cd /home/hajimi; fi' \
    >> /root/.bashrc

ENV VLLM_PORT=6969
WORKDIR /home/hajimi
CMD ["/bin/bash"]
