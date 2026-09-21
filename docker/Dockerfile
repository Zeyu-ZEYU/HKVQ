FROM vllm/vllm-openai:v0.29.0

# The image has no `python` executable, and it ships LMCache, which HACK does not use and which
# requires an older NumPy than the one pinned in pyproject.toml.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python && pip uninstall -y lmcache

WORKDIR /workspace/hack
COPY . .
RUN pip install --no-cache-dir -e ".[benchmarks,test]"

ENTRYPOINT ["/bin/bash"]
