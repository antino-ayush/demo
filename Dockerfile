# Runs the batch evaluation CLI (evalsuite/cli.py). Kept minimal on purpose:
# only the hard dependency (PyYAML) is installed by default, so the image
# builds fast and the mock judge works out of the box with no API keys.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optional extras -- DeepEval-backed scorers (--use-deepeval) and the real
# judge providers (--judge claude / --judge openai) need packages this
# image doesn't install by default. Pull them in at build time if needed:
#   docker build --build-arg EXTRAS="deepeval anthropic openai" -t evalsuite .
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir $EXTRAS; fi

COPY . .

ENV PYTHONUNBUFFERED=1

# Default: score the sample dataset with the offline mock judge, gate on
# failures. Override at `docker run` time -- see README.md's Docker section
# for mounting your own dataset/config and passing real judge credentials.
ENTRYPOINT ["python", "-m", "evalsuite.cli"]
CMD ["run", "--dataset", "data/sample_dataset.jsonl", "--config-dir", "configs/modules", "--judge", "mock", "--db", "eval_results.db", "--gate"]
