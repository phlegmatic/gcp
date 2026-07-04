"""Integration test: compile both pipelines end-to-end (no cloud submission).

This validates that the KFP DAG wiring (artifact passing, parallel branches,
fan-in) is correct without incurring any GCP cost. Marked `integration` because
it imports the full `kfp` compiler and pipeline graph.
"""

import pytest
from kfp import compiler

from forecasting.pipelines import PIPELINES

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("pipeline_key", list(PIPELINES))
def test_pipeline_compiles_to_yaml(pipeline_key, tmp_path):
    out = tmp_path / f"{pipeline_key}.yaml"
    compiler.Compiler().compile(
        pipeline_func=PIPELINES[pipeline_key],
        package_path=str(out),
    )
    assert out.exists()
    content = out.read_text()
    # Sanity: compiled spec should reference the KFP pipeline schema.
    assert (
        "pipelineSpec" in content
        or "pipeline_spec" in content
        or "components" in content
    )
