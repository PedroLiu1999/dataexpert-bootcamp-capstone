import subprocess
import sys


def test_app_dependency_graph_does_not_import_pyspark():
    code = (
        "import sys; "
        "import src.agent.tools, src.embedding, src.db.repository, src.analytics.delta_cdf; "
        "assert not any(m == 'pyspark' or m.startswith('pyspark.') for m in sys.modules), "
        "sorted(m for m in sys.modules if m.startswith('pyspark'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
