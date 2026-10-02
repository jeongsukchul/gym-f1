import subprocess
import sys
from pathlib import Path


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )


def test_importing_envs_params_does_not_eagerly_import_gkenv():
    result = _run_python(
        "from gymkhana.envs.params import load_params; "
        "import sys; "
        "print('gymkhana.presets' in sys.modules); "
        "print('gymkhana.envs.gymkhana_env' in sys.modules); "
        "print(load_params('f1tenth_std')['m'])"
    )
    stdout = result.stdout.strip().splitlines()
    assert stdout[0] == "False"
    assert stdout[1] == "False"
    assert float(stdout[2]) > 0.0


def test_importing_envs_track_does_not_eagerly_import_gkenv():
    result = _run_python(
        "from gymkhana.envs.track import Track; "
        "import sys; "
        "print('gymkhana.presets' in sys.modules); "
        "print('gymkhana.envs.gymkhana_env' in sys.modules); "
        "print(Track.__name__)"
    )
    stdout = result.stdout.strip().splitlines()
    assert stdout == ["False", "False", "Track"]


def test_importing_jax_sampler_ppo_does_not_eagerly_import_optional_modules():
    result = _run_python(
        "import gymkhana.jax_sampler_ppo as jsp; "
        "import sys; "
        "print('gymkhana.jax_sampler_ppo.export_onnx' in sys.modules); "
        "print('gymkhana.jax_sampler_ppo.gmmvi.network' in sys.modules); "
        "print(hasattr(jsp, 'SamplerPPOTrainer')); "
        "print('gymkhana.jax_sampler_ppo.export_onnx' in sys.modules); "
        "print('gymkhana.jax_sampler_ppo.gmmvi.network' in sys.modules)"
    )
    stdout = result.stdout.strip().splitlines()
    assert stdout == ["False", "False", "True", "False", "False"]
