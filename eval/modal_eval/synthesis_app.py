"""
Modal app for specification synthesis and validation.

This replaces AWS Lambda for parallel synthesis evaluation.
It runs the games/synt synthesis pipeline on Modal's cloud infrastructure.

Usage:
    # Test locally first
    modal run modal_eval/synthesis_app.py::test_synthesis

    # Deploy (optional)
    modal deploy modal_eval/synthesis_app.py

References:
    - https://modal.com/docs/guide/mounting
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import modal

# Modal app definition
app = modal.App("tslf-synthesis")

# Paths to local resources
PROJECT_ROOT = Path(__file__).parent.parent.parent
VALIDATOR_PATH = PROJECT_ROOT / "games" / "synt"

# Issy binary path
ISSY_OPTIMIZED_PATH = Path(__file__).parent / "issy-linux-optimized"
ISSY_STATIC_PATH = None  # Static binary no longer used

# Check which binaries are available
HAS_OPTIMIZED = ISSY_OPTIMIZED_PATH.exists()
HAS_STATIC = False  # Static binary no longer used

if HAS_OPTIMIZED:
    print(f"Found optimized issy binary: {ISSY_OPTIMIZED_PATH}")
else:
    print(f"No optimized binary found. Run build_issy_linux.sh to build it.")

if HAS_STATIC:
    print(f"Found static issy binary: {ISSY_STATIC_PATH}")

# Build the image with all dependencies
# Includes BOTH static and optimized issy binaries for benchmarking
synthesis_image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install system dependencies
    .apt_install(
        "gcc", "g++", "make", "wget", "tar", "gzip", "unzip", "curl", "ca-certificates"
    )
    # Install Spot LTL library (required by issy)
    .run_commands(
        "cd /tmp && "
        "wget -q https://www.lrde.epita.fr/dload/spot/spot-2.11.6.tar.gz && "
        "tar xzf spot-2.11.6.tar.gz && "
        "cd spot-2.11.6 && "
        "./configure --disable-python --prefix=/opt && "
        "make -j$(nproc) && "
        "make install && "
        "cd / && rm -rf /tmp/spot*"
    )
    # Install Z3 SMT solver
    .run_commands(
        "cd /tmp && "
        "wget -q https://github.com/Z3Prover/z3/releases/download/z3-4.13.0/z3-4.13.0-x64-glibc-2.31.zip && "
        "unzip -q z3-4.13.0-x64-glibc-2.31.zip && "
        "mkdir -p /opt/bin && "
        "cp z3-4.13.0-x64-glibc-2.31/bin/z3 /opt/bin/z3 && "
        "chmod +x /opt/bin/z3 && "
        "rm -rf /tmp/z3*"
    )
    # Set environment variables
    .env({
        "PATH": "/opt/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/lib",
        "TMPDIR": "/tmp"
    })
    # Install Python dependencies
    .pip_install("pyyaml")
    # Add BOTH issy binaries for comparison benchmarking
    .add_local_file(
        str(ISSY_STATIC_PATH) if HAS_STATIC else str(ISSY_OPTIMIZED_PATH),
        remote_path="/opt/bin/issy-static",
        copy=True
    )
    .add_local_file(
        str(ISSY_OPTIMIZED_PATH) if HAS_OPTIMIZED else str(ISSY_STATIC_PATH),
        remote_path="/opt/bin/issy-optimized",
        copy=True
    )
    # Create symlink to default issy (optimized if available)
    .run_commands(
        "chmod +x /opt/bin/issy-static /opt/bin/issy-optimized && "
        "ln -sf /opt/bin/issy-optimized /opt/bin/issy"
    )
    # Add synt (synthesis) code
    .add_local_dir(
        str(VALIDATOR_PATH),
        remote_path="/app/synt"
    )
)


@app.function(
    image=synthesis_image,
    timeout=3600,  # 60 minutes (Modal Pro allows up to 24h)
    memory=4096,   # 4GB
    cpu=4,         # 4 cores for better parallelism
)
def synthesize_and_validate(config_yaml: str, game_name: str = "ice_lake", task_id: str = "task") -> dict:
    """
    Run the specification synthesis and validation pipeline.

    Args:
        config_yaml: YAML configuration string with synthesis params and game config
        game_name: Game type (ice_lake, taxi, cliff_walking)
        task_id: Unique identifier for this task

    Returns:
        dict with success, passed, total, results, synthesis_time, error_message
    """
    start_time = time.time()

    # Add specification_validator to path
    sys.path.insert(0, "/app/synt")

    try:
        from pipeline import Pipeline, load_config

        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            # Load config with root_dir pointing to /app/synt
            config = load_config(config_path, root_dir=Path("/app/synt"))
            config.name = game_name
            # Override root_dir to /tmp for writable logs/output
            config.root_dir = Path("/tmp")

            # Run pipeline
            pipeline = Pipeline(config)
            result = pipeline.run(debug_override=False)

            elapsed = time.time() - start_time

            # Extract results
            results = []
            total_passed = 0
            total_configs = 0

            for cfg_result in result.configurations:
                for obj in cfg_result.objectives:
                    # Determine failure type
                    error_msg = obj.error_message or ""
                    if obj.success:
                        failure_type = "success"
                    elif "No 'void main()'" in error_msg or "unrealizable" in error_msg.lower():
                        failure_type = "unrealizable"
                    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        failure_type = "synthesis_timeout"
                    elif obj.steps is not None and obj.steps > 0:
                        failure_type = "execution_fail"  # Controller ran but didn't reach goal
                    else:
                        failure_type = "synthesis_error"

                    result_entry = {
                        "config_name": cfg_result.config_name,
                        "objective": obj.objective[:50] if obj.objective else "",
                        "passed": obj.success,
                        "steps": obj.steps,
                        "error": obj.error_message,
                        "failure_type": failure_type,
                        "synthesis_time": obj.synthesis_time,
                        "game_output": obj.game_output,  # Full trajectory log
                    }
                    results.append(result_entry)
                total_passed += cfg_result.passed
                total_configs += cfg_result.total

            # Check if MuVal was called (read the log file)
            muval_log = None
            muval_call_count = 0
            try:
                with open("/tmp/muval_calls.log", "r") as f:
                    muval_log = f.read()
                    muval_call_count = muval_log.count("MUVAL INVOKED")
            except FileNotFoundError:
                muval_log = "No MuVal log file found - MuVal was NOT called"

            return {
                "task_id": task_id,
                "success": total_passed > 0,
                "passed": total_passed,
                "total": total_configs,
                "results": results,
                "synthesis_time": elapsed,
                "error_message": result.error_message,
                "muval_call_count": muval_call_count,
                "muval_log": muval_log[-2000:] if muval_log else None  # Last 2000 chars
            }

        finally:
            # Cleanup
            try:
                os.unlink(config_path)
            except:
                pass

    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "passed": 0,
            "total": 1,
            "results": [],
            "synthesis_time": time.time() - start_time,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }


@app.function(
    image=synthesis_image,
    timeout=300,
)
def debug_muval() -> dict:
    """Debug function to verify MuVal is working correctly."""
    import subprocess
    import os

    results = {}

    # Check if MuVal binary exists
    muval_path = "/muval/_build/default/main.exe"
    results["muval_exists"] = os.path.exists(muval_path)

    # Check if config exists
    config_path = "/muval/config/solver/dbg_muval_parallel_exc_tb_ar.json"
    results["config_exists"] = os.path.exists(config_path)

    # Check OPAM
    try:
        opam_result = subprocess.run(
            ["bash", "-c", "export OPAMROOT=/root/.opam && eval $(opam env --root=/root/.opam --switch=5.2.0) && which ocaml"],
            capture_output=True, text=True, timeout=30
        )
        results["opam_ocaml"] = opam_result.stdout.strip()
        results["opam_error"] = opam_result.stderr.strip() if opam_result.returncode != 0 else None
    except Exception as e:
        results["opam_error"] = str(e)

    # Check call-muval script
    results["call_muval_exists"] = os.path.exists("/opt/bin/call-muval")

    # Try running MuVal with a trivial input
    try:
        # Simple HES that should be satisfiable
        test_hes = "(set-logic HORN)\n(declare-fun P () Bool)\n(assert P)\n(check-sat)\n"

        muval_test = subprocess.run(
            ["bash", "-c", f"""
                export OPAMROOT=/root/.opam
                eval $(opam env --root=/root/.opam --switch=5.2.0)
                cd /muval
                echo 'Testing MuVal binary directly...'
                ./_build/default/main.exe --help 2>&1 | head -5
            """],
            capture_output=True, text=True, timeout=60
        )
        results["muval_help_output"] = muval_test.stdout[:500]
        results["muval_help_stderr"] = muval_test.stderr[:500] if muval_test.stderr else None
        results["muval_help_returncode"] = muval_test.returncode
    except Exception as e:
        results["muval_test_error"] = str(e)

    # Check issy can see MuVal
    try:
        issy_test = subprocess.run(
            ["/opt/bin/issy", "--help"],
            capture_output=True, text=True, timeout=30
        )
        # Look for muval-related options in help
        results["issy_help_has_pruning"] = "--pruning" in issy_test.stdout
        results["issy_help_snippet"] = issy_test.stdout[:300]
    except Exception as e:
        results["issy_test_error"] = str(e)

    return results


def make_config(
    name: str,
    grid_size: int,
    goal_x: int,
    goal_y: int,
    holes: list,
    objective: str,
    issy_binary: str = "issy",  # issy, issy-static, or issy-optimized
    accel: str = "full",  # full, no
    timeout_minutes: int = 10
) -> str:
    """Generate a config YAML string with configurable issy binary and acceleration."""
    holes_yaml = "\n".join([f"      - {{x: {h['x']}, y: {h['y']}}}" for h in holes]) if holes else "[]"
    if holes:
        holes_section = f"""holes:
{holes_yaml}"""
    else:
        holes_section = "holes: []"

    # Build args based on acceleration setting
    if accel == "no":
        accel_args = """    - "--accel"
    - "no" """
    else:
        accel_args = """    - "--accel"
    - "full"
    - "--accel-attr"
    - "polycomp-ext"
    - "--accel-difficulty"
    - "hard"
    - "--enable-summaries" """

    return f"""
name: ice_lake

synthesis:
  command: {issy_binary}
  args:
    - "--tslmt"
    - "--synt"
    - "--pruning"
    - "1"
{accel_args}
  timeout_minutes: {timeout_minutes}

debug: false

run_configuration:
  - name: "{name}"
    grid_size: {grid_size}
    start_pos: {{x: 0, y: 0}}
    goal: {{x: {goal_x}, y: {goal_y}}}
    {holes_section}
    objectives:
      - objective: "{objective}"
        timeout: 200
"""


@app.local_entrypoint()
def test_synthesis():
    """Benchmark issy configurations: static vs optimized, accel vs no-accel."""

    # Define test problem - simple enough to complete quickly
    test_problem = {
        "name": "3x3_liveness",
        "grid_size": 3,
        "goal_x": 2, "goal_y": 2,
        "holes": [],
        "objective": "F ((eq x goalx) && (eq y goaly))"
    }

    # 4 configurations: 2 binaries × 2 accel settings
    issy_versions = ["issy-static", "issy-optimized"]
    accel_settings = ["full", "no"]

    configs = []
    for issy_bin in issy_versions:
        for accel in accel_settings:
            task_id = f"{issy_bin}_{accel}"
            config_yaml = make_config(
                name=task_id,
                grid_size=test_problem["grid_size"],
                goal_x=test_problem["goal_x"],
                goal_y=test_problem["goal_y"],
                holes=test_problem["holes"],
                objective=test_problem["objective"],
                issy_binary=issy_bin,
                accel=accel,
                timeout_minutes=10
            )
            configs.append({
                "task_id": task_id,
                "config_yaml": config_yaml,
                "issy_binary": issy_bin,
                "accel": accel
            })

    print("=" * 75)
    print("  Issy Configuration Benchmark")
    print("  Problem: 3x3 grid, reach goal (liveness only)")
    print("  Configurations: 2 binaries × 2 accel settings = 4 total")
    print("=" * 75)
    print()

    # Run all 4 in parallel using starmap
    config_yamls = [c["config_yaml"] for c in configs]
    game_names = ["ice_lake"] * len(configs)
    task_ids = [c["task_id"] for c in configs]

    print("Starting parallel synthesis...")
    start_time = time.time()

    results = list(synthesize_and_validate.starmap(
        zip(config_yamls, game_names, task_ids)
    ))

    total_time = time.time() - start_time

    # Analyze results
    print()
    print("=" * 75)
    print("  RESULTS")
    print("=" * 75)
    print()
    print(f"{'Binary':<20} {'Accel':<10} {'Status':<15} {'Time (s)':<12} {'Steps':<8}")
    print("-" * 75)

    times_by_config = {}

    for config, result in zip(configs, results):
        synth_time = result.get("synthesis_time", 0)
        success = result.get("success", False)
        status = "PASS" if success else "FAIL"
        steps = None

        # Get detailed failure info
        if not success and result.get("results"):
            failure_type = result["results"][0].get("failure_type", "unknown")
            status = f"FAIL ({failure_type})"
        elif success and result.get("results"):
            steps = result["results"][0].get("steps")

        steps_str = str(steps) if steps else "-"
        print(f"{config['issy_binary']:<20} {config['accel']:<10} {status:<15} {synth_time:<12.2f} {steps_str:<8}")

        key = (config['issy_binary'], config['accel'])
        times_by_config[key] = synth_time

    print("-" * 75)
    print()

    # Summary comparison
    print("Summary:")
    static_full = times_by_config.get(("issy-static", "full"), 0)
    static_no = times_by_config.get(("issy-static", "no"), 0)
    opt_full = times_by_config.get(("issy-optimized", "full"), 0)
    opt_no = times_by_config.get(("issy-optimized", "no"), 0)

    print(f"  Static  + accel full: {static_full:.2f}s")
    print(f"  Static  + accel no:   {static_no:.2f}s")
    print(f"  Optimized + accel full: {opt_full:.2f}s")
    print(f"  Optimized + accel no:   {opt_no:.2f}s")
    print()

    # Find the fastest
    all_times = [
        (static_full, "static + accel"),
        (static_no, "static + no-accel"),
        (opt_full, "optimized + accel"),
        (opt_no, "optimized + no-accel"),
    ]
    fastest = min(all_times, key=lambda x: x[0] if x[0] > 0 else float('inf'))
    print(f"  → Fastest: {fastest[1]} ({fastest[0]:.2f}s)")

    # Compare static vs optimized
    if static_full > 0 and opt_full > 0:
        speedup = static_full / opt_full
        print(f"  → Optimized vs Static (with accel): {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    # Compare accel vs no-accel
    if opt_full > 0 and opt_no > 0:
        speedup = opt_full / opt_no
        print(f"  → No-accel vs Full-accel (optimized): {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    print()
    print(f"Total wall-clock time (parallel): {total_time:.2f}s")
    print("=" * 75)

    return {
        "results": [{"config": c, "result": r} for c, r in zip(configs, results)],
        "times": times_by_config,
        "total_time": total_time
    }


@app.local_entrypoint()
def test_muval():
    """Debug entrypoint to verify MuVal is installed and working."""
    print("=" * 60)
    print("  MuVal Debug Test")
    print("=" * 60)
    print()

    result = debug_muval.remote()

    print("MuVal binary exists:", result.get("muval_exists"))
    print("Config file exists:", result.get("config_exists"))
    print("call-muval script exists:", result.get("call_muval_exists"))
    print()

    print("OPAM ocaml path:", result.get("opam_ocaml"))
    if result.get("opam_error"):
        print("OPAM error:", result.get("opam_error"))
    print()

    print("MuVal --help output:")
    print(result.get("muval_help_output", "N/A"))
    if result.get("muval_help_stderr"):
        print("MuVal stderr:", result.get("muval_help_stderr"))
    print("MuVal returncode:", result.get("muval_help_returncode"))
    print()

    print("issy has --pruning option:", result.get("issy_help_has_pruning"))
    if result.get("issy_test_error"):
        print("issy error:", result.get("issy_test_error"))
    print()

    print("=" * 60)

    # Now run a quick synthesis with pruning 2 and check if MuVal was called
    print()
    print("Running quick synthesis with --pruning 2 to check if MuVal is invoked...")
    print()

    test_config = make_config(
        name="muval_test",
        grid_size=2,
        goal_x=1, goal_y=1,
        holes=[],
        objective="F ((eq x goalx) && (eq y goaly))",
        issy_binary="issy-optimized",
        accel="full"
    )

    synth_result = synthesize_and_validate.remote(
        config_yaml=test_config,
        game_name="ice_lake",
        task_id="muval_invoke_test"
    )

    print(f"Synthesis success: {synth_result.get('success')}")
    print(f"Synthesis time: {synth_result.get('synthesis_time', 0):.1f}s")
    print()
    print(f"*** MuVal call count: {synth_result.get('muval_call_count', 'N/A')} ***")
    print()
    if synth_result.get('muval_log'):
        print("MuVal log contents:")
        print("-" * 40)
        print(synth_result.get('muval_log'))
        print("-" * 40)
    else:
        print("NO MUVAL LOG FOUND - MuVal was NOT called by issy!")

    return result
