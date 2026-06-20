#!/usr/bin/env python3
"""
CTDenoiser Training Management - Kubernetes Jobs

Usage:
    python sweep.py                       # Print help
    python sweep.py sweep.yml             # Sweep the abdomen cache + 1 agent
    python sweep.py sweep.yml --agents 8  # Sweep abdomen + 8 agents
    python sweep.py sweep.yml --anatomy chest --agents 8   # Sweep /data/ldct_chest.h5
    python sweep.py --deploy SWEEP_ID     # Deploy agents for existing sweep + watch

Each anatomy is downloaded once (data pod -> /data/ldct_<anatomy>.h5) and swept
separately; merge the exports with scripts/benchmark_report.py for one report.
"""

import os
import sys
import time
import json
import yaml
import wandb
import subprocess
import argparse
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'k8s', 'tr_job_template.yml')
NAMESPACE = 'usd-djha'
DEFAULT_ENTITY = 'timgsereda'
DEFAULT_PROJECT = 'ctdenoiser-sweep'


# ============================================================================
# WANDB SWEEP MANAGEMENT
# ============================================================================

def load_sweep_config(config_path):
    """Load sweep configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found!")
        sys.exit(1)


def create_sweep(config_path):
    """Create a new wandb sweep and return sweep ID."""
    config = load_sweep_config(config_path)

    entity = config.get('entity', DEFAULT_ENTITY)
    project = config.get('project', DEFAULT_PROJECT)

    print(f"\nCreating W&B sweep in {entity}/{project}")

    try:
        sweep_id = wandb.sweep(config, entity=entity, project=project)
        print(f"Sweep created: {sweep_id}")
        print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"Error creating sweep: {e}")
        return None


# ============================================================================
# KUBERNETES INDEXED JOB
# ============================================================================

def generate_indexed_job(sweep_id, entity, project, num_agents=4,
                         h5_name='ldct_abdomen.h5'):
    """Generate an Indexed Job YAML from template.

    ``h5_name`` is the preprocessed cache on the PVC to sweep (the data pod
    writes one ``ldct_<anatomy>.h5`` per anatomy). Each anatomy is its own
    sweep over its own cache; the anatomy/window is logged from the file's
    attrs so the runs stay distinguishable in one W&B project.
    """
    try:
        with open(TEMPLATE_PATH, 'r') as f:
            job_yaml = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Template file '{TEMPLATE_PATH}' not found!")
        sys.exit(1)

    job_yaml['spec']['completions'] = num_agents
    job_yaml['spec']['parallelism'] = num_agents

    container = job_yaml['spec']['template']['spec']['containers'][0]

    wandb_command = f"wandb agent {entity}/{project}/{sweep_id}"
    container['args'][0] = f"""set -e

apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

pip install wandb matplotlib h5py

# Copy preprocessed HDF5 from PVC to local emptyDir for fast I/O
echo "Copying preprocessed data ({h5_name})..."
t0=$(date +%s)
cp /data/{h5_name} /workspace/data.h5
echo "Copy done: $(du -h /workspace/data.h5 | cut -f1) in $(($(date +%s) - t0))s"

git clone https://github.com/tsereda/ctdenoiser.git /workspace/ctdenoiser
cd /workspace/ctdenoiser
pip install -e .

{wandb_command}
"""

    for env_var in container['env']:
        if env_var['name'] == 'WANDB_PROJECT':
            env_var['value'] = project
        elif env_var['name'] == 'WANDB_ENTITY':
            env_var['value'] = entity

    output_dir = os.path.join(SCRIPT_DIR, 'k8s')
    output_file = os.path.join(output_dir, 'sweep_job.yml')

    with open(output_file, 'w') as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_file}")
    print(f"  completions={num_agents}, parallelism={num_agents}")

    return output_file


def deploy_job(job_file):
    """Deploy job to Kubernetes, replacing any existing job with the same name."""
    print(f"\nDeploying Indexed Job to Kubernetes (namespace: {NAMESPACE})...")

    try:
        with open(job_file, 'r') as f:
            job_yaml = yaml.safe_load(f)
        job_name = job_yaml['metadata']['name']

        existing = subprocess.run(
            ['kubectl', 'get', 'job', job_name, '-n', NAMESPACE],
            capture_output=True, text=True
        )
        if existing.returncode == 0:
            print(f"  Deleting existing job '{job_name}'...")
            subprocess.run(
                ['kubectl', 'delete', 'job', job_name, '-n', NAMESPACE,
                 '--cascade=foreground', '--wait=true'],
                capture_output=True, text=True, check=True
            )
            print(f"  Deleted.")

        result = subprocess.run(
            ['kubectl', 'apply', '-f', job_file, '-n', NAMESPACE],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  Error: kubectl not found. Deploy manually:")
        print(f"     kubectl apply -f {job_file} -n {NAMESPACE}")
        return False


def delete_jobs():
    """Delete all wandb sweep jobs from Kubernetes."""
    print(f"\nDeleting all wandb sweep jobs (namespace: {NAMESPACE})...")

    try:
        list_result = subprocess.run(
            ['kubectl', 'get', 'jobs', '-n', NAMESPACE,
             '-l', 'app=wandb-sweep', '-o', 'name'],
            capture_output=True,
            text=True,
            check=True
        )

        jobs = [j.strip() for j in list_result.stdout.strip().split('\n') if j.strip()]

        if not jobs:
            print("  No jobs found with label app=wandb-sweep")
            return True

        print(f"  Found {len(jobs)} job(s) to delete:")
        for job in jobs:
            print(f"    - {job}")

        confirm = input("\n  Delete these jobs? [y/N]: ").lower().strip()
        if confirm != 'y':
            print("  Deletion cancelled")
            return False

        result = subprocess.run(
            ['kubectl', 'delete', 'jobs', '-n', NAMESPACE,
             '-l', 'app=wandb-sweep'],
            capture_output=True,
            text=True,
            check=True
        )

        print(f"\n  Deleted successfully:")
        print(f"  {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  Error: Failed to delete jobs: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  Error: kubectl not found. Please delete manually:")
        print(f"     kubectl delete jobs -n {NAMESPACE} -l app=wandb-sweep")
        return False


# ============================================================================
# POD MONITORING
# ============================================================================

JOB_NAME = 'ctdenoiser-sweep'
POD_LABEL = f'job-name={JOB_NAME}'

WAITING_PHASES = {'Pending', 'ContainerCreating', 'PodInitializing'}
BAD_STATES = {'Error', 'CrashLoopBackOff', 'ImagePullBackOff',
              'ErrImagePull', 'CreateContainerConfigError', 'OOMKilled'}


def _kubectl(*args, check=True):
    """Run kubectl with namespace and return stdout."""
    cmd = ['kubectl', '-n', NAMESPACE] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout.strip()


def _get_pods_json():
    """Get pod list as parsed JSON."""
    raw = _kubectl('get', 'pods', '-l', POD_LABEL, '-o', 'json', check=False)
    if not raw:
        return []
    return json.loads(raw).get('items', [])


def _pod_status(pod):
    """Extract a human-readable status from a pod object."""
    phase = pod['status'].get('phase', 'Unknown')
    for cs in pod['status'].get('containerStatuses', []):
        state = cs.get('state', {})
        if 'waiting' in state:
            return state['waiting'].get('reason', 'Waiting')
        if 'terminated' in state:
            reason = state['terminated'].get('reason', '')
            return reason if reason else ('Completed' if state['terminated'].get('exitCode') == 0 else 'Error')
        if 'running' in state:
            return 'Running'
    return phase


def _print_pod_table(pods):
    """Print a compact pod status table."""
    if not pods:
        print("  No pods found")
        return
    name_w = max(len(p['metadata']['name']) for p in pods)
    for p in pods:
        name = p['metadata']['name']
        status = _pod_status(p)
        node = p['spec'].get('nodeName', '<unscheduled>')
        age_s = ''
        start = p['status'].get('startTime')
        if start:
            dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            delta = datetime.now(timezone.utc) - dt
            mins = int(delta.total_seconds() // 60)
            age_s = f'{mins}m' if mins < 60 else f'{mins // 60}h{mins % 60}m'
        print(f"  {name:<{name_w}}  {status:<28} {node}  {age_s}")


def watch_pods(interval=5, timeout=600):
    """Poll pods until all have stabilized, then show describe/logs."""
    print(f"\nWatching pods (label: {POD_LABEL}, poll every {interval}s)...\n")
    deadline = time.time() + timeout

    while time.time() < deadline:
        pods = _get_pods_json()
        # Clear line and redraw
        sys.stdout.write(f"\033[2J\033[H")
        print(f"=== Pod Status (namespace: {NAMESPACE}) ===\n")
        _print_pod_table(pods)
        print(f"\n  (timeout in {int(deadline - time.time())}s, Ctrl+C to stop)\n")

        if not pods:
            time.sleep(interval)
            continue

        statuses = [_pod_status(p) for p in pods]
        all_settled = all(s not in WAITING_PHASES for s in statuses)

        if all_settled:
            failed = [(p, s) for p, s in zip(pods, statuses) if s in BAD_STATES]
            running = [(p, s) for p, s in zip(pods, statuses) if s == 'Running']
            succeeded = [(p, s) for p, s in zip(pods, statuses) if s == 'Completed']

            if failed:
                print(f"--- {len(failed)} pod(s) in error state, showing describe ---\n")
                for p, s in failed:
                    name = p['metadata']['name']
                    print(f"=== kubectl describe pod {name} ===")
                    print(_kubectl('describe', 'pod', name))
                    print()
            if succeeded:
                print(f"--- {len(succeeded)} pod(s) completed ---\n")
            if running:
                print(f"--- {len(running)} pod(s) running, tailing logs from first ---\n")
                name = running[0][0]['metadata']['name']
                print(f"=== kubectl logs -f {name} ===\n")
                try:
                    subprocess.run(
                        ['kubectl', '-n', NAMESPACE, 'logs', '-f', name],
                        check=False
                    )
                except KeyboardInterrupt:
                    pass
            if not failed and not running:
                print("All pods completed.")
            return

        time.sleep(interval)

    print("\nTimeout reached. Final pod state:")
    _print_pod_table(_get_pods_json())


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CTDenoiser Training Management - Create sweep + deploy training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python sweep.py sweep.yml                     # Sweep abdomen cache + 1 agent
  python sweep.py sweep.yml --agents 8          # Sweep abdomen + 8 agents
  python sweep.py sweep.yml --anatomy chest     # Sweep /data/ldct_chest.h5
  python sweep.py sweep.yml --h5-name ldct_preprocessed.h5   # legacy cache
  python sweep.py --deploy dnffyu6j             # Deploy agents for existing sweep + watch

  python sweep.py --delete                      # Delete all sweep jobs
        """
    )

    parser.add_argument('sweep_file', nargs='?',
                        help='Path to sweep configuration file')
    parser.add_argument('--agents', type=int, default=1,
                        help='Number of agents to deploy (default: 1)')
    parser.add_argument('--anatomy', default='abdomen',
                        choices=['abdomen', 'chest', 'head'],
                        help='which per-anatomy cache to sweep: /data/ldct_<anatomy>.h5 '
                             '(default: abdomen)')
    parser.add_argument('--h5-name', default=None,
                        help='explicit PVC cache filename, overriding --anatomy '
                             '(e.g. ldct_preprocessed.h5 for a legacy cache)')
    parser.add_argument('--deploy', type=str, metavar='SWEEP_ID',
                        help='Deploy agents for existing sweep ID')
    parser.add_argument('--delete', action='store_true',
                        help='Delete all wandb sweep jobs')

    args = parser.parse_args()

    if args.delete:
        delete_jobs()
        sys.exit(0)

    if not args.sweep_file and not args.deploy:
        parser.print_help()
        sys.exit(0)

    if args.deploy:
        sweep_id = args.deploy
        print(f"\nUsing existing sweep: {sweep_id}")
        config = load_sweep_config(os.path.join(SCRIPT_DIR, 'sweep.yml'))
        entity = config.get('entity', DEFAULT_ENTITY)
        project = config.get('project', DEFAULT_PROJECT)
    elif args.sweep_file:
        print("\nCreating W&B sweep...")
        sweep_id = create_sweep(config_path=args.sweep_file)

        if not sweep_id:
            print("Error: Failed to create sweep. Aborting...")
            sys.exit(1)

        config = load_sweep_config(args.sweep_file)
        entity = config.get('entity', DEFAULT_ENTITY)
        project = config.get('project', DEFAULT_PROJECT)

    print(f"\nGenerating Indexed Job YAML ({args.agents} agents)...")
    h5_name = args.h5_name or f"ldct_{args.anatomy}.h5"
    print(f"Sweeping cache /data/{h5_name}")
    job_file = generate_indexed_job(
        sweep_id=sweep_id,
        entity=entity,
        project=project,
        num_agents=args.agents,
        h5_name=h5_name,
    )

    print(f"\nDeploying job to Kubernetes...")
    deploy_job(job_file)

    print(f"\nSweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

    print(f"\nWatching pods...")
    watch_pods()

    print(f"\nDelete all jobs:")
    print(f"  python sweep.py --delete")


if __name__ == "__main__":
    main()
