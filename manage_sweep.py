#!/usr/bin/env python3
"""
CTDenoiser Training Management - Kubernetes Jobs

Usage:
    python manage_sweep.py                      # Print help
    python manage_sweep.py sweep.yml            # Create sweep + deploy 1 agent
    python manage_sweep.py sweep.yml --agents 8 # Create sweep + deploy 8 agents
    python manage_sweep.py --deploy SWEEP_ID    # Deploy agents for existing sweep
"""

import os
import sys
import yaml
import wandb
import subprocess
import argparse

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

def generate_indexed_job(sweep_id, entity, project, num_agents=4):
    """Generate an Indexed Job YAML from template."""
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
    container['args'][0] = f"""pip install wandb h5py matplotlib

git clone https://github.com/tsereda/ctdenoiser.git
cd ctdenoiser
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
    """Deploy job to Kubernetes."""
    print(f"\nDeploying Indexed Job to Kubernetes (namespace: {NAMESPACE})...")

    try:
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
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CTDenoiser Training Management - Create sweep + deploy training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Create sweep + deploy 1 agent
  python manage_sweep.py sweep.yml

  # Create sweep + deploy 8 agents
  python manage_sweep.py sweep.yml --agents 8

  # Deploy agents to existing sweep
  python manage_sweep.py --deploy dnffyu6j

Tip:
  kubectl get jobs -n {NAMESPACE} -l app=wandb-sweep
  kubectl get pods -n {NAMESPACE} -l job-name=ctdenoiser-sweep
  kubectl logs -f -n {NAMESPACE} job/ctdenoiser-sweep
  python manage_sweep.py --delete
        """
    )

    parser.add_argument('sweep_file', nargs='?',
                        help='Path to sweep configuration file')
    parser.add_argument('--agents', type=int, default=1,
                        help='Number of agents to deploy (default: 1)')
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
    job_file = generate_indexed_job(
        sweep_id=sweep_id,
        entity=entity,
        project=project,
        num_agents=args.agents
    )

    print(f"\nDeploying job to Kubernetes...")
    deploy_job(job_file)

    print("DONE")

    print(f"\nSweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

    print(f"\nMonitor:")
    print(f"  kubectl get jobs -n {NAMESPACE} -l app=wandb-sweep")
    print(f"  kubectl get pods -n {NAMESPACE} -l job-name=ctdenoiser-sweep")
    print(f"  kubectl logs -f -n {NAMESPACE} job/ctdenoiser-sweep")

    print(f"\nDelete all jobs:")
    print(f"  python manage_sweep.py --delete")


if __name__ == "__main__":
    main()
