#!/usr/bin/env python3
"""Submit SLURM job(s) to the cluster and optionally monitor their output.

Features:
- Commit & push a local SLURM file (single source of truth) before submission.
- Submit one game or multiple games by setting GAME env var when calling sbatch remotely.
- Optionally stream the remote negotiation_<JOBID>.out (and .err) file via SSH.

Usage examples:
  python submit_and_monitor.py --slurm Masterthesis/run_gpu_job_final.slurm --commit --game resource_allocation --monitor
  python submit_and_monitor.py --slurm Masterthesis/run_gpu_job_final.slurm --game all --monitor

Note: adjust --user and --host for your cluster credentials.
"""
import argparse
import subprocess
import re
import shlex
import sys
import time


def run_local(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(res.stdout, end='')
    if res.stderr:
        print(res.stderr, end='', file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nReturn code: {res.returncode}")
    return res


def commit_and_push(slurm_path, repo_root='.'):
    # Stage, commit and push
    run_local(f'git -C {shlex.quote(repo_root)} add {shlex.quote(slurm_path)}')
    msg = f'Update SLURM: {slurm_path}'
    run_local(f'git -C {shlex.quote(repo_root)} commit -m "{msg}" || echo "Nothing to commit"')
    run_local(f'git -C {shlex.quote(repo_root)} push origin main')


def submit_remote(user, host, remote_slurm_path, game_env=None):
    # Build remote submission command. Use GAME env var to specify which game the script should run.
    env_prefix = f'GAME={shlex.quote(game_env)}' if game_env else ''
    cmd = f"cd ~/ && {env_prefix} sbatch {shlex.quote(remote_slurm_path)}"
    ssh_cmd = f'ssh {user}@{host} "{cmd}"'
    print(f"Submitting remote job for game={game_env or '<default>'}...")
    res = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    print(res.stdout, end='')
    if res.stderr:
        print(res.stderr, end='', file=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Remote sbatch failed: {res.stderr}")

    m = re.search(r'Submitted batch job (\d+)', res.stdout)
    if not m:
        # try searching stderr
        m = re.search(r'Submitted batch job (\d+)', res.stderr or '')
    if not m:
        raise RuntimeError(f"Could not parse sbatch output for job id: {res.stdout} {res.stderr}")
    jobid = m.group(1)
    print(f"Submitted job {jobid}")
    return jobid


def monitor_remote_output(user, host, jobid, follow=True, poll_interval=2, stream_err=False):
    out_path = f'~/negotiation_{jobid}.out'
    err_path = f'~/negotiation_{jobid}.err'

    try:
        if follow:
            # Use tail -F to follow file growth; if file doesn't exist yet, tail waits
            tail_cmd = f"ssh {user}@{host} 'tail -n +1 -F {out_path}'"
            print(f"Streaming {out_path} (Ctrl-C to stop)")
            p = subprocess.Popen(tail_cmd, shell=True)
            if stream_err:
                tail_err_cmd = f"ssh {user}@{host} 'tail -n +1 -F {err_path}'"
                print(f"Streaming {err_path} (Ctrl-C to stop)")
                p_err = subprocess.Popen(tail_err_cmd, shell=True)
            try:
                p.wait()
            except KeyboardInterrupt:
                print('\nStopped streaming (user interrupt)')
                p.terminate()
                if stream_err:
                    p_err.terminate()
        else:
            # One-shot fetch
            cmd = f"ssh {user}@{host} 'sed -n \'/ Testing RESOURCE ALLOCATION/,/ Resource allocation test PASSED/p\' {out_path} || true'"
            run_local(cmd)
            if stream_err:
                cmd_err = f"ssh {user}@{host} 'sed -n \'/ Testing RESOURCE ALLOCATION/,/ Resource allocation test PASSED/p\' {err_path} || true'"
                run_local(cmd_err)

    except Exception as e:
        print(f"Error while monitoring: {e}")


def main():
    parser = argparse.ArgumentParser(description="Commit SLURM, submit job(s) and optionally monitor output on the cluster")
    parser.add_argument('--slurm', required=True, help='Local SLURM file path to commit and that exists on remote as run_gpu_job_final.slurm')
    parser.add_argument('--commit', action='store_true', help='Commit & push the SLURM file before submission')
    parser.add_argument('--user', default='s391129', help='Cluster SSH user')
    parser.add_argument('--host', default='julia2.hpc.uni-wuerzburg.de', help='Cluster host')
    parser.add_argument('--remote-slurm-path', default='run_gpu_job_final.slurm', help='Remote SLURM path in home directory')
    parser.add_argument('--game', default='resource_allocation', help='Which game to run: resource_allocation, integrative_negotiation, price_bargaining, or "all"')
    parser.add_argument('--monitor', action='store_true', help='Stream negotiation_<jobid>.out after submission')
    parser.add_argument('--monitor-err', action='store_true', help='Also stream the corresponding .err file')
    args = parser.parse_args()

    repo_root = '.'
    slurm_path = args.slurm

    if args.commit:
        print('Committing SLURM file...')
        commit_and_push(slurm_path, repo_root=repo_root)

    games = []
    if args.game == 'all':
        games = ['resource_allocation', 'integrative_negotiation', 'price_bargaining']
    else:
        games = [args.game]

    submitted = []
    for g in games:
        jobid = submit_remote(args.user, args.host, args.remote_slurm_path, game_env=g)
        submitted.append((g, jobid))
        # small pause to avoid overwhelming the scheduler
        time.sleep(1)

    if args.monitor:
        # If multiple jobs were started, monitor them sequentially
        for g, jobid in submitted:
            print(f"--- Monitoring game={g} job={jobid} ---")
            monitor_remote_output(args.user, args.host, jobid, follow=True, stream_err=args.monitor_err)


if __name__ == '__main__':
    main()
