import streamlit as st
import subprocess
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'negotiation_platform' / 'configs'
SLURM_LOCAL = REPO_ROOT / 'run_gpu_job_final.slurm'
SUBMIT_SCRIPT = REPO_ROOT / 'scripts' / 'submit_and_monitor.py'


def run_cmd(cmd, capture_output=True):
    """Run a command either as a list (preferred, no shell) or a shell string.

    If cmd is a list/tuple, subprocess is invoked with shell=False to avoid quoting issues on Windows/PowerShell.
    """
    try:
        if isinstance(cmd, (list, tuple)):
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        else:
            # fallback to shell string
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p
    except Exception as e:
        st.error(f"Failed to run command: {e}")
        return None


def scp_push(local_path: Path, user: str, host: str, key_path: str = None, remote_path: str = None):
    if remote_path is None:
        remote_path = f'~/negotiation_platform/configs/{local_path.name}'
    key_part = f"-i {shlex.quote(key_path)}" if key_path else ""
    cmd = f"scp {key_part} {shlex.quote(str(local_path))} {shlex.quote(user)}@{shlex.quote(host)}:{shlex.quote(remote_path)}"
    st.info(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        st.error(res.stderr or res.stdout)
    else:
        st.success(f"Pushed {local_path.name} to {host}:{remote_path}")


def submit_job(user, host, game, commit, key_path):
    # Build argument list and invoke with the current Python executable to avoid shell quoting problems on Windows
    args = [sys.executable, str(SUBMIT_SCRIPT), '--slurm', str(SLURM_LOCAL), '--game', game, '--user', user, '--host', host]
    if commit:
        args.append('--commit')
    st.info(f"Submitting with: {' '.join(shlex.quote(a) for a in args)}")
    p = run_cmd(args)
    if not p:
        return None
    # read first few lines to capture job id
    jobid = None
    out_area = st.empty()
    output_text = ""
    try:
        for line in p.stdout:
            output_text += line
            out_area.text(output_text)
            if 'Submitted batch job' in line:
                parts = line.strip().split()
                jobid = parts[-1]
        p.wait()
    except Exception as e:
        st.error(f"Error while submitting: {e}")
    return jobid


def monitor_job(user, host, jobid, key_path=None):
    out_path = f"~/negotiation_{jobid}.out"
    key_part = f"-i {shlex.quote(key_path)}" if key_path else ""
    cmd = f"ssh {key_part} {shlex.quote(user)}@{shlex.quote(host)} 'tail -n +1 -F {out_path}'"
    st.info(f"Monitoring {out_path} on {host} (Ctrl-C in terminal to stop)")
    p = run_cmd(cmd)
    if not p:
        return
    out_area = st.empty()
    output_text = ""
    try:
        for line in p.stdout:
            output_text += line
            out_area.text(output_text)
    except Exception as e:
        st.error(f"Monitoring stopped: {e}")


def list_yaml_files():
    files = list(CONFIG_DIR.glob('*.yaml'))
    return [f.name for f in files]


def load_yaml(name):
    p = CONFIG_DIR / name
    if not p.exists():
        return ''
    return p.read_text(encoding='utf-8')


def save_yaml(name, content):
    p = CONFIG_DIR / name
    p.write_text(content, encoding='utf-8')
    return p


def main():
    st.set_page_config(page_title='Negotiation Runner', layout='wide')

    st.title('Negotiation Platform — Submit & Monitor')

    menu = st.sidebar.radio('Page', ['Run', 'Config'])

    if menu == 'Run':
        st.header('Start a job')
        user = st.text_input('SSH user', value='s391129')
        host = st.text_input('SSH host', value='julia2.hpc.uni-wuerzburg.de')
        key_path = st.text_input('SSH private key path (optional)', value='')

        game = st.selectbox('Game to run', ['resource_allocation', 'integrative_negotiation', 'price_bargaining', 'all'])
        commit = st.checkbox('Commit & push SLURM before submitting', value=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Start job'):
                jobid = submit_job(user, host, game, commit, key_path)
                if jobid:
                    st.success(f'Job submitted: {jobid}')
                    if st.checkbox('Monitor job output now'):
                        monitor_job(user, host, jobid, key_path or None)
        with col2:
            st.markdown('''
            **Quick actions**
            - Ensure your SSH key is loaded or provided.
            - If you prefer password login, open an SSH session manually then run the submit command.
            ''')

    else:
        st.header('Edit YAML configs')
        files = list_yaml_files()
        sel = st.selectbox('Config file', files)
        content = load_yaml(sel)
        edited = st.text_area('YAML content', value=content, height=500)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Save locally'):
                p = save_yaml(sel, edited)
                st.success(f'Saved {p}')
        with col2:
            user = st.text_input('SSH user for push', value='s391129')
            host = st.text_input('SSH host for push', value='julia2.hpc.uni-wuerzburg.de')
            key_path = st.text_input('SSH private key path (optional)', value='')
            if st.button('Push to cluster'):
                p = save_yaml(sel, edited)
                scp_push(p, user, host, key_path)


if __name__ == '__main__':
    main()
