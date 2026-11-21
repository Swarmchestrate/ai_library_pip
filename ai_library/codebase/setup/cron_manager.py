import os
import sys
import subprocess
import shlex
import yaml
from ..helpers.config_helper import validate_config

def train_to_cron():
    # 1. Gather current execution context
    python_path = sys.executable  # Path to current Python virtualenv/environment
    working_dir = os.getcwd()      # Directory where this script is run
    config = validate_config()
    retrain_frequency = config.get("retrain_frequency")
    # 2. Construct the full command
    # Runs python -m ai_library.codebase.setup.train and appends logs to cron.log
    log_file = os.path.join(working_dir, "cron.log")
    # Use the package-qualified module path so the cron job works when the
    # package is installed (or when run from the repo root with PYTHONPATH).
    command = f"cd {working_dir} && {python_path} -m ai_library.codebase.setup.train >> {log_file} 2>&1"
    
    # 3. Define schedule: Every {retrain_frequency} (*/{retrain_frequency} * * * *)
    cron_schedule = f"*/{retrain_frequency.split('m')[0]} * * * *"
    cron_entry = f"{cron_schedule} {command}"
    
    # 4. Fetch existing crontab
    try:
        current_cron = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        current_cron = ""  # No crontab exists for this user yet
        
    # 5. Avoid adding duplicate entries
    if command in current_cron:
        print("Job already exists in crontab. No changes made.")
        return

    # 6. Append new job and update crontab
    new_cron = current_cron + cron_entry + "\n"
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(input=new_cron)
    
    if process.returncode == 0:
        print("Successfully added job to cron!")
        print(f"Scheduled entry:\n{cron_entry}")
    else:
        print("Failed to update crontab.", file=sys.stderr)

def remove_train_from_cron():
    # 1. Gather current execution context
    python_path = sys.executable  # Path to current Python virtualenv/environment
    working_dir = os.getcwd()      # Directory where this script is run
    config = validate_config()
    retrain_frequency = config.get("retrain_frequency")
    # 2. Construct the full command
    log_file = os.path.join(working_dir, "cron.log")
    command = f"cd {working_dir} && {python_path} -m ai_library.codebase.setup.train >> {log_file} 2>&1"
    
    # 3. Fetch existing crontab
    try:
        current_cron = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        print("No crontab exists for this user. Nothing to remove.")
        return
        
    # 4. Remove the specific job entry if it exists
    if command not in current_cron:
        print("Job not found in crontab. No changes made.")
        return

    new_cron = "\n".join(
        line for line in current_cron.splitlines() if command not in line
    ) + "\n"
    
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(input=new_cron)
    
    if process.returncode == 0:
        print("Successfully removed job from cron!")
        print(f"Removed entry:\n{command}")
    else:
        print("Failed to update crontab.", file=sys.stderr)

def _adwin_cron_command():
    python_path = sys.executable
    working_dir = os.getcwd()
    log_file = os.path.join(working_dir, "adwin_cron.log")
    script = "from ai_library.codebase.helpers.adwin import apply_adwin; apply_adwin()"
    return (
        f"cd {shlex.quote(working_dir)} && {shlex.quote(python_path)} "
        f"-c {shlex.quote(script)} >> {shlex.quote(log_file)} 2>&1"
    )


def adwin_to_cron():
    config = validate_config()
    retrain_frequency = config.get("retrain_frequency")
    interval_minutes = retrain_frequency.split("m")[0]
    cron_schedule = f"*/{interval_minutes} * * * *"
    command = _adwin_cron_command()
    cron_entry = f"{cron_schedule} {command}"

    try:
        current_cron = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        current_cron = ""

    if command in current_cron:
        print("ADWIN job already exists in crontab. No changes made.")
        return

    new_cron = current_cron + cron_entry + "\n"
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(input=new_cron)

    if process.returncode == 0:
        print("Successfully added ADWIN job to cron!")
        print(f"Scheduled entry:\n{cron_entry}")
    else:
        print("Failed to update crontab.", file=sys.stderr)


def remove_adwin_from_cron():
    command = _adwin_cron_command()

    try:
        current_cron = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        print("No crontab exists for this user. Nothing to remove.")
        return

    if command not in current_cron:
        print("ADWIN job not found in crontab. No changes made.")
        return

    new_cron = "\n".join(
        line for line in current_cron.splitlines() if command not in line
    ) + "\n"
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(input=new_cron)

    if process.returncode == 0:
        print("Successfully removed ADWIN job from cron!")
        print(f"Removed entry:\n{command}")
    else:
        print("Failed to update crontab.", file=sys.stderr)
