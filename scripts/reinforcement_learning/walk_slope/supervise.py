"""Bounded train/evaluate cycles with checkpoint provenance and independent acceptance."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--block', type=int, default=500)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--completed-iterations', type=int, default=0)
    args = parser.parse_args()
    root = args.experiment.resolve()
    lock = root/'supervisor.lock'
    if lock.exists():
        pid = int(lock.read_text())
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Supervisor already active: {pid}')
    lock.write_text(str(os.getpid()))
    status = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_at': now(),
              'completed_iterations': args.completed_iterations, 'history': [],
              'iteration_budget': args.max_iterations, 'screen_pass_streak': 0}
    checkpoint = args.continue_from.resolve() if args.continue_from else root/'parent_model_4200.pt'
    runtime = root/'specialization_source/run.py'
    environment = os.environ.copy()
    environment['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    active_child = None

    def stop(signum, frame):
        if active_child is not None and active_child.poll() is None:
            active_child.send_signal(signal.SIGINT)
        raise KeyboardInterrupt('User or supervisor stop signal')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def run(mode, name, extra):
        nonlocal active_child
        if shutil.disk_usage(root).free < 5*1024**3:
            raise RuntimeError('Less than 5 GiB disk available; preserve all checkpoints and request review')
        out = root/name
        if out.exists():
            raise RuntimeError(f'Refusing to overwrite existing phase: {out}')
        command = [sys.executable, '-u', str(runtime), '--mode', mode,
                   '--source', str(root/'training_source_snapshot'), '--checkpoint', str(checkpoint),
                   '--output', str(out), '--headless', '--device', 'cuda:0', *extra]
        status.update(status=mode, phase=name, current_checkpoint=str(checkpoint), updated_at=now())
        with (root/f'{name}.log').open('w') as log:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active_child = child
            status.update(child_pid=child.pid, command=command)
            write(root/'status.json', status)
            result = child.wait()
            active_child = None
        if result:
            raise RuntimeError(f'{name} exited {result}; see {root/name}.log')
        invalid = ('out of memory', 'capacity exceeded', 'increase the gpu', 'nan detected')
        log_text = (root/f'{name}.log').read_text(errors='replace').lower()
        if any(token in log_text for token in invalid):
            raise RuntimeError(f'{name}: GPU/physics/numerical diagnostic needs review')
        return out

    try:
        while status['completed_iterations'] < args.max_iterations:
            count = min(args.block, args.max_iterations-status['completed_iterations'])
            target = status['completed_iterations']+count
            extra = ['--num-envs', str(args.num_envs), '--iterations', str(count), '--seed', '42']
            if checkpoint.name != 'parent_model_4200.pt':
                extra += ['--resume']
            phase = run('train', f'train_{target:05d}', extra)
            event = json.loads((phase/'completed.json').read_text())
            if event['iterations_completed'] != count:
                raise RuntimeError('Training completion count mismatch')
            checkpoint = Path(event['checkpoint'])
            status['completed_iterations'] = target
            evaluation = run('evaluate', f'eval_{target:05d}', ['--replicates', '8', '--seed', '1042'])
            summary = json.loads((evaluation/'summary.json').read_text())
            entry = {'iterations': target, 'checkpoint': str(checkpoint), 'evaluation': str(evaluation),
                     'worst_target_success_rate': summary['worst_target_success_rate'], 'passed': summary['passed'],
                     'total_successes': sum(c['successes'] for c in summary['cells'].values()),
                     'mean_tracking_rmse': sum(c['mean_tracking_rmse'] for c in summary['cells'].values())/len(summary['cells']),
                     'course': event['course']}
            status['history'].append(entry)
            best = status.get('best')
            rank = lambda item: (item['worst_target_success_rate'], item['total_successes'], -item['mean_tracking_rmse'])
            if best is None or rank(entry) > rank(best):
                status['best'] = entry
            status['screen_pass_streak'] = status['screen_pass_streak']+1 if summary['passed'] else 0
            if status['screen_pass_streak'] >= 2:
                confirmations = []
                for seed in (2042, 3042):
                    path = run('evaluate', f'confirm_{target:05d}_{seed}', ['--replicates', '32', '--seed', str(seed)])
                    confirmations.append(json.loads((path/'summary.json').read_text()))
                if all(item['passed'] for item in confirmations):
                    status.update(status='accepted', accepted_checkpoint=str(checkpoint), finished_at=now(),
                                  acceptance='Two consecutive fixed screens and two independent 864-episode confirmation batches; every +/-30deg and flat speed/friction cell >=95% success.')
                    write(root/'status.json', status)
                    return
                status['screen_pass_streak'] = 0
            write(root/'status.json', status)
        status.update(status='needs_review', reason='Initial iteration budget exhausted without fixed acceptance; inspect failure conditions and adapt continuation.', finished_at=now())
    except BaseException as error:
        status.update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed',
                      reason=str(error), traceback=traceback.format_exc(), finished_at=now())
    finally:
        write(root/'status.json', status)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
