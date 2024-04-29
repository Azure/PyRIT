#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def load_logs(logdir, mode='behaviors', max_examples=100):
    # Build file path pattern
    pattern = f"{logdir}individual_{mode}_*_ascii*"

    # List files that match the pattern and contain 'json'
    files = [f for f in os.listdir(logdir) if f.startswith(f"individual_{mode}_") and f.endswith(".json")]
    files = sorted(files, key=lambda x: "_".join(x.split('_')[:-1]))

    logs = []
    # Load data from files
    for logfile in files:
        path = os.path.join(logdir, logfile)
        try:
            with open(path, 'r') as f:
                logs.append(json.load(f))
        except Exception as e:
            print(f"Failed to read {logfile}: {str(e)}")

    

    return logs

def main():
    method = 'gcg'
    logdir = f'results/' # Adjust path as necessary

    # Load logs
    logs = load_logs(logdir, mode='behaviors')
    if logs:
        log = logs[0]
        print(f"Loaded {len(logs)} logs.")

    config = log['params']
    print(config.keys())

    total_steps = config['n_steps']
    test_steps = config.get('test_steps', 50)
    log_steps = total_steps // test_steps + 1
    print('log_steps', log_steps)


    examples = 0
    test_logs = []
    control_logs = []
    goals, targets = [],[]
    for l in logs:
        sub_test_logs = l['tests']
        sub_examples = len(sub_test_logs) // log_steps
        examples += sub_examples
        test_logs.extend(sub_test_logs[:sub_examples * log_steps])
        control_logs.extend(l['controls'][:sub_examples * log_steps])
        goals.extend(l['params']['goals'][:sub_examples])
        targets.extend(l['params']['targets'][:sub_examples])
    
    print(examples)

    passed, em, loss, total, controls = [],[],[],[],[]
    for i in range(examples):
        sub_passed, sub_em, sub_loss, sub_total, sub_control = [],[],[],[],[]
        for res in test_logs[i*log_steps:(i+1)*log_steps]:
            sub_passed.append(res['n_passed'])
            sub_em.append(res['n_em'])
            sub_loss.append(res['n_loss'])
            sub_total.append(res['total'])
        sub_control = control_logs[i*log_steps:(i+1)*log_steps]
        passed.append(sub_passed)
        em.append(sub_em)
        loss.append(sub_loss)
        total.append(sub_total)
        controls.append(sub_control)
    passed = np.array(passed)
    em = np.array(em)
    loss = np.array(loss)
    total = np.array(total)
    print(total.shape)

    saved_controls = [c[-1] for c in controls]
    json_obj = {
        'goal': goals,
        'target': targets,
        'controls': saved_controls
    }
    print(saved_controls)
    with open('results/individual_behavior_controls.json', 'w') as f:
        json.dump(json_obj, f)

    # data = json.load(open('eval/individual_behavior_controls.json', 'r'))
     
if __name__ == '__main__':
    main()
