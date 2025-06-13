from flask import Flask, render_template, request
import sys
import os
import numpy as np
import torch
from stable_baselines3 import DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'logic')))
from logic.predictor import predict_complexity
from logic.sorting_lstm import SortingLSTM, predict_next_swap
from logic.sorting_env import SortingEnv
from logic.stack_queue_rnn import StackQueueRNN, predict_output
from logic.binary_tree_gnn import GNN, predict_inorder

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    sort_steps = None
    rl_steps = None
    stack_queue_results = None
    tree_results = None
    
    if request.method == 'POST':
        tab = request.form.get('tab', 'complexity')
        
        if tab == 'complexity':
            code = request.form.get('code')
            if code:
                prediction = predict_complexity(code)
        
        elif tab == 'sort_lstm':
            input_list = request.form.get('sort_list')
            algo_type = request.form.get('algo_type', 'bubble_sort')
            if input_list:
                try:
                    arr = [int(x) for x in input_list.split(',')]
                    if len(arr) != 10:
                        sort_steps = {"error": "Enter 10 numbers"}
                    else:
                        model = SortingLSTM()
                        model.load_state_dict(torch.load("models/sorting_lstm.pt"))
                        steps = []
                        current = arr.copy()
                        for _ in range(20):
                            step = predict_next_swap(model, current, algo_type)
                            steps.append({"state": current.copy(), "action": step["action"], "type": step["type"]})
                            if "swap" in step["action"]:
                                i, j = step["action"]["swap"]
                                if i < len(current) and j < len(current):
                                    current[i], current[j] = current[j], current[i]
                            elif "merge" in step["action"]:
                                k = step["action"]["merge"][0]
                                if k < len(current):
                                    current[k] = min(current[k], current[k])
                            if all(current[i] <= current[i+1] for i in range(len(current)-1)):
                                break
                        sort_steps = steps
                except Exception as e:
                    sort_steps = {"error": f"Invalid input: {e}"}
        
        elif tab == 'sort_rl':
            input_list = request.form.get('sort_list')
            if input_list:
                try:
                    arr = [int(x) for x in input_list.split(',')]
                    if len(arr) != 10:
                        rl_steps = {"error": "Enter 10 numbers"}
                    else:
                        env = SortingEnv(list_size=10)
                        env.state = np.array(arr)
                        model = DQN.load("models/sorting_dqn")
                        obs = env.state
                        done = False
                        steps = []
                        for _ in range(20):
                            action, _ = model.predict(obs)
                            obs, reward, done, info = env.step(action)
                            steps.append({"state": obs.tolist(), "swap": info["swap"]})
                            print(f"RL Step: state={obs.tolist()}, swap={info['swap']}, reward={reward}")
                            if done:
                                break
                        rl_steps = steps
                except Exception as e:
                    rl_steps = {"error": f"Invalid input: {e}"}
        
        elif tab == 'stack_queue':
            sequence = request.form.get('sequence')
            ds_type = request.form.get('ds_type', 'stack')
            if sequence:
                try:
                    ops = [op.strip() for op in sequence.split(';')]
                    outputs = predict_output(ops, ds_type)
                    stack_queue_results = [{"op": op, "output": out} for op, out in zip(ops, outputs)]
                except ValueError as e:
                    stack_queue_results = {"error": f"Invalid operation: {e}"}
                except Exception as e:
                    stack_queue_results = {"error": f"Unexpected error: {e}"}
        
        elif tab == 'binary_tree':
            nodes = request.form.get('nodes')
            edges = request.form.get('edges')
            if nodes and edges:
                try:
                    node_list = [int(x) for x in nodes.split(',')]
                    edge_list = [[int(x) for x in e.split(',')] for e in edges.split(';')]
                    model = GNN()
                    model.load_state_dict(torch.load("models/binary_tree_gnn.pt"))
                    inorder = predict_inorder(model, node_list, edge_list)
                    tree_results = {"nodes": node_list, "edges": edge_list, "inorder": inorder}
                except Exception as e:
                    tree_results = {"error": f"Invalid input: {e}"}
    
    return render_template('index.html', prediction=prediction, sort_steps=sort_steps, rl_steps=rl_steps,
                          stack_queue_results=stack_queue_results, tree_results=tree_results)

if __name__ == '__main__':
    app.run(debug=True)