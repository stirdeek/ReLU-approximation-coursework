# -*- coding: utf-8 -*-
"""
Полный эксперимент с усреднением (n_runs=5), но графики в исходном стиле.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------- 1. Тестовые функции -------------------
def f_sin(x): return np.sin(2 * np.pi * x)
def f_abs(x): return np.abs(x - 0.5)
def f_square(x): return x ** 2
def f_mixed(x): return np.sin(2 * np.pi * x) + 0.5 * np.abs(x - 0.5)

functions = {
    'sin': f_sin,
    'abs': f_abs,
    'square': f_square,
    'mixed': f_mixed
}

# ------------------- 2. Генерация данных -------------------
def generate_data(func, n_train=200, n_test=1000, seed=42):
    np.random.seed(seed)
    x_train = np.random.uniform(0, 1, n_train)
    y_train = func(x_train)
    x_test = np.linspace(0, 1, n_test)
    y_test = func(x_test)
    return (x_train.astype(np.float32), y_train.astype(np.float32),
            x_test.astype(np.float32), y_test.astype(np.float32))

# ------------------- 3. Модели -------------------
class ReLUNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        prev = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SigmoidNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        prev = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Sigmoid())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------- 4. Обучение (без verbose) -------------------
def train_model(model, x_train, y_train, x_test, y_test,
                epochs=1500, lr=0.005, weight_decay=1e-5, patience=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40)
    
    x_train_t = torch.from_numpy(x_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1)
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_test_t), y_test_t)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_t).cpu().numpy().flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    linf = np.max(np.abs(y_test - y_pred))
    return mse, linf

# ------------------- 5. Усреднение (только средние, без вывода std на графики) -------------------
def evaluate_averaged(func, hidden_sizes, net_class, n_runs=5):
    x_train, y_train, x_test, y_test = generate_data(func, seed=42)
    mse_list = []
    linf_list = []
    for run in range(n_runs):
        set_seed(42 + run)
        model = net_class(hidden_sizes)
        mse, linf = train_model(model, x_train, y_train, x_test, y_test)
        mse_list.append(mse)
        linf_list.append(linf)
    return np.mean(mse_list), np.mean(linf_list)

# ------------------- 6. Сбор всех результатов с усреднением -------------------
def run_experiments_averaged(n_runs=5):
    architectures = {
        'Shallow_5': [5], 'Shallow_10': [10], 'Shallow_20': [20], 'Shallow_40': [40],
        'Deep_2x5': [5,5], 'Deep_3x5': [5,5,5], 'Deep_2x10': [10,10], 'Deep_4x4': [4,4,4,4],
    }
    results = {}
    for func_name, func in functions.items():
        print(f"\n{'='*60}\n{func_name}\n{'='*60}")
        results[func_name] = {}
        for arch_name, hidden in architectures.items():
            mse_avg, linf_avg = evaluate_averaged(func, hidden, ReLUNet, n_runs=n_runs)
            model = ReLUNet(hidden)
            params = count_params(model)
            results[func_name][arch_name] = {
                'mse': mse_avg, 'linf': linf_avg, 'params': params, 'hidden': hidden
            }
            print(f"{arch_name:12s} params={params:3d} MSE={mse_avg:.3e} L∞={linf_avg:.3e}")
    return results

# ------------------- 7. Графики в исходном стиле (как в первом коде) -------------------
def plot_results_original_style(results):
    # ---- График 1: все функции, scatter с подписями (как в plot_results) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (func_name, arch_results) in enumerate(results.items()):
        ax = axes[idx]
        arch_names = list(arch_results.keys())
        params = [arch_results[a]['params'] for a in arch_names]
        mse = [arch_results[a]['mse'] for a in arch_names]
        linf = [arch_results[a]['linf'] for a in arch_names]
        ax.scatter(params, mse, label='MSE', marker='o', color='blue')
        ax.scatter(params, linf, label='L∞ error', marker='s', color='red')
        for i, name in enumerate(arch_names):
            ax.annotate(name, (params[i], mse[i]), xytext=(5,5), textcoords='offset points', fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of parameters')
        ax.set_ylabel('Error')
        ax.set_title(f'Function: {func_name}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Figure_1.png', dpi=150)
    plt.show()

    # ---- График 2: ошибка vs ширина (однослойные ReLU на mixed) ----
    mixed = results['mixed']
    shallow = {n:d for n,d in mixed.items() if len(d['hidden'])==1}
    # сортируем по ширине
    items = sorted(shallow.items(), key=lambda x: x[1]['hidden'][0])
    widths = [d[1]['hidden'][0] for d in items]
    mse_vals = [d[1]['mse'] for d in items]
    linf_vals = [d[1]['linf'] for d in items]
    plt.figure(figsize=(10,6))
    plt.plot(widths, mse_vals, 'o-', label='MSE', color='blue', linewidth=2, markersize=8)
    plt.plot(widths, linf_vals, 's-', label='L∞', color='red', linewidth=2, markersize=8)
    plt.xlabel('Width (number of hidden units)')
    plt.ylabel('Error')
    plt.title('Shallow ReLU networks: error vs width (function: mixed)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figure_2.png', dpi=150)
    plt.show()

    # ---- График 3: ошибка vs глубина (фиксированная ширина 5) ----
    depth_names = ['Shallow_5', 'Deep_2x5', 'Deep_3x5']
    depths = []
    mse_depth = []
    for name in depth_names:
        data = mixed[name]
        depths.append(len(data['hidden']))
        mse_depth.append(data['mse'])
    plt.figure(figsize=(8,6))
    plt.plot(depths, mse_depth, 'o-', color='green', linewidth=2, markersize=10)
    plt.xlabel('Depth (number of hidden layers)')
    plt.ylabel('MSE')
    plt.title('Effect of depth (fixed width = 5 neurons per layer)')
    plt.xticks(depths)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figure_3.png', dpi=150)
    plt.show()

    # ---- График 4: сравнение лучших моделей (визуализация предсказаний) ----
    # Находим лучшую широкую и глубокую ReLU
    best_shallow_name = min(shallow.items(), key=lambda x: x[1]['mse'])[0]
    best_shallow_data = mixed[best_shallow_name]
    deep_items = {n:d for n,d in mixed.items() if len(d['hidden'])>1}
    best_deep_name = min(deep_items.items(), key=lambda x: x[1]['mse'])[0]
    best_deep_data = mixed[best_deep_name]
    # Подбираем лучшую сигмоиду (отдельно)
    sigmoid_candidates = [([10],'Sigmoid_10'), ([20],'Sigmoid_20'), ([5,5],'Sigmoid_2x5'), ([10,10],'Sigmoid_2x10')]
    best_sigmoid_mse = float('inf')
    best_sigmoid_hidden = None
    best_sigmoid_name = None
    for hidden, name in sigmoid_candidates:
        mse_avg, _ = evaluate_averaged(f_mixed, hidden, SigmoidNet, n_runs=5)
        if mse_avg < best_sigmoid_mse:
            best_sigmoid_mse = mse_avg
            best_sigmoid_hidden = hidden
            best_sigmoid_name = name
    print(f"\nЛучшая сигмоида: {best_sigmoid_name} (MSE={best_sigmoid_mse:.3e})")
    
    # Визуализация предсказаний (один дополнительный запуск для картинки)
    x_train, y_train, x_test, y_test = generate_data(f_mixed, seed=123)
    x_plot = np.linspace(0,1,1000)
    x_plot_t = torch.from_numpy(x_plot.reshape(-1,1).astype(np.float32))
    
    model_shallow = ReLUNet(best_shallow_data['hidden'])
    train_model(model_shallow, x_train, y_train, x_test, y_test, epochs=500)
    with torch.no_grad():
        y_shallow = model_shallow(x_plot_t).numpy().flatten()
    
    model_deep = ReLUNet(best_deep_data['hidden'])
    train_model(model_deep, x_train, y_train, x_test, y_test, epochs=500)
    with torch.no_grad():
        y_deep = model_deep(x_plot_t).numpy().flatten()
    
    model_sigmoid = SigmoidNet(best_sigmoid_hidden)
    train_model(model_sigmoid, x_train, y_train, x_test, y_test, epochs=500)
    with torch.no_grad():
        y_sigmoid = model_sigmoid(x_plot_t).numpy().flatten()
    
    plt.figure(figsize=(12,6))
    plt.plot(x_plot, f_mixed(x_plot), 'k-', linewidth=2, label='True function')
    plt.plot(x_plot, y_shallow, '--', label=f'Shallow ReLU: {best_shallow_name}\nMSE={best_shallow_data["mse"]:.2e}', color='blue')
    plt.plot(x_plot, y_deep, '--', label=f'Deep ReLU: {best_deep_name}\nMSE={best_deep_data["mse"]:.2e}', color='red')
    plt.plot(x_plot, y_sigmoid, '--', label=f'Sigmoid: {best_sigmoid_name}\nMSE={best_sigmoid_mse:.2e}', color='orange')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('Comparison: best shallow vs deep ReLU vs sigmoid')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure_4.png', dpi=150)
    plt.show()

# ------------------- 8. Запуск -------------------
if __name__ == "__main__":
    print("Запуск эксперимента с усреднением (n_runs=5)...")
    res = run_experiments_averaged(n_runs=5)
    plot_results_original_style(res)
    print("\nГрафики сохранены: Figure_1.png, Figure_2.png, Figure_3.png, Figure_4.png")