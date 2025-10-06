import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def hybrid_loss(y_true, y_pred, alpha=0.5):
    return alpha * l1_loss(y_true, y_pred) + (1 - alpha) * mse_loss(y_true, y_pred)

def generate_noisy_data(noise_type="gaussian", num_examples=500, stddev=1.0, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    X = tf.random.normal([num_examples])
    y_true = 3 * X + 2
    if noise_type == "gaussian":
        noise = tf.random.normal([num_examples], stddev=stddev)
    elif noise_type == "uniform":
        noise = tf.random.uniform([num_examples], minval=-stddev, maxval=stddev)
    elif noise_type == "laplace":
        noise_np = np.random.laplace(loc=0.0, scale=stddev, size=num_examples)
        noise = tf.convert_to_tensor(noise_np, dtype=tf.float32)
    elif noise_type == "salt_pepper":
        noise_np = np.zeros(num_examples)
        idx_salt = np.random.choice(num_examples, int(0.05 * num_examples), replace=False)
        idx_pepper = np.random.choice(num_examples, int(0.05 * num_examples), replace=False)
        noise_np[idx_salt] = stddev
        noise_np[idx_pepper] = -stddev
        noise = tf.convert_to_tensor(noise_np, dtype=tf.float32)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    y = y_true + noise
    return X, y

def train_model(
    loss_fn,
    lr=0.05,
    steps=1000,
    noise_std=1.0,
    noise_type="gaussian",
    patience_window=None,
    init_mean=0.0,
    init_std=0.1,
    add_weight_noise=False,
    add_lr_noise=False,
    seed_val=None,
    device="/CPU:0",
):
    if seed_val is not None:
        tf.random.set_seed(seed_val)
        np.random.seed(seed_val)
    with tf.device(device):
        X, y = generate_noisy_data(noise_type, 500, noise_std, seed_val)
        W = tf.Variable(tf.random.normal([], mean=init_mean, stddev=init_std))
        b = tf.Variable(tf.random.normal([], mean=init_mean, stddev=init_std))
        losses = []
        lr_current = lr
        best_loss = np.inf
        last_improvement = 0
        start = time.time()
        for step in range(steps):
            if add_weight_noise and step % 100 == 0 and step > 0:
                W.assign_add(tf.random.normal([], stddev=0.05))
                b.assign_add(tf.random.normal([], stddev=0.05))
            lr_effective = lr_current
            if add_lr_noise and step % 100 == 0 and step > 0:
                lr_effective = lr_current * (1.0 + np.random.uniform(-0.3, 0.3))
            with tf.GradientTape() as tape:
                y_pred = W * X + b
                loss = loss_fn(y, y_pred)
            dW, db = tape.gradient(loss, [W, b])
            W.assign_sub(lr_effective * dW)
            b.assign_sub(lr_effective * db)
            losses.append(loss.numpy())
            if patience_window is not None:
                if loss.numpy() < best_loss - 1e-5:
                    best_loss = loss.numpy()
                    last_improvement = 0
                else:
                    last_improvement += 1
                    if last_improvement >= patience_window:
                        lr_current *= 0.5
                        last_improvement = 0
                        print(f"Step {step}: reducing learning rate to {lr_current:.5f}")
        duration = time.time() - start
    return W.numpy(), b.numpy(), losses, duration

if __name__ == "__main__":
    seed_decimal = sum([ord(c) for c in "Mahsa"])
    print(f"Unique experiment seed: {seed_decimal}\n")
    print(" Loss Function Comparison ")
    loss_results = {}
    for name, fn in {"L1": l1_loss, "L2": mse_loss, "Hybrid": lambda y, y_pred: hybrid_loss(y, y_pred)}.items():
        W, b, losses, t = train_model(fn, lr=0.05, seed_val=seed_decimal)
        loss_results[name] = losses
        print(f"{name:7s} -> W={W:.3f}, b={b:.3f}, final loss={losses[-1]:.4f}, time={t:.2f}s")
    plt.figure(figsize=(7, 4))
    for name, losses in loss_results.items():
        plt.plot(losses, label=name)
    plt.title("Loss Function Comparison")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\n Learning Rate Comparison ")
    lr_results = {}
    for lr in [0.001, 0.05, 0.2]:
        W, b, losses, t = train_model(mse_loss, lr=lr, seed_val=seed_decimal)
        lr_results[lr] = losses
        print(f"LR={lr}: W={W:.3f}, b={b:.3f}, final loss={losses[-1]:.4f}, time={t:.2f}s")
    plt.figure(figsize=(7, 4))
    for lr, losses in lr_results.items():
        plt.plot(losses, label=f"α={lr}")
    plt.title("Effect of Learning Rate on Convergence")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nInitialization Comparison")
    init_results = {}
    for label, (mean, std) in {"Near-zero": (0.0, 0.1), "Large positive": (5.0, 0.1), "Large negative": (-5.0, 0.1)}.items():
        W, b, losses, t = train_model(mse_loss, init_mean=mean, init_std=std, seed_val=seed_decimal)
        init_results[label] = losses
        print(f"{label:15s}: W={W:.3f}, b={b:.3f}, final loss={losses[-1]:.4f}, time={t:.2f}s")
    plt.figure(figsize=(7, 4))
    for label, losses in init_results.items():
        plt.plot(losses, label=label)
    plt.title("Effect of Initialization on Convergence")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nPatience Scheduling Experiment")
    W, b, losses, t = train_model(hybrid_loss, lr=0.05, steps=1500, seed_val=seed_decimal, patience_window=100)
    print(f"Patience schedule -> W={W:.3f}, b={b:.3f}, final loss={losses[-1]:.4f}, time={t:.2f}s")
    print("\nData Noise Experiments")
    noise_results = {}
    for nl in [0.2, 1.0, 2.0]:
        W, b, losses, t = train_model(mse_loss, noise_std=nl, steps=2000, seed_val=seed_decimal)
        noise_results[nl] = losses
        print(f"Noise std={nl}: final loss={losses[-1]:.4f}")
    plt.figure(figsize=(7, 4))
    for nl, losses in noise_results.items():
        plt.plot(losses, label=f"Noise std={nl}")
    plt.title("Effect of Gaussian Data Noise on Convergence")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nVarious Noise Type Experiments")
    noise_types = ["gaussian", "uniform", "laplace", "salt_pepper"]
    type_results = {}
    for nt in noise_types:
        W, b, losses, t = train_model(mse_loss, noise_type=nt, steps=1500, seed_val=seed_decimal)
        type_results[nt] = losses
        print(f"{nt:12s} -> W={W:.3f}, b={b:.3f}, final loss={losses[-1]:.4f}")
    plt.figure(figsize=(7, 4))
    for nt, losses in type_results.items():
        plt.plot(losses, label=nt)
    plt.title("Comparison of Different Noise Types")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nAdding weight noise:")
    Ww, bw, losses_w, _ = train_model(hybrid_loss, add_weight_noise=True, seed_val=seed_decimal)
    print("\nAdding learning rate noise:")
    Wlr, blr, losses_lr, _ = train_model(hybrid_loss, add_lr_noise=True, seed_val=seed_decimal)
    plt.figure(figsize=(7, 4))
    plt.plot(losses_w, label="Weight Noise")
    plt.plot(losses_lr, label="Learning Rate Noise")
    plt.title("Effect of Parameter and LR Noise")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nGPU vs CPU Timing")
    def timing_experiment(device="/CPU:0", steps=2000):
        start = time.time()
        W, b, _, _ = train_model(mse_loss, steps=steps, device=device, seed_val=seed_decimal)
        total = time.time() - start
        per_step = total / steps
        print(f"{device}: total={total:.3f}s, per step={per_step:.6f}s, W={W:.3f}, b={b:.3f}")
        return total, per_step
    t_cpu_total, t_cpu_step = timing_experiment("/CPU:0")
    if tf.config.list_physical_devices("GPU"):
        t_gpu_total, t_gpu_step = timing_experiment("/GPU:0")
        plt.figure(figsize=(4, 4))
        plt.bar(["CPU", "GPU"], [t_cpu_total, t_gpu_total], color=["gray", "green"])
        plt.ylabel("Training Time (s)")
        plt.title("CPU vs GPU Performance")
        plt.tight_layout(); plt.show()
        print(f"Speed-up = {t_cpu_total / t_gpu_total:.2f}x")
    else:
        print("No GPU detected on this system.")

