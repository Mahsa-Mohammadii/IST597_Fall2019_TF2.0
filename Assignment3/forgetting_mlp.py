# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_tasks = 10
epochs_first_task = 50
epochs_per_task = 20
batch_size = 32
learning_rate = 0.001
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

permutations = [np.random.permutation(784) for _ in range(num_tasks)]

def get_task_data(task_id):
    perm = permutations[task_id]
    x_train_t = x_train[:, perm]
    x_test_t = x_test[:, perm]
    return (x_train_t, y_train), (x_test_t, y_test)

def build_mlp(depth=2, hidden_units=256, dropout_rate=0.0,
              loss_fn="categorical_crossentropy", optimizer="adam"):
    layers = [tf.keras.layers.Input(shape=(784,))]
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

def train_forgetting_experiment(model, num_tasks=num_tasks,
                                epochs_first=50, epochs_next=20):
    R = np.zeros((num_tasks, num_tasks))
    for t in range(num_tasks):
        print(f"\nTraining on Task {t+1}/{num_tasks}")
        (xtr, ytr), (xte, yte) = get_task_data(t)
        ytr = tf.keras.utils.to_categorical(ytr, 10)
        yte = tf.keras.utils.to_categorical(yte, 10)
        epochs = epochs_first if t == 0 else epochs_next
        model.fit(xtr, ytr, batch_size=batch_size, epochs=epochs, verbose=0)
        for j in range(t+1):
            (_, _), (xte2, yte2) = get_task_data(j)
            yte2 = tf.keras.utils.to_categorical(yte2, 10)
            loss, acc = model.evaluate(xte2, yte2, verbose=0)
            R[t, j] = acc
    return R

def compute_metrics(R):
    T = R.shape[0]
    ACC = np.mean(R[-1, :])
    BWT = np.mean([R[-1, i] - R[i, i] for i in range(T-1)])
    return ACC, BWT

if __name__ == "__main__":
    model = build_mlp(depth=2, dropout_rate=0.3,
                      optimizer=tf.keras.optimizers.Adam(learning_rate))
    R = train_forgetting_experiment(model)
    ACC, BWT = compute_metrics(R)
    print(f"Final ACC: {ACC:.4f}, BWT: {BWT:.4f}")

def custom_loss(name):
    if name == "L1":
        return lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred))
    elif name == "L2":
        return lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))
    elif name == "L1+L2":
        return lambda y_true, y_pred: (
            tf.reduce_mean(tf.abs(y_true - y_pred)) +
            tf.reduce_mean(tf.square(y_true - y_pred))
        )
    elif name == "NLL":
        return "categorical_crossentropy"
    else:
        raise ValueError("Unknown loss name")

def run_experiment(depth, dropout, loss_name, optimizer_name):
    print(f"\nRunning: Depth={depth}, Dropout={dropout}, Loss={loss_name}, Optimizer={optimizer_name}")
    optimizer_dict = {
        "Adam": tf.keras.optimizers.Adam(learning_rate),
        "SGD": tf.keras.optimizers.SGD(learning_rate),
        "RMSProp": tf.keras.optimizers.RMSprop(learning_rate)
    }
    model = build_mlp(
        depth=depth,
        dropout_rate=dropout,
        loss_fn=custom_loss(loss_name),
        optimizer=optimizer_dict[optimizer_name]
    )
    R = train_forgetting_experiment(model)
    ACC, BWT = compute_metrics(R)
    print(f"Result: ACC={ACC:.4f}, BWT={BWT:.4f}")
    return ACC, BWT, R

experiments = [
    {"depth": 2, "dropout": 0.0, "loss": "NLL", "opt": "Adam"},
    {"depth": 3, "dropout": 0.0, "loss": "NLL", "opt": "Adam"},
    {"depth": 4, "dropout": 0.0, "loss": "NLL", "opt": "Adam"},
    {"depth": 2, "dropout": 0.0, "loss": "L1", "opt": "Adam"},
    {"depth": 2, "dropout": 0.0, "loss": "L2", "opt": "Adam"},
    {"depth": 2, "dropout": 0.0, "loss": "L1+L2", "opt": "Adam"},
    {"depth": 2, "dropout": 0.3, "loss": "NLL", "opt": "Adam"},
    {"depth": 2, "dropout": 0.0, "loss": "NLL", "opt": "SGD"},
    {"depth": 2, "dropout": 0.0, "loss": "NLL", "opt": "RMSProp"}
]

results = []

for exp in experiments:
    ACC, BWT, R = run_experiment(exp["depth"], exp["dropout"], exp["loss"], exp["opt"])
    results.append({**exp, "ACC": ACC, "BWT": BWT})

print("\nSummary of 9 Experiments")
for res in results:
    print(f"Depth={res['depth']} Dropout={res['dropout']} Loss={res['loss']} "
          f"Opt={res['opt']} → ACC={res['ACC']:.4f}, BWT={res['BWT']:.4f}")

chosen = results[0]
_, _, R = run_experiment(chosen["depth"], chosen["dropout"], chosen["loss"], chosen["opt"])

plt.figure(figsize=(7, 4))
for i in range(R.shape[1]):
    plt.plot(range(1, R.shape[0]+1), R[:, i], marker='o', label=f"Task {i+1}")
plt.title("Validation Accuracy per Task (Forgetting Over Time)")
plt.xlabel("After Training on Task")
plt.ylabel("Accuracy")
plt.xticks(range(1, R.shape[0]+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Validation_Accuracy_Drop_Baseline.png", bbox_inches="tight")
plt.show()

