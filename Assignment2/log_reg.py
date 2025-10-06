import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, ensemble
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

EPOCHS = 50
BATCH_SIZES = [64, 256]
TRAIN_SPLITS = [0.8, 0.9]
n_classes = 10
n_features = 784

(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train_full = x_train_full.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train_full = x_train_full.reshape([-1, n_features])
x_test = x_test.reshape([-1, n_features])

def one_hot(y):
    return tf.one_hot(y, depth=n_classes)

def create_model():
    w = tf.Variable(tf.random.normal([n_features, n_classes], stddev=0.01))
    b = tf.Variable(tf.zeros([n_classes]))
    return w, b

def model(X, w, b):
    return tf.matmul(X, w) + b

def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def accuracy(y_pred, y_true):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

def train_one_config(opt_name, train_split, batch_size, device="/CPU:0"):
    print(f"\n{'='*80}")
    print(f"Optimizer: {opt_name}, Train Split: {train_split}, Batch Size: {batch_size}, Device: {device}")
    print(f"{'='*80}")

    if opt_name == "SGD":
        optimizer = tf.optimizers.SGD(learning_rate=0.01)
    elif opt_name == "RMSProp":
        optimizer = tf.optimizers.RMSprop(learning_rate=0.001)
    elif opt_name == "Adam":
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
    else:
        raise ValueError("Unknown optimizer")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=1-train_split, random_state=42
    )

    y_train_oh, y_val_oh, y_test_oh = one_hot(y_train), one_hot(y_val), one_hot(y_test)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh)).shuffle(10000).batch(batch_size)

    w, b = create_model()
    train_accs, val_accs, losses = [], [], []
    epoch_times = []

    with tf.device(device):
        for epoch in range(EPOCHS):
            total_loss = 0
            n_batches = 0
            start_time = time.time()
            for Xb, yb in train_data:
                with tf.GradientTape() as tape:
                    logits = model(Xb, w, b)
                    loss = loss_fn(logits, yb)
                grads = tape.gradient(loss, [w, b])
                optimizer.apply_gradients(zip(grads, [w, b]))
                total_loss += loss.numpy()
                n_batches += 1

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            val_logits = model(x_val, w, b)
            val_acc = accuracy(val_logits, y_val_oh).numpy()
            train_acc = accuracy(model(x_train, w, b), y_train_oh).numpy()
            losses.append(total_loss / n_batches)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Loss={losses[-1]:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                      f"Time={epoch_time:.2f}s")

    test_acc = accuracy(model(x_test, w, b), one_hot(y_test)).numpy()
    avg_time = np.mean(epoch_times)
    overfit_gap = np.mean(train_accs[-5:]) - np.mean(val_accs[-5:])
    print(f"Final Test Accuracy ({opt_name}): {test_acc:.4f}")
    print(f"Avg epoch time: {avg_time:.3f}s | Overfit gap: {overfit_gap:.4f}")

    return {
        "optimizer": opt_name,
        "train_split": train_split,
        "batch_size": batch_size,
        "device": device,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "losses": losses,
        "w": w.numpy(),
        "test_acc": test_acc,
        "avg_time": avg_time,
        "overfit_gap": overfit_gap
    }

results = []
devices = ["/CPU:0"]
if tf.config.list_physical_devices('GPU'):
    devices.append("/GPU:0")

for device in devices:
    for opt_name in ["SGD", "RMSProp", "Adam"]:
        for split in TRAIN_SPLITS:
            for bs in BATCH_SIZES:
                results.append(train_one_config(opt_name, split, bs, device=device))

summary = pd.DataFrame([{
    "Optimizer": r["optimizer"],
    "Train Split": r["train_split"],
    "Batch Size": r["batch_size"],
    "Device": r["device"].replace("/",""),
    "Test Acc": r["test_acc"],
    "Overfit Gap": r["overfit_gap"],
    "Avg Epoch Time (s)": r["avg_time"]
} for r in results])

summary = summary.sort_values(by="Test Acc", ascending=False)
summary.to_csv("results_summary.csv", index=False)
print("\n=== Summary Table ===")
print(summary)

for r in results:
    plt.figure(figsize=(6, 4))
    plt.plot(r["train_accs"], label="Train Acc")
    plt.plot(r["val_accs"], label="Val Acc")
    plt.title(f"{r['optimizer']} | Split={r['train_split']} | Batch={r['batch_size']} | {r['device']}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

print("\n=== Random Forest and SVM Comparison ===")
subset = 10000
rf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(x_train_full[:subset], y_train_full[:subset])
rf_acc = rf.score(x_test, y_test)

svm_clf = svm.LinearSVC(max_iter=1000)
svm_clf.fit(x_train_full[:subset], y_train_full[:subset])
svm_acc = svm_clf.score(x_test, y_test)

print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
print(f"SVM Test Accuracy: {svm_acc:.4f}")

print("\n=== Weight Clustering (t-SNE + KMeans) ===")
best = max(results, key=lambda x: x["test_acc"])
w_best = best["w"]

w_emb = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(w_best.T)
plt.figure(figsize=(6, 6))
plt.scatter(w_emb[:, 0], w_emb[:, 1], c=range(10), cmap="tab10")
for i, txt in enumerate(range(10)):
    plt.annotate(txt, (w_emb[i, 0], w_emb[i, 1]), fontsize=10)
plt.title(f"t-SNE Clusters ({best['optimizer']} best model)")
plt.show()

kmeans = KMeans(n_clusters=10, random_state=42).fit(w_best.T)
plt.figure(figsize=(6, 6))
plt.scatter(w_emb[:, 0], w_emb[:, 1], c=kmeans.labels_, cmap="rainbow")
plt.title("K-Means Clusters of Weight Vectors")
plt.show()

print("\nAll experiments completed successfully.")
print("Results saved to results_summary.csv")
