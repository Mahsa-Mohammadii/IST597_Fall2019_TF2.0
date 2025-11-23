import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

tf.random.set_seed(1234)
np.random.seed(1234)

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train[..., None].astype("float32") / 255.
x_test  = x_test[..., None].astype("float32") / 255.

y_train = tf.one_hot(y_train, 10)
y_test  = tf.one_hot(y_test, 10)

batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

def batch_norm_manual(x, gamma, beta, eps=1e-5):
    axes = [0, 1, 2]
    mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    var  = tf.reduce_mean((x - mean)**2, axis=axes, keepdims=True)
    xhat = (x - mean) / tf.sqrt(var + eps)
    return gamma * xhat + beta

class ManualBN(tf.keras.layers.Layer):
    def build(self, input_shape):
        c = input_shape[-1]
        self.gamma = self.add_weight(shape=(1,1,1,c), initializer="ones")
        self.beta  = self.add_weight(shape=(1,1,1,c), initializer="zeros")
    def call(self, x):
        return batch_norm_manual(x, self.gamma, self.beta)

def layer_norm_manual(x, gamma, beta, eps=1e-5):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var  = tf.reduce_mean((x - mean)**2, axis=-1, keepdims=True)
    return gamma * (x - mean) / tf.sqrt(var + eps) + beta

class ManualLN(tf.keras.layers.Layer):
    def build(self, input_shape):
        c = input_shape[-1]
        self.gamma = self.add_weight(shape=(c,), initializer="ones")
        self.beta  = self.add_weight(shape=(c,), initializer="zeros")
    def call(self, x):
        return layer_norm_manual(x, self.gamma, self.beta)

class WeightNormDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.v = self.add_weight(shape=(in_dim, self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.g = self.add_weight(shape=(self.units,),
                                 initializer="ones",
                                 trainable=True)
    def call(self, x):
        v_norm = tf.nn.l2_normalize(self.v, axis=0)
        w = v_norm * self.g
        return tf.matmul(x, w)

def base_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

def bn_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        ManualBN(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

def ln_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        ManualLN(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        WeightNormDense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10)
    ])

def wn_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        WeightNormDense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10)
    ])

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def train_one_epoch(model, optimizer):
    total_loss, total_acc, n = 0, 0, 0
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),
                                              tf.argmax(y,1)), tf.float32))
        total_loss += loss.numpy()
        total_acc += acc.numpy()
        n += 1
    return total_loss/n, total_acc/n

def evaluate(model):
    total_acc, n = 0, 0
    for x, y in test_ds:
        logits = model(x, training=False)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),
                                              tf.argmax(y,1)), tf.float32))
        total_acc += acc.numpy()
        n += 1
    return total_acc/n

models = {
    "none"      : base_model(),
    "batchnorm" : bn_model(),
    "layernorm" : ln_model(),
    "weightnorm": wn_model(),
}

results = {}

for name, model in models.items():
    print(f"\n============================\n Training: {name}\n============================")
    optimizer = tf.keras.optimizers.Adam()
    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(4):
        loss, acc = train_one_epoch(model, optimizer)
        test_acc = evaluate(model)
        train_losses.append(loss)
        train_accs.append(acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch+1} | loss={loss:.4f}, train_acc={acc:.3f}, test_acc={test_acc:.3f}")
    results[name] = (train_losses, train_accs, test_accs)

plt.figure(figsize=(10,5))
for name in results:
    plt.plot(results[name][2], label=name, marker='o')

plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(SAVE_DIR, "accuracy_plot.png"), dpi=200)
plt.show()

print("\nSaved plot →", os.path.join(SAVE_DIR, "accuracy_plot.png"))

