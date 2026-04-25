import matplotlib.pyplot as plt

epochs = list(range(1, 51))

train_loss = [
    4512.4189, 634.8940, 70.1622, 12.0954, 5.5317, 3.6457, 2.6267, 2.1701,
    1.8464, 1.7143, 1.6427, 1.5987, 1.5360, 1.5232, 1.4688, 1.4765, 1.4111,
    1.4278, 1.3835, 1.3494, 1.3529, 1.3367, 1.3132, 1.3180, 1.3185, 1.2571,
    1.2750, 1.2375, 1.2480, 1.2283, 1.2078, 1.2028, 1.1931, 1.1875, 1.1812,
    1.1823, 1.1609, 1.1675, 1.1532, 1.1416, 1.1549, 1.1259, 1.1283, 1.1125,
    1.1146, 1.1055, 1.1093, 1.1128, 1.0832, 1.0891
]

val_loss = [
    1771.4444, 126.6966, 19.4883, 6.4479, 4.0135, 2.8001, 2.8440, 2.1388,
    1.5612, 1.3647, 1.4485, 1.4630, 1.1718, 1.1275, 1.6560, 1.9792, 1.2893,
    1.0709, 1.7449, 1.4905, 1.6139, 1.3874, 1.2578, 1.4011, 1.0912, 1.1754,
    1.0014, 1.0275, 1.2851, 1.3423, 1.1220, 1.1636, 1.3486, 1.0093, 1.3080,
    1.1526, 0.9694, 1.6370, 1.1093, 0.9178, 1.2038, 0.9810, 0.8681, 1.1365,
    1.0817, 0.8441, 1.0151, 1.0826, 0.9424, 1.0142
]

best_epoch = val_loss.index(min(val_loss)) + 1
best_val   = min(val_loss)

zoom_start = 5
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs[zoom_start:], train_loss[zoom_start:], label="Training Loss", color="steelblue", linewidth=1.5)
ax.plot(epochs[zoom_start:], val_loss[zoom_start:],   label="Validation Loss", color="coral", linewidth=1.5)
ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.annotate(f"Best val: {best_val:.4f}\n(epoch {best_epoch})",
            xy=(best_epoch, best_val), xytext=(best_epoch + 2, best_val + 0.2),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training and Validation Loss (Epoch 6–50)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/training_curve.png", dpi=150)
print("Saved to plots/training_curve.png")
plt.show()
