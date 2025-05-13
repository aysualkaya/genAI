import matplotlib.pyplot as plt

def plot_losses(losses, save_path="training_loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", color='blue', linewidth=2)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Loss grafiği '{save_path}' olarak kaydedildi.")

# Test etmek istersen:
if __name__ == "__main__":
    sample_losses = [5.0 / (i+1) for i in range(50)]
    plot_losses(sample_losses)
 