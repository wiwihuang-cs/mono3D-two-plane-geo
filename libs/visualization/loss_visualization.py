import matplotlib.pyplot as plt

def plot_loss(losses, save_path):
    batches = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")

    plt.plot(batches, losses, "bo")

    plt.savefig(save_path)
    # plt.show()
    plt.close()