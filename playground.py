import matplotlib.pyplot as plt
from train_options import opt

iterations = []
a_loss = []
h_loss = []
l_loss = []
file_path = rf"./models/{opt.MODEL}/loss.txt"

with open(file_path, 'r') as file:
    for line in file:
        if "finished_process" in line:
            parts = line.split()
            # Appending values to their respective lists
            a_loss.append(float(parts[7]))
            h_loss.append(float(parts[9].rstrip(',')))
            l_loss.append(float(parts[11]))

# Re-plotting with the corrected data
iterations = [*range(0, len(a_loss)*opt.PRINT_RESULTS, opt.PRINT_RESULTS)]

plt.figure(figsize=(14, 8))
plt.plot(iterations, a_loss, label='a_loss', marker='o')
# plt.plot(iterations, h_loss, label='h_loss', marker='x')
# plt.plot(iterations, l_loss, label='l_loss', marker='^')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.title('Losses over Iterations')
plt.legend()
plt.grid(True)
plt.show()
