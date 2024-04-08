import matplotlib.pyplot as plt

iterations = []
a_loss = []
h_loss = []
l_loss = []
file_path = r"./models/Beta/loss2.txt"

with open(file_path, 'r') as file:
    for line in file:
        # Filtering lines containing the loss values
        if "finished_process" in line:

            parts = line.split()
            if float(parts[5]) > 3000:
                continue
            # Appending values to their respective lists
            iterations.append(float(parts[3].rstrip(')')))  # Adjusting based on the actual data format observed
            a_loss.append(float(parts[5]))
            # h_loss.append(float(parts[7].rstrip(',')))
            l_loss.append(float(parts[9]))

# Re-plotting with the corrected data
plt.figure(figsize=(14, 8))
plt.plot(iterations, a_loss, label='a_loss', marker='o')
# plt.plot(iterations, h_loss, label='h_loss', marker='x')
plt.plot(iterations, l_loss, label='l_loss', marker='^')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.title('Losses over Iterations')
plt.legend()
plt.grid(True)
plt.show(), iterations[:5], a_loss[:5], h_loss[:5], l_loss[:5]  # Showing a sample of the data to verify correctness
