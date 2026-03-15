import numpy as np
import matplotlib.pyplot as plt

# --- Detection times ---
availability_times = [22.5, 22.5]
normal_times       = [22.5, 20.9]
security_times     = [21.2, 20.5]

modes  = ["Availability", "Normal", "Security"]
colors = ["green", "orange", "red"]

time_data  = [availability_times, normal_times, security_times]
time_means = [float(np.mean(x)) for x in time_data]

interrupt_start_s = 20.0

plt.figure()

x = np.arange(len(modes))
bars = plt.bar(x, time_means, color=colors)

plt.ylabel("Interruption detection time (s)")
plt.title("Interruption detection vs mode")
plt.xticks(x, modes, rotation=20)

# ---- START AXIS FROM 20 ----
plt.ylim(interrupt_start_s, max(time_means) + 1)

# ---- Annotate bars ----
for bar, val in zip(bars, time_means):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.15,
        f"{val:.2f}s",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()