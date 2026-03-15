import numpy as np
import matplotlib.pyplot as plt

# --- Detection times ---
baseline_time      = [35]
availability_times = [35.2, 35.9]
normal_times       = [32.6, 32]
security_times     = [30.9, 31.7]

modes  = ["Baseline (Full/5s)", "Availability", "Normal", "Security"]
colors = ["gray", "green", "orange", "red"]

injection_start_s = 30.0

time_data = [
    baseline_time,
    availability_times,
    normal_times,
    security_times
]

time_means = [float(np.mean(x)) for x in time_data]

plt.figure()

x = np.arange(len(modes))
bars = plt.bar(x, time_means, color=colors)

plt.ylabel("Detection time (s)")
plt.title("Injection detection vs mode")
plt.xticks(x, modes, rotation=20)

# ---- START AXIS FROM 30 ----
plt.ylim(injection_start_s, max(time_means) + 1)

# ---- Annotate bars ----
for bar, val in zip(bars, time_means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.15,
        f"{val:.2f}s",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

modes = ["Baseline (Full/5s)", "Availability", "Normal", "Security"]
colors = ["gray", "green", "orange", "red"]

# Run 1
coverage_run1 = [7.0, 3.56, 6, 7.6]

# Run 2 (βάλε τις πραγματικές τιμές σου εδώ)
coverage_run2 = [7.0, 5.67, 5.02, 8.8]

# Μετατροπή σε numpy array
cov_array = np.array([coverage_run1, coverage_run2])

# Μέσος όρος
coverage_mean = np.mean(cov_array, axis=0)

plt.figure()

x = np.arange(len(modes))
bars = plt.bar(x, coverage_mean, color=colors,)

plt.ylabel("Memory coverage ")
plt.title("Memory coverage vs mode ")
plt.xticks(x, modes, rotation=20)
plt.ylim(0, 10)

for bar, cov in zip(bars, coverage_mean):
    plt.text(bar.get_x() + bar.get_width()/2,
             cov + 0.03,
             f"{cov:.2f}",
             ha="center", va="bottom")

plt.tight_layout()
plt.show()