import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def calculate_psr(sharpe_ratio, bench_sharpe_ratio, T, skewness, kurtosis):
    """
    Function to calculate Probabilistic Sharpe Ratio (PSR)
    """
    numerator = (sharpe_ratio - bench_sharpe_ratio) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio ** 2)
    return norm.cdf(numerator / denominator)

# Sharpe ratios from 0 to 2 with 0.01 increments
sharpe_ratios = np.arange(0, 2, 0.01)

# Example parameters
bench_sharpe_ratio = 1  # Benchmark Sharpe Ratio
T = 5 * 252  # 5 years of trading days
skewness = 0  # assuming returns are normally distributed
kurtosis = 3  # assuming returns are normally distributed

# Calculate PSR for each Sharpe Ratio
psrs = [calculate_psr(sr, bench_sharpe_ratio, T, skewness, kurtosis) for sr in sharpe_ratios]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sharpe_ratios, psrs)
plt.xlabel('Sharpe Ratio')
plt.ylabel('Probabilistic Sharpe Ratio')
plt.title('Probabilistic Sharpe Ratio for Different Sharpe Ratios')
plt.grid(True)

# Add legend on plot
legend_text = (
    f"Benchmark Sharpe Ratio: {bench_sharpe_ratio}\n"
    f"Length of track record: {T} days\n"
    f"Skewness: {skewness}\n"
    f"Kurtosis: {kurtosis}"
)
plt.text(1.2, 0.2, legend_text, fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.show()
