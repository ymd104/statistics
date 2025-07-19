import tkinter as tk
from tkinter import ttk
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib_fontja

matplotlib_fontja.japanize()


def update_result(*args):
    try:
        mu = float(mu_var.get())
        sigma = float(sigma_var.get())
        x = float(x_var.get())
        if sigma <= 0:
            result_var.set("σ must be > 0")
            return

        z = (x - mu) / sigma
        p = 1 - norm.cdf(z)
        if p < 1e-8:
            result_var.set("P(X ≥ x) ≈ 0")
            ax.clear()
            canvas.draw()
            return

        conditional_mean = mu + sigma * norm.pdf(z) / p
        result_var.set(f"E[X | X ≥ x] ≈ {conditional_mean:.4f}")

        # --- グラフ描画 ---
        ax.clear()
        x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        y_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
        ax.plot(x_vals, y_vals, label="正規分布", color="blue")

        fill_x = np.linspace(x, mu + 4 * sigma, 300)
        fill_y = norm.pdf(fill_x, loc=mu, scale=sigma)
        ax.fill_between(
            fill_x, fill_y, color="orange", alpha=0.5, label="X >= x の領域"
        )

        ax.axvline(x, color="red", linestyle="--", label=f"x = {x}")

        ax.set_title("条件付き期待値")
        ax.set_xlabel("x")
        ax.set_ylabel("確率密度")
        ax.legend()
        ax.grid(True)

        canvas.draw()

    except Exception as e:
        result_var.set("Invalid input", e)


root = tk.Tk()
root.title("条件付き期待値 E[X | X ≥ x]")

mu_var = tk.StringVar(value="50")
sigma_var = tk.StringVar(value="10")
x_var = tk.StringVar(value="60")
result_var = tk.StringVar()

frame = ttk.Frame(root, padding=[80, 20])
frame.grid()

ttk.Label(frame, text="μ (平均):").grid(column=0, row=0, sticky="e")
ttk.Entry(frame, textvariable=mu_var, width=10).grid(column=1, row=0)

ttk.Label(frame, text="σ (標準偏差):").grid(column=0, row=1, sticky="e")
ttk.Entry(frame, textvariable=sigma_var, width=10).grid(column=1, row=1)

ttk.Label(frame, text="x (しきい値):").grid(column=0, row=2, sticky="e")
ttk.Entry(frame, textvariable=x_var, width=10).grid(column=1, row=2)

ttk.Label(
    frame, textvariable=result_var, font=("Arial", 12, "bold"), padding=[0, 20, 0, 10]
).grid(column=0, row=4, columnspan=2)

# --- グラフ描画エリア ---
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=50)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

for var in (mu_var, sigma_var, x_var):
    var.trace_add("write", update_result)

update_result()
root.mainloop()
