import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib_fontja

matplotlib_fontja.japanize()


def plot_contour(mu1, mu2, sigma1, sigma2, rho):
    ax.clear()
    x = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)
    y = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 100)
    X, Y = np.meshgrid(x, y)

    # 2変量正規分布の確率密度関数
    Z = (
        1
        / (2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho**2))
        * np.exp(
            -1
            / (2 * (1 - rho**2))
            * (
                ((X - mu1) / sigma1) ** 2
                - 2 * rho * ((X - mu1) / sigma1) * ((Y - mu2) / sigma2)
                + ((Y - mu2) / sigma2) ** 2
            )
        )
    )

    ax.contour(X, Y, Z, levels=10, cmap="viridis")
    ax.set_title("2変量正規分布の等高線")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    canvas.draw()


def update_plot():
    try:
        mu1 = float(mu1_var.get())
        mu2 = float(mu2_var.get())
        sigma1 = float(sigma1_var.get())
        sigma2 = float(sigma2_var.get())
        rho = float(rho_var.get())
        if not -1 < rho < 1:
            raise ValueError("相関係数ρは -1 < ρ < 1 の範囲である必要があります")
        plot_contour(mu1, mu2, sigma1, sigma2, rho)
    except Exception as e:
        print("エラー:", e)


# GUIの構築
root = tk.Tk()
root.title("2変量正規分布 可視化ツール")

frame = ttk.Frame(root)
frame.pack()

# 入力フィールドの作成
ttk.Label(frame, text="μ₁:").grid(row=0, column=0)
mu1_var = tk.StringVar(value="0")
ttk.Entry(frame, textvariable=mu1_var).grid(row=0, column=1)

ttk.Label(frame, text="μ₂:").grid(row=1, column=0)
mu2_var = tk.StringVar(value="0")
ttk.Entry(frame, textvariable=mu2_var).grid(row=1, column=1)

ttk.Label(frame, text="σ₁:").grid(row=2, column=0)
sigma1_var = tk.StringVar(value="1")
ttk.Entry(frame, textvariable=sigma1_var).grid(row=2, column=1)

ttk.Label(frame, text="σ₂:").grid(row=3, column=0)
sigma2_var = tk.StringVar(value="1")
ttk.Entry(frame, textvariable=sigma2_var).grid(row=3, column=1)

ttk.Label(frame, text="ρ:").grid(row=4, column=0)
rho_var = tk.StringVar(value="0")
ttk.Entry(frame, textvariable=rho_var).grid(row=4, column=1)

ttk.Button(frame, text="更新", command=update_plot).grid(row=5, column=0, columnspan=2)

# matplotlibの図をTkinterに埋め込み
fig, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 初期描画
plot_contour(0, 0, 1, 1, 0)

root.mainloop()
