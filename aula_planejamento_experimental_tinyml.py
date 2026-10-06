# -*- coding: utf-8 -*-
"""Interface gráfica para análise experimental TinyML com Tkinter.

Executa análise de Friedman, Nemenyi, Cliff's Delta e Bootstrap usando planilhas CSV.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

ESTRATEGIAS = ["Baseline", "Quantizacao", "Poda", "HW_NAS"]


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    maior = 0
    menor = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                maior += 1
            elif xi < yi:
                menor += 1
    total = len(x) * len(y)
    if total == 0:
        return 0.0
    return (maior - menor) / total


def interpretar_delta(delta):
    a = abs(float(delta))
    if a < 0.147:
        magnitude = "negligível"
    elif a < 0.33:
        magnitude = "pequena"
    elif a < 0.474:
        magnitude = "média"
    else:
        magnitude = "grande"

    direcao = "positiva" if delta > 0 else "negativa" if delta < 0 else "nula"
    return magnitude, direcao


def save_figure(filename, plot_builder):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_builder(ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def analyze_friedman(df):
    missing = [col for col in ESTRATEGIAS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no CSV de Friedman/Nemenyi: {missing}")

    resultado = stats.friedmanchisquare(*[df[c] for c in ESTRATEGIAS])
    ranks = df[ESTRATEGIAS].rank(axis=1, method="average")
    postos = ranks.mean().sort_values(ascending=False)

    posthoc = sp.posthoc_nemenyi_friedman(df[ESTRATEGIAS])
    comparacoes = []
    for i in range(len(ESTRATEGIAS)):
        for j in range(i + 1, len(ESTRATEGIAS)):
            p = float(posthoc.iloc[i, j])
            if p < 0.05:
                comparacoes.append(f"{ESTRATEGIAS[i]} × {ESTRATEGIAS[j]}: p = {p:.6f}")

    save_figure(
        "friedman_boxplot.png",
        lambda ax: (
            ax.boxplot([df[c] for c in ESTRATEGIAS]),
            ax.set_xticks(range(1, 5)),
            ax.set_xticklabels(["Baseline", "Quantização", "Poda", "HW-NAS"]),
            ax.set_ylabel("F1-score (%)"),
            ax.set_title("F1-score por estratégia"),
        ),
    )

    return {
        "resultado": resultado,
        "postos": postos,
        "posthoc": posthoc,
        "comparacoes": comparacoes,
        "plot": "friedman_boxplot.png",
    }


def analyze_cliff(df):
    required = ["Baseline", "Quantizacao"]
    if any(col not in df.columns for col in required):
        raise ValueError("CSV de Cliff's Delta deve conter Baseline e Quantizacao.")

    delta = cliffs_delta(df["Quantizacao"], df["Baseline"])
    magnitude, direcao = interpretar_delta(delta)

    save_figure(
        "cliff_boxplot.png",
        lambda ax: (
            ax.boxplot([df["Baseline"], df["Quantizacao"]]),
            ax.set_xticks([1, 2]),
            ax.set_xticklabels(["Baseline", "Quantização"]),
            ax.set_ylabel("F1-score (%)"),
            ax.set_title("Baseline × Quantização"),
        ),
    )

    return {
        "delta": delta,
        "magnitude": magnitude,
        "direcao": direcao,
        "plot": "cliff_boxplot.png",
    }


def analyze_bootstrap(df):
    if "Diferenca_ms" not in df.columns:
        raise ValueError("CSV de Bootstrap deve conter a coluna Diferenca_ms.")

    dif = df["Diferenca_ms"].to_numpy(dtype=float)
    rng = np.random.default_rng(2026)
    B = 10000
    medias_bootstrap = np.empty(B)
    for i in range(B):
        amostra = rng.choice(dif, size=len(dif), replace=True)
        medias_bootstrap[i] = np.mean(amostra)

    media_bootstrap = float(np.mean(medias_bootstrap))
    ic95 = np.percentile(medias_bootstrap, [2.5, 97.5])

    save_figure(
        "bootstrap_hist.png",
        lambda ax: (
            ax.hist(medias_bootstrap, bins=35, color="steelblue", edgecolor="black"),
            ax.axvline(media_bootstrap, linestyle="--", color="red", label=f"Média = {media_bootstrap:.2f} ms"),
            ax.axvline(ic95[0], linestyle=":", color="darkgreen", label=f"IC95% inferior = {ic95[0]:.2f}"),
            ax.axvline(ic95[1], linestyle=":", color="darkgreen", label=f"IC95% superior = {ic95[1]:.2f}"),
            ax.set_xlabel("Diferença média de latência (ms)"),
            ax.set_ylabel("Frequência"),
            ax.set_title("Distribuição Bootstrap — 10.000 reamostragens"),
            ax.legend(),
        ),
    )

    return {
        "dif": dif,
        "media": media_bootstrap,
        "ic95": ic95,
        "plot": "bootstrap_hist.png",
    }


def analyze_integrado(df):
    required = ["Hardware", "Estrategia", "F1_score", "Latencia_ms", "Memoria_KB", "Energia_mJ"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV integrado está incompleto. Faltam: {missing}")

    resumo = (
        df.groupby(["Hardware", "Estrategia"])
        [["F1_score", "Latencia_ms", "Memoria_KB", "Energia_mJ"]]
        .mean()
        .round(2)
    )
    return {"resumo": resumo}


def run_analysis(friedman_path=None, cliff_path=None, bootstrap_path=None, integrado_path=None):
    output = []

    if friedman_path:
        df_friedman = pd.read_csv(friedman_path)
        friedman = analyze_friedman(df_friedman)
        output.append("=== FRIEDMAN ===")
        output.append(f"Estatística: {friedman['resultado'].statistic:.6f}")
        output.append(f"p-valor: {friedman['resultado'].pvalue:.6f}")
        output.append("Conclusão: " + ("rejeitamos H0. Existe diferença global entre as estratégias." if friedman['resultado'].pvalue < 0.05 else "não rejeitamos H0."))
        output.append("\nPostos médios:")
        output.append(str(friedman['postos'].to_frame('Posto_medio')))
        output.append("\nComparações significativas (p < 0.05):")
        if friedman['comparacoes']:
            output.extend(friedman['comparacoes'])
        else:
            output.append("Nenhuma comparação significativa encontrada.")

    if cliff_path:
        df_cliff = pd.read_csv(cliff_path)
        cliff = analyze_cliff(df_cliff)
        output.append("\n=== CLIFF'S DELTA ===")
        output.append(f"Delta = {cliff['delta']:.3f}")
        output.append(f"Magnitude: {cliff['magnitude']}")
        output.append(f"Direção: {cliff['direcao']}")

    if bootstrap_path:
        df_boot = pd.read_csv(bootstrap_path)
        boot = analyze_bootstrap(df_boot)
        output.append("\n=== BOOTSTRAP ===")
        output.append(f"Diferença média observada: {boot['dif'].mean():.2f} ms")
        output.append(f"Estimativa média: {boot['media']:.2f} ms")
        output.append(f"IC95%: [{boot['ic95'][0]:.2f}; {boot['ic95'][1]:.2f}] ms")

    if integrado_path:
        df_integrado = pd.read_csv(integrado_path)
        integrado = analyze_integrado(df_integrado)
        output.append("\n=== DATASET INTEGRADO ===")
        output.append(str(integrado['resumo']))

    return "\n".join(output)


class TinyMLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Planejamento Experimental TinyML")
        self.geometry("980x720")
        self.minsize(900, 650)

        self.friedman_path = tk.StringVar()
        self.cliff_path = tk.StringVar()
        self.bootstrap_path = tk.StringVar()
        self.integrado_path = tk.StringVar()

        self._configure_default_paths()

        self._build_ui()

    def _configure_default_paths(self):
        project_dir = Path(__file__).resolve().parent
        auto_files = {
            "friedman": ["friedman.csv", "friedman_nemenyi.csv", "df_friedman.csv"],
            "cliff": ["cliff.csv", "cliffs_delta.csv", "df_cliff.csv"],
            "bootstrap": ["bootstrap.csv", "df_boot.csv", "latencia_bootstrap.csv"],
            "integrado": ["integrado.csv", "dataset_integrado.csv", "df_integrado.csv"],
        }

        for key, names in auto_files.items():
            for name in names:
                candidate = project_dir / name
                if candidate.exists():
                    setattr(self, f"_{key}_default", str(candidate))
                    break
            else:
                setattr(self, f"_{key}_default", "")

        self.friedman_path.set(getattr(self, "_friedman_default", ""))
        self.cliff_path.set(getattr(self, "_cliff_default", ""))
        self.bootstrap_path.set(getattr(self, "_bootstrap_default", ""))
        self.integrado_path.set(getattr(self, "_integrado_default", ""))

    def _build_ui(self):
        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Planejamento Experimental em TinyML", font=("Arial", 16, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        self.fields = []
        for label, var, key in [
            ("CSV Friedman/Nemenyi", self.friedman_path, "friedman"),
            ("CSV Cliff's Delta", self.cliff_path, "cliff"),
            ("CSV Bootstrap", self.bootstrap_path, "bootstrap"),
            ("CSV Integrado", self.integrado_path, "integrado"),
        ]:
            row = ttk.Frame(main)
            row.pack(fill="x", pady=6)

            ttk.Label(row, text=label, width=18, anchor="w").pack(side="left")
            entry = ttk.Entry(row, textvariable=var, width=80)
            entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
            btn = ttk.Button(row, text="Procurar", command=lambda v=var, k=key: self._browse_file(v, k))
            btn.pack(side="left")
            self.fields.append((label, var))

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(14, 8))
        ttk.Button(actions, text="Executar análise", command=self.run_gui_analysis).pack(side="left")
        ttk.Button(actions, text="Limpar", command=self.clear_output).pack(side="left", padx=(10, 0))

        output_label = ttk.Label(main, text="Resultado:", font=("Arial", 10, "bold"))
        output_label.pack(anchor="w", pady=(8, 4))

        self.output = scrolledtext.ScrolledText(main, wrap=tk.WORD, height=24)
        self.output.pack(fill="both", expand=True)
        self.output.insert(tk.END, "Selecione os arquivos CSV e clique em Executar análise.\n")
        self.output.configure(state="disabled")

    def _browse_file(self, var, key):
        path = filedialog.askopenfilename(
            title=f"Selecionar {key}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent),
        )
        if path:
            var.set(path)

    def clear_output(self):
        self.output.configure(state="normal")
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Selecione os arquivos CSV e clique em Executar análise.\n")
        self.output.configure(state="disabled")

    def run_gui_analysis(self):
        paths = {
            "friedman": self.friedman_path.get(),
            "cliff": self.cliff_path.get(),
            "bootstrap": self.bootstrap_path.get(),
            "integrado": self.integrado_path.get(),
        }

        missing = [name for name, path in paths.items() if not path or not os.path.exists(path)]
        if missing:
            messagebox.showwarning("Arquivos ausentes", "Selecione os arquivos CSV necessários antes de executar.\nFaltando: " + ", ".join(missing))
            return

        try:
            report = run_analysis(
                friedman_path=paths["friedman"],
                cliff_path=paths["cliff"],
                bootstrap_path=paths["bootstrap"],
                integrado_path=paths["integrado"],
            )
            self.output.configure(state="normal")
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, report)
            self.output.configure(state="disabled")
        except Exception as exc:
            self.output.configure(state="normal")
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Erro ao executar a análise:\n{exc}")
            self.output.configure(state="disabled")
            messagebox.showerror("Erro", str(exc))


if __name__ == "__main__":
    if "--gui" in sys.argv or len(sys.argv) == 1:
        app = TinyMLApp()
        app.mainloop()
    else:
        args = sys.argv[1:]
        if len(args) == 4:
            report = run_analysis(*args)
            print(report)
        else:
            print("Uso: python aula_planejamento_experimental_tinyml.py [--gui] [friedman.csv cliff.csv bootstrap.csv integrado.csv]")
            print("Ou apenas execute sem argumentos para abrir a interface gráfica.")
