# -*- coding: utf-8 -*-
"""Interface gráfica para análise experimental TinyML com Tkinter.

A interface separa cada teste em abas e exibe os gráficos dentro da janela.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


def build_friedman_figure(df):
    fig = Figure(figsize=(7.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.boxplot([df[c] for c in ESTRATEGIAS])
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["Baseline", "Quantização", "Poda", "HW-NAS"])
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-score por estratégia")
    fig.tight_layout()
    return fig


def analyze_friedman(df):
    missing = [col for col in ESTRATEGIAS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas: {missing}")

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

    report = [
        "=== FRIEDMAN ===",
        f"Estatística: {resultado.statistic:.6f}",
        f"p-valor: {resultado.pvalue:.6f}",
        "Conclusão: " + ("rejeitamos H0. Existe diferença global entre as estratégias." if resultado.pvalue < 0.05 else "não rejeitamos H0."),
        "",
        "Postos médios:",
        str(postos.to_frame("Posto_medio")),
        "",
        "Comparações significativas (p < 0.05):",
    ]
    if comparacoes:
        report.extend(comparacoes)
    else:
        report.append("Nenhuma comparação significativa encontrada.")

    fig = build_friedman_figure(df)
    return {"resultado": resultado, "postos": postos, "comparacoes": comparacoes, "report": "\n".join(report), "figure": fig}


def build_cliff_figure(df):
    fig = Figure(figsize=(7.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.boxplot([df["Baseline"], df["Quantizacao"]])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Quantização"])
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Baseline × Quantização")
    fig.tight_layout()
    return fig


def analyze_cliff(df):
    required = ["Baseline", "Quantizacao"]
    if any(col not in df.columns for col in required):
        raise ValueError("CSV deve conter Baseline e Quantizacao.")

    delta = cliffs_delta(df["Quantizacao"], df["Baseline"])
    magnitude, direcao = interpretar_delta(delta)
    report = [
        "=== CLIFF'S DELTA ===",
        f"Delta = {delta:.3f}",
        f"Magnitude: {magnitude}",
        f"Direção: {direcao}",
    ]
    fig = build_cliff_figure(df)
    return {"delta": delta, "magnitude": magnitude, "direcao": direcao, "report": "\n".join(report), "figure": fig}


def build_bootstrap_figure(dif):
    rng = np.random.default_rng(2026)
    B = 10000
    medias_bootstrap = np.empty(B)
    for i in range(B):
        amostra = rng.choice(dif, size=len(dif), replace=True)
        medias_bootstrap[i] = np.mean(amostra)

    media_bootstrap = float(np.mean(medias_bootstrap))
    ic95 = np.percentile(medias_bootstrap, [2.5, 97.5])

    fig = Figure(figsize=(7.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(medias_bootstrap, bins=35, color="steelblue", edgecolor="black")
    ax.axvline(media_bootstrap, linestyle="--", color="red", label=f"Média = {media_bootstrap:.2f} ms")
    ax.axvline(ic95[0], linestyle=":", color="darkgreen", label=f"IC95% inferior = {ic95[0]:.2f}")
    ax.axvline(ic95[1], linestyle=":", color="darkgreen", label=f"IC95% superior = {ic95[1]:.2f}")
    ax.set_xlabel("Diferença média de latência (ms)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição Bootstrap — 10.000 reamostragens")
    ax.legend()
    fig.tight_layout()
    return fig, media_bootstrap, ic95


def analyze_bootstrap(df):
    if "Diferenca_ms" not in df.columns:
        raise ValueError("CSV deve conter a coluna Diferenca_ms.")

    dif = df["Diferenca_ms"].to_numpy(dtype=float)
    fig, media_bootstrap, ic95 = build_bootstrap_figure(dif)
    report = [
        "=== BOOTSTRAP ===",
        f"Diferença média observada: {dif.mean():.2f} ms",
        f"Estimativa média: {media_bootstrap:.2f} ms",
        f"IC95%: [{ic95[0]:.2f}; {ic95[1]:.2f}] ms",
    ]
    return {"dif": dif, "media": media_bootstrap, "ic95": ic95, "report": "\n".join(report), "figure": fig}


def build_integrado_figure(df):
    fig = Figure(figsize=(8, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    resumo = df.groupby(["Hardware", "Estrategia"])["F1_score"].mean().unstack(fill_value=0)
    resumo.plot(kind="bar", ax=ax)
    ax.set_title("F1-score médio por hardware e estratégia")
    ax.set_ylabel("F1-score (%)")
    ax.set_xlabel("Hardware")
    ax.legend(title="Estratégia")
    fig.tight_layout()
    return fig


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
    fig = build_integrado_figure(df)
    report = [
        "=== DATASET INTEGRADO ===",
        str(resumo),
    ]
    return {"resumo": resumo, "report": "\n".join(report), "figure": fig}


def run_analysis(friedman_path=None, cliff_path=None, bootstrap_path=None, integrado_path=None):
    parts = []
    if friedman_path:
        df = pd.read_csv(friedman_path)
        result = analyze_friedman(df)
        parts.append(result["report"])
    if cliff_path:
        df = pd.read_csv(cliff_path)
        result = analyze_cliff(df)
        parts.append(result["report"])
    if bootstrap_path:
        df = pd.read_csv(bootstrap_path)
        result = analyze_bootstrap(df)
        parts.append(result["report"])
    if integrado_path:
        df = pd.read_csv(integrado_path)
        result = analyze_integrado(df)
        parts.append(result["report"])
    return "\n\n".join(parts)


class TinyMLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Planejamento Experimental TinyML")
        self.geometry("1100x760")
        self.minsize(980, 640)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)

        self.friedman_tab = ttk.Frame(self.tabs)
        self.cliff_tab = ttk.Frame(self.tabs)
        self.bootstrap_tab = ttk.Frame(self.tabs)
        self.integrado_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.friedman_tab, text="Friedman")
        self.tabs.add(self.cliff_tab, text="Cliff's Delta")
        self.tabs.add(self.bootstrap_tab, text="Bootstrap")
        self.tabs.add(self.integrado_tab, text="Integrado")

        self.file_vars = {
            "friedman": tk.StringVar(),
            "cliff": tk.StringVar(),
            "bootstrap": tk.StringVar(),
            "integrado": tk.StringVar(),
        }
        self._configure_default_paths()

        self._build_tab(self.friedman_tab, "CSV Friedman/Nemenyi", "friedman", self.execute_friedman)
        self._build_tab(self.cliff_tab, "CSV Cliff's Delta", "cliff", self.execute_cliff)
        self._build_tab(self.bootstrap_tab, "CSV Bootstrap", "bootstrap", self.execute_bootstrap)
        self._build_tab(self.integrado_tab, "CSV Integrado", "integrado", self.execute_integrado)

        self.status_var = tk.StringVar(value="Selecione os arquivos e execute cada teste em sua aba.")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=12, pady=(0, 12))

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
                    self.file_vars[key].set(str(candidate))
                    break

    def _build_tab(self, container, label_text, key, command):
        top = ttk.Frame(container, padding=12)
        top.pack(fill="x")

        row = ttk.Frame(top)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label_text, width=20, anchor="w").pack(side="left")
        entry = ttk.Entry(row, textvariable=self.file_vars[key], width=80)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        tk.Button(row, text="Procurar", bg="#4F46E5", fg="white", activebackground="#4338CA", command=lambda: self._browse_file(self.file_vars[key])).pack(side="left")

        action_row = ttk.Frame(top)
        action_row.pack(fill="x", pady=(8, 12))
        tk.Button(action_row, text="Executar teste", bg="#16A34A", fg="white", activebackground="#15803D", command=command, width=18, height=1).pack(side="left")

        result_frame = ttk.LabelFrame(container, text="Resultado e gráfico")
        result_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        result_box = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=11)
        result_box.pack(fill="x", padx=8, pady=(8, 6))
        result_box.insert(tk.END, "Aguardando execução...\n")
        result_box.configure(state="disabled")
        setattr(container, "result_box", result_box)

        plot_holder = ttk.Frame(result_frame)
        plot_holder.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        setattr(container, "plot_holder", plot_holder)

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent),
        )
        if path:
            var.set(path)

    def _render_plot(self, parent, fig):
        for widget in parent.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _set_result(self, tab, text):
        tab.result_box.configure(state="normal")
        tab.result_box.delete(1.0, tk.END)
        tab.result_box.insert(tk.END, text)
        tab.result_box.configure(state="disabled")

    def execute_friedman(self):
        tab = self.friedman_tab
        path = self.file_vars["friedman"].get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Arquivo obrigatório", "Selecione o CSV do teste de Friedman.")
            return
        try:
            df = pd.read_csv(path)
            result = analyze_friedman(df)
            self._set_result(tab, result["report"])
            self._render_plot(tab.plot_holder, result["figure"])
            self.status_var.set("Teste de Friedman executado com sucesso.")
        except Exception as exc:
            self._set_result(tab, f"Erro:\n{exc}")
            self.status_var.set(f"Erro no teste de Friedman: {exc}")
            messagebox.showerror("Erro", str(exc))

    def execute_cliff(self):
        tab = self.cliff_tab
        path = self.file_vars["cliff"].get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Arquivo obrigatório", "Selecione o CSV do teste de Cliff's Delta.")
            return
        try:
            df = pd.read_csv(path)
            result = analyze_cliff(df)
            self._set_result(tab, result["report"])
            self._render_plot(tab.plot_holder, result["figure"])
            self.status_var.set("Teste de Cliff's Delta executado com sucesso.")
        except Exception as exc:
            self._set_result(tab, f"Erro:\n{exc}")
            self.status_var.set(f"Erro no teste de Cliff's Delta: {exc}")
            messagebox.showerror("Erro", str(exc))

    def execute_bootstrap(self):
        tab = self.bootstrap_tab
        path = self.file_vars["bootstrap"].get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Arquivo obrigatório", "Selecione o CSV do bootstrap.")
            return
        try:
            df = pd.read_csv(path)
            result = analyze_bootstrap(df)
            self._set_result(tab, result["report"])
            self._render_plot(tab.plot_holder, result["figure"])
            self.status_var.set("Bootstrap executado com sucesso.")
        except Exception as exc:
            self._set_result(tab, f"Erro:\n{exc}")
            self.status_var.set(f"Erro no bootstrap: {exc}")
            messagebox.showerror("Erro", str(exc))

    def execute_integrado(self):
        tab = self.integrado_tab
        path = self.file_vars["integrado"].get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Arquivo obrigatório", "Selecione o CSV integrado.")
            return
        try:
            df = pd.read_csv(path)
            result = analyze_integrado(df)
            self._set_result(tab, result["report"])
            self._render_plot(tab.plot_holder, result["figure"])
            self.status_var.set("Resumo integrado executado com sucesso.")
        except Exception as exc:
            self._set_result(tab, f"Erro:\n{exc}")
            self.status_var.set(f"Erro no resumo integrado: {exc}")
            messagebox.showerror("Erro", str(exc))


if __name__ == "__main__":
    if "--cli" in sys.argv:
        args = sys.argv[2:]
        if len(args) == 4:
            print(run_analysis(*args))
        else:
            print("Uso: python aula_planejamento_experimental_tinyml.py --cli friedman.csv cliff.csv bootstrap.csv integrado.csv")
    else:
        app = TinyMLApp()
        app.mainloop()
