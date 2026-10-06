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
        self.geometry("1280x820")
        self.minsize(1020, 680)
        self.dark_mode = False
        self._text_boxes = []

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        header = ttk.Frame(self, padding=(20, 16, 20, 10))
        header.pack(fill="x")
        heading = ttk.Frame(header)
        heading.pack(side="left", fill="x", expand=True)
        ttk.Label(heading, text="Planejamento Experimental", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            heading,
            text="Análises estatísticas e visualizações para experimentos TinyML",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        self.theme_button = ttk.Button(
            header,
            text="Modo escuro",
            command=self._toggle_theme,
            style="Secondary.TButton",
        )
        self.theme_button.pack(side="right", padx=(12, 0))

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(4, 10))

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
        footer = ttk.Frame(self, padding=(18, 0, 18, 12))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var, anchor="w").pack(fill="x")
        self._apply_theme()

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
        top = ttk.Frame(container, padding=(16, 14, 16, 10))
        top.pack(fill="x")

        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Label(row, text=label_text, width=20, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=self.file_vars[key]).pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(
            row,
            text="Selecionar CSV",
            style="Accent.TButton",
            command=lambda: self._browse_file(self.file_vars[key]),
        ).pack(side="left")

        action_row = ttk.Frame(top)
        action_row.pack(fill="x", pady=(12, 0))
        ttk.Button(
            action_row,
            text="Executar análise",
            style="Success.TButton",
            command=command,
        ).pack(side="left")
        ttk.Button(
            action_row,
            text="Limpar",
            style="Secondary.TButton",
            command=lambda: self._clear_tab(container, key),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            action_row,
            text="Exportar relatório TXT",
            style="Secondary.TButton",
            command=lambda: self._export_report(container, key),
        ).pack(side="left", padx=(8, 0))

        output = ttk.Panedwindow(container, orient="horizontal")
        output.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        result_frame = ttk.LabelFrame(output, text="Relatório", padding=8)
        plot_frame = ttk.LabelFrame(output, text="Visualização", padding=8)
        output.add(result_frame, weight=1)
        output.add(plot_frame, weight=2)

        result_box = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10),
            padx=10,
            pady=10,
            borderwidth=0,
        )
        result_box.pack(fill="both", expand=True)
        result_box.insert(tk.END, "Selecione um arquivo CSV e execute a análise.")
        result_box.configure(state="disabled")
        setattr(container, "result_box", result_box)
        setattr(container, "report_text", "")
        setattr(container, "figure", None)
        setattr(container, "canvas", None)
        self._text_boxes.append(result_box)

        plot_holder = ttk.Frame(plot_frame)
        plot_holder.pack(fill="both", expand=True)
        setattr(container, "plot_holder", plot_holder)

    def _browse_file(self, var):
        path = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent),
        )
        if path:
            var.set(path)

    def _render_plot(self, tab, fig):
        parent = tab.plot_holder
        previous_figure = getattr(tab, "figure", None)
        if previous_figure is not None and previous_figure is not fig:
            plt.close(previous_figure)
        for widget in parent.winfo_children():
            widget.destroy()
        self._style_figure(fig)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(tab, "figure", fig)
        setattr(tab, "canvas", canvas)
        return canvas

    def _set_result(self, tab, text):
        setattr(tab, "report_text", text)
        tab.result_box.configure(state="normal")
        tab.result_box.delete(1.0, tk.END)
        tab.result_box.insert(tk.END, text)
        tab.result_box.configure(state="disabled")

    def _clear_tab(self, tab, key):
        self.file_vars[key].set("")
        self._set_result(tab, "")
        tab.result_box.configure(state="normal")
        tab.result_box.insert(tk.END, "Selecione um arquivo CSV e execute a análise.")
        tab.result_box.configure(state="disabled")

        figure = getattr(tab, "figure", None)
        if figure is not None:
            plt.close(figure)
            tab.figure = None
        tab.canvas = None
        for widget in tab.plot_holder.winfo_children():
            widget.destroy()
        self.status_var.set("Aba limpa. Selecione um arquivo CSV para começar.")

    def _export_report(self, tab, key):
        report = getattr(tab, "report_text", "")
        if not report.strip():
            messagebox.showwarning("Sem relatório", "Execute uma análise antes de exportar o relatório.")
            return

        path = filedialog.asksaveasfilename(
            title="Exportar relatório",
            defaultextension=".txt",
            initialfile=f"relatorio_{key}.txt",
            filetypes=[("Arquivo de texto", "*.txt")],
        )
        if not path:
            return
        try:
            Path(path).write_text(report + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Falha ao exportar", f"Não foi possível salvar o relatório:\n{exc}")
            return
        self.status_var.set(f"Relatório exportado: {path}")

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_button.configure(text="Modo claro" if self.dark_mode else "Modo escuro")
        self._apply_theme()
        for tab in (self.friedman_tab, self.cliff_tab, self.bootstrap_tab, self.integrado_tab):
            figure = getattr(tab, "figure", None)
            canvas = getattr(tab, "canvas", None)
            if figure is not None and canvas is not None:
                self._style_figure(figure)
                canvas.draw_idle()

    def _apply_theme(self):
        if self.dark_mode:
            colors = {
                "background": "#111827",
                "surface": "#1F2937",
                "foreground": "#F9FAFB",
                "muted": "#CBD5E1",
                "input": "#0F172A",
                "border": "#374151",
            }
        else:
            colors = {
                "background": "#F3F4F6",
                "surface": "#FFFFFF",
                "foreground": "#111827",
                "muted": "#64748B",
                "input": "#FFFFFF",
                "border": "#D1D5DB",
            }

        background = colors["background"]
        surface = colors["surface"]
        foreground = colors["foreground"]
        self.configure(background=background)
        self.style.configure("TFrame", background=background)
        self.style.configure("TLabel", background=background, foreground=foreground)
        self.style.configure("Heading.TLabel", background=background, foreground=foreground, font=("Segoe UI", 20, "bold"))
        self.style.configure("Subtitle.TLabel", background=background, foreground=colors["muted"], font=("Segoe UI", 10))
        self.style.configure("TLabelFrame", background=background, foreground=foreground, bordercolor=colors["border"])
        self.style.configure("TLabelFrame.Label", background=background, foreground=foreground, font=("Segoe UI", 10, "bold"))
        self.style.configure(
            "TNotebook",
            background=background,
            bordercolor=colors["border"],
            tabmargins=(2, 4, 2, 0),
        )
        self.style.configure(
            "TNotebook.Tab",
            background=surface,
            foreground=foreground,
            padding=(16, 9),
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#4F46E5" if not self.dark_mode else "#6366F1")],
            foreground=[("selected", "#FFFFFF")],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=colors["input"],
            foreground=foreground,
            insertcolor=foreground,
            bordercolor=colors["border"],
            padding=7,
        )
        self.style.configure(
            "Secondary.TButton",
            background=surface,
            foreground=foreground,
            bordercolor=colors["border"],
            padding=(12, 7),
            font=("Segoe UI", 9, "bold"),
        )
        self.style.map("Secondary.TButton", background=[("active", colors["border"])])
        for style_name, color, active in (
            ("Accent.TButton", "#4F46E5", "#4338CA"),
            ("Success.TButton", "#16A34A", "#15803D"),
        ):
            self.style.configure(
                style_name,
                background=color,
                foreground="#FFFFFF",
                bordercolor=color,
                padding=(12, 7),
                font=("Segoe UI", 9, "bold"),
            )
            self.style.map(style_name, background=[("active", active)])

        for text_box in self._text_boxes:
            text_box.configure(
                background=colors["input"],
                foreground=foreground,
                insertbackground=foreground,
                selectbackground="#4F46E5",
                selectforeground="#FFFFFF",
            )

    def _style_figure(self, fig):
        background = "#1F2937" if self.dark_mode else "#FFFFFF"
        foreground = "#F9FAFB" if self.dark_mode else "#111827"
        fig.set_facecolor(background)
        for axis in fig.axes:
            axis.set_facecolor(background)
            axis.title.set_color(foreground)
            axis.xaxis.label.set_color(foreground)
            axis.yaxis.label.set_color(foreground)
            axis.tick_params(colors=foreground)
            for spine in axis.spines.values():
                spine.set_color("#64748B" if self.dark_mode else "#9CA3AF")
            legend = axis.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(background)
                legend.get_frame().set_edgecolor("#64748B" if self.dark_mode else "#D1D5DB")
                for text in legend.get_texts():
                    text.set_color(foreground)

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
            self._render_plot(tab, result["figure"])
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
            self._render_plot(tab, result["figure"])
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
            self._render_plot(tab, result["figure"])
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
            self._render_plot(tab, result["figure"])
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
