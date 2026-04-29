#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF Inspector Ultra v3.0
=========================
Interface graphique ultra-puissante pour l'analyse de fichiers GGUF.
Corrections principales:
  - Protection totale contre les divisions par zero
  - Support GGUF v3 avec alignment
  - Visualisations enrichies et interactives
  - Theme moderne avec customtkinter (fallback tkinter)
  - Export JSON/CSV/Markdown
  - Comparaison multi-modeles
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Matplotlib configuration
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

# Try customtkinter for modern look
try:
    import customtkinter as ctk
    USE_CTK = True
except ImportError:
    USE_CTK = False

# Import our bulletproof analyzer
from GGUF_analyzer_v3 import GGUFAnalyzer, safe_div, safe_mean, safe_std, safe_min, safe_max


# =============================================================================
# THEME & STYLING
# =============================================================================

class Theme:
    """Palette de couleurs moderne"""
    BG_DARK = "#0d1117"
    BG_CARD = "#161b22"
    BG_INPUT = "#21262d"
    BORDER = "#30363d"
    TEXT_PRIMARY = "#c9d1d9"
    TEXT_SECONDARY = "#8b949e"
    ACCENT = "#58a6ff"
    ACCENT_HOVER = "#79b8ff"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    DANGER = "#f85149"
    INFO = "#58a6ff"
    CHART_COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#56d4dd', '#79c0ff', '#ffa657']


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ModernButton:
    """Bouton style moderne"""
    def __init__(self, parent, text, command, color=Theme.ACCENT, **kwargs):
        if USE_CTK:
            self.widget = ctk.CTkButton(
                parent, text=text, command=command,
                fg_color=color, hover_color=Theme.ACCENT_HOVER,
                text_color="white", font=("Segoe UI", 11, "bold"),
                corner_radius=8, height=36, **kwargs
            )
        else:
            self.widget = tk.Button(
                parent, text=text, command=command,
                bg=color, fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=20, pady=8,
                activebackground=Theme.ACCENT_HOVER, cursor="hand2",
                **kwargs
            )

    def pack(self, **kwargs):
        self.widget.pack(**kwargs)

    def grid(self, **kwargs):
        self.widget.grid(**kwargs)

    def config(self, **kwargs):
        self.widget.configure(**kwargs)


class CardFrame:
    """Carte avec bordure arrondie"""
    def __init__(self, parent, title="", **kwargs):
        if USE_CTK:
            self.frame = ctk.CTkFrame(parent, fg_color=Theme.BG_CARD, corner_radius=12, **kwargs)
        else:
            self.frame = tk.LabelFrame(parent, text=title, bg=Theme.BG_CARD,
                                       fg=Theme.TEXT_PRIMARY, font=("Segoe UI", 10, "bold"),
                                       relief="solid", bd=1, **kwargs)
        if USE_CTK and title:
            self.label = ctk.CTkLabel(self.frame, text=title, font=("Segoe UI", 12, "bold"),
                                      text_color=Theme.TEXT_PRIMARY)
            self.label.pack(anchor="w", padx=15, pady=(10, 5))

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class GGUFInspectorUltra(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔬 GGUF Inspector Ultra v3.0")
        self.geometry("1400x900")
        self.minsize(1200, 700)

        # Theme
        self.configure(bg=Theme.BG_DARK)
        self._setup_styles()

        # State
        self.file_path = tk.StringVar()
        self.sample_size = tk.IntVar(value=10)
        self.enable_structure = tk.BooleanVar(value=True)
        self.enable_tensors = tk.BooleanVar(value=True)
        self.enable_architecture = tk.BooleanVar(value=True)
        self.enable_advanced = tk.BooleanVar(value=True)
        self.result_queue = queue.Queue()
        self.current_analyzer = None
        self.plot_canvases = []
        self.analyzers_history = []  # For comparison

        self._build_ui()
        self._configure_layout()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=Theme.BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=Theme.BG_CARD, foreground=Theme.TEXT_PRIMARY,
                        font=("Segoe UI", 10), padding=[15, 8])
        style.map("TNotebook.Tab", background=[("selected", Theme.ACCENT)],
                  foreground=[("selected", "white")])
        style.configure("TProgressbar", background=Theme.ACCENT, troughcolor=Theme.BG_INPUT)

    def _build_ui(self):
        # Main container
        main = tk.Frame(self, bg=Theme.BG_DARK)
        main.pack(fill="both", expand=True, padx=15, pady=15)
        main.grid_rowconfigure(3, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # === HEADER ===
        header = tk.Frame(main, bg=Theme.BG_DARK)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(header, text="🔬 GGUF Inspector Ultra", font=("Segoe UI", 20, "bold"),
                 bg=Theme.BG_DARK, fg=Theme.ACCENT).pack(side="left")

        tk.Label(header, text="v3.0 — Zero-Division-Proof", font=("Segoe UI", 10),
                 bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY).pack(side="left", padx=(10, 0))

        # === FILE SELECTION ===
        file_card = CardFrame(main, title="📁 Fichier GGUF")
        file_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        fc = file_card.frame

        tk.Entry(fc, textvariable=self.file_path, state="readonly",
                 bg=Theme.BG_INPUT, fg=Theme.TEXT_PRIMARY,
                 font=("Consolas", 10), relief="flat", highlightthickness=1,
                 highlightbackground=Theme.BORDER).pack(side="left", fill="x", expand=True, padx=15, pady=10)

        ModernButton(fc, "📂 Parcourir...", self._select_file).widget.pack(side="left", padx=(0, 15))

        # === CONFIGURATION ===
        config_card = CardFrame(main, title="⚙️ Configuration")
        config_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        cc = config_card.frame

        opts = tk.Frame(cc, bg=Theme.BG_CARD)
        opts.pack(fill="x", padx=15, pady=10)

        checks = [
            ("Structure", self.enable_structure),
            ("Tenseurs", self.enable_tensors),
            ("Architecture", self.enable_architecture),
            ("Patterns Avances", self.enable_advanced),
        ]
        for i, (text, var) in enumerate(checks):
            cb = tk.Checkbutton(opts, text=text, variable=var,
                                bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                                selectcolor=Theme.BG_INPUT, activebackground=Theme.BG_CARD,
                                font=("Segoe UI", 10))
            cb.grid(row=0, column=i, padx=15, sticky="w")

        tk.Label(opts, text="Sample:", bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                 font=("Segoe UI", 10)).grid(row=0, column=4, padx=(30, 5))
        sp = tk.Spinbox(opts, from_=1, to=50, textvariable=self.sample_size, width=5,
                        bg=Theme.BG_INPUT, fg=Theme.TEXT_PRIMARY, font=("Consolas", 10))
        sp.grid(row=0, column=5)

        # === CONTROLS ===
        ctrl = tk.Frame(main, bg=Theme.BG_DARK)
        ctrl.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.run_btn = ModernButton(ctrl, "🚀 Lancer l'analyse", self._run_analysis, Theme.SUCCESS)
        self.run_btn.pack(side="left")
        self.run_btn.config(state="disabled")

        self.progress = ttk.Progressbar(ctrl, mode='indeterminate', length=300)
        self.progress.pack(side="left", padx=15, fill="x", expand=True)

        ModernButton(ctrl, "📊 Comparer", self._compare_models, Theme.INFO).widget.pack(side="left", padx=(0, 10))
        ModernButton(ctrl, "💾 Exporter", self._export_results, Theme.WARNING).widget.pack(side="left", padx=(0, 10))

        self.status_label = tk.Label(ctrl, text="Pret. Selectionnez un fichier GGUF.",
                                     bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY,
                                     font=("Consolas", 9))
        self.status_label.pack(side="left", padx=15)

        # === NOTEBOOK (Results) ===
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=4, column=0, sticky="nsew")
        main.grid_rowconfigure(4, weight=1)

        # Tab 1: Report
        self._build_report_tab()
        # Tab 2: Visualizations
        self._build_viz_tab()
        # Tab 3: Tensors Table
        self._build_table_tab()
        # Tab 4: Comparison
        self._build_compare_tab()

    def _build_report_tab(self):
        report_tab = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.notebook.add(report_tab, text="📄 Rapport")

        self.text_widget = tk.Text(report_tab, wrap='word', font=("Consolas", 10),
                                   bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                                   insertbackground=Theme.TEXT_PRIMARY,
                                   relief="flat", padx=10, pady=10)
        vs = ttk.Scrollbar(report_tab, orient='vertical', command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=vs.set)

        report_tab.grid_rowconfigure(0, weight=1)
        report_tab.grid_columnconfigure(0, weight=1)
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")

    def _build_viz_tab(self):
        viz_tab = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.notebook.add(viz_tab, text="📊 Visualisations")
        viz_tab.grid_rowconfigure(0, weight=1)
        viz_tab.grid_columnconfigure(0, weight=1)

        self.plot_canvas_container = tk.Canvas(viz_tab, bg=Theme.BG_DARK, highlightthickness=0)
        self.plot_frame = tk.Frame(self.plot_canvas_container, bg=Theme.BG_DARK)
        self.plot_scroll = ttk.Scrollbar(viz_tab, orient="vertical", command=self.plot_canvas_container.yview)
        self.plot_canvas_container.configure(yscrollcommand=self.plot_scroll.set)

        self.plot_frame.bind("<Configure>",
            lambda e: self.plot_canvas_container.configure(
                scrollregion=self.plot_canvas_container.bbox("all")))
        self.plot_canvas_container.create_window((0, 0), window=self.plot_frame, anchor="nw")

        self.plot_canvas_container.grid(row=0, column=0, sticky="nsew")
        self.plot_scroll.grid(row=0, column=1, sticky="ns")

        self.plot_canvas_container.bind("<Enter>", self._bind_mousewheel)
        self.plot_canvas_container.bind("<Leave>", self._unbind_mousewheel)

    def _build_table_tab(self):
        table_tab = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.notebook.add(table_tab, text="📋 Tenseurs")

        columns = ("Nom", "Type", "Shape", "Taille (MB)", "Elements", "Quantifie")
        self.tree = ttk.Treeview(table_tab, columns=columns, show="headings", height=20)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)

        vs = ttk.Scrollbar(table_tab, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(table_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)

        table_tab.grid_rowconfigure(0, weight=1)
        table_tab.grid_columnconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")

    def _build_compare_tab(self):
        compare_tab = tk.Frame(self.notebook, bg=Theme.BG_DARK)
        self.notebook.add(compare_tab, text="🔄 Comparaison")

        self.compare_text = tk.Text(compare_tab, wrap='word', font=("Consolas", 10),
                                    bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                                    relief="flat", padx=10, pady=10)
        vs = ttk.Scrollbar(compare_tab, orient='vertical', command=self.compare_text.yview)
        self.compare_text.configure(yscrollcommand=vs.set)

        compare_tab.grid_rowconfigure(0, weight=1)
        compare_tab.grid_columnconfigure(0, weight=1)
        self.compare_text.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")

    def _configure_layout(self):
        pass

    def _bind_mousewheel(self, event):
        self.plot_canvas_container.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.plot_canvas_container.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.plot_canvas_container.yview_scroll(int(-1*(event.delta/120)), "units")

    def _select_file(self):
        file = filedialog.askopenfilename(
            title="Selectionner un fichier GGUF",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")]
        )
        if file:
            self.file_path.set(file)
            self.run_btn.config(state="normal")
            self.status_label.config(text=f"Charge: {Path(file).name}", fg=Theme.SUCCESS)
            self._clear_results()

    def _clear_results(self):
        self.text_widget.config(state='normal')
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.config(state='disabled')

        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.plot_canvases.clear()

        for item in self.tree.get_children():
            self.tree.delete(item)

    def _run_analysis(self):
        if not self.file_path.get():
            return

        self.run_btn.config(state="disabled")
        self.progress.start(10)
        self.status_label.config(text="Analyse en cours...", fg=Theme.WARNING)
        self._clear_results()

        options = {
            'structure': self.enable_structure.get(),
            'tensors': self.enable_tensors.get(),
            'architecture': self.enable_architecture.get(),
            'advanced': self.enable_advanced.get(),
            'sample_size': self.sample_size.get()
        }

        thread = threading.Thread(target=self._analysis_worker, args=(options,), daemon=True)
        thread.start()

    def _analysis_worker(self, options):
        try:
            analyzer = GGUFAnalyzer(self.file_path.get())
            if options['structure']: analyzer.analyze_structure()
            if options['tensors']: analyzer.analyze_tensors(options['sample_size'])
            if options['architecture']: analyzer.analyze_architecture()
            if options['advanced']: analyzer.analyze_advanced_patterns()

            report = analyzer.generate_comprehensive_report()
            self.result_queue.put({'status': 'success', 'analyzer': analyzer, 'report': report})
        except Exception as e:
            import traceback
            self.result_queue.put({'status': 'error', 'message': str(e) + "\n" + traceback.format_exc()})
        finally:
            self.after(0, self._on_analysis_complete)

    def _on_analysis_complete(self):
        self.progress.stop()
        self.run_btn.config(state="normal")

        try:
            result = self.result_queue.get_nowait()
            if result['status'] == 'error':
                messagebox.showerror("Erreur", result['message'])
                self.status_label.config(text="Analyse echouee.", fg=Theme.DANGER)
                return

            self.current_analyzer = result['analyzer']
            self.analyzers_history.append(self.current_analyzer)

            # Update report
            self.text_widget.config(state='normal')
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, result['report'])
            self.text_widget.config(state='disabled')

            # Update table
            self._populate_table()

            # Generate plots
            self._generate_plots()

            self.status_label.config(text="Analyse terminee avec succes!", fg=Theme.SUCCESS)
            self.notebook.select(1)

        except queue.Empty:
            pass

    def _populate_table(self):
        if not self.current_analyzer:
            return
        for name, info in self.current_analyzer.tensors_info.items():
            self.tree.insert("", "end", values=(
                name[:60] + "..." if len(name) > 60 else name,
                info['type_name'],
                str(info['shape']),
                f"{info['size_mb']:.3f}",
                f"{info['element_count']:,}",
                "Oui" if info['is_quantized'] else "Non"
            ))

    def _generate_plots(self):
        if not self.current_analyzer:
            return

        arch_res = self.current_analyzer.analysis_results.get('architecture', {})
        tens_res = self.current_analyzer.analysis_results.get('tensors', {})
        adv_res = self.current_analyzer.analysis_results.get('advanced', {})

        colors = Theme.CHART_COLORS

        # 1. Distribution des types de quantification (Pie)
        if tens_res.get('quantization_analysis'):
            fig1 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax1 = fig1.add_subplot(111)
            ax1.set_facecolor(Theme.BG_CARD)
            q_data = tens_res['quantization_analysis']
            types = [k for k in q_data if isinstance(q_data[k], dict)]
            counts = [q_data[t]['count'] for t in types]
            if types:
                wedges, texts, autotexts = ax1.pie(
                    counts, labels=types, autopct='%1.1f%%', startangle=90,
                    colors=colors[:len(types)], textprops={'color': Theme.TEXT_PRIMARY, 'fontsize': 9}
                )
                ax1.set_title("Distribution des Types de Quantification", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            self._embed_figure(fig1)

        # 2. Histogramme des tailles
        if tens_res.get('size_distribution'):
            fig2 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax2 = fig2.add_subplot(111)
            ax2.set_facecolor(Theme.BG_CARD)
            sizes = [t['size_mb'] for t in tens_res['size_distribution']]
            if sizes:
                ax2.hist(sizes, bins=min(30, len(sizes)), color=Theme.ACCENT, edgecolor=Theme.BG_DARK, alpha=0.8)
                ax2.set_xlabel("Taille (MB)", color=Theme.TEXT_SECONDARY)
                ax2.set_ylabel("Nombre de tenseurs", color=Theme.TEXT_SECONDARY)
                ax2.set_title("Distribution des Tailles de Tenseurs", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
                ax2.tick_params(colors=Theme.TEXT_SECONDARY)
                ax2.spines['bottom'].set_color(Theme.BORDER)
                ax2.spines['left'].set_color(Theme.BORDER)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                min_size = safe_min(sizes, 1.0)
                max_size = safe_max(sizes, 1.0)
                if max_size > 0 and safe_div(max_size, min_size, 1.0) > 10:
                    ax2.set_yscale('log')
            self._embed_figure(fig2)

        # 3. Progression des parametres par couche
        layer_prog = arch_res.get('layer_analysis', {}).get('parameter_progression', {})
        if layer_prog.get('layer_params'):
            fig3 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax3 = fig3.add_subplot(111)
            ax3.set_facecolor(Theme.BG_CARD)
            layers = range(len(layer_prog['layer_params']))
            params = layer_prog['layer_params']
            mean_p = layer_prog.get('mean_params', 0)
            std_p = layer_prog.get('std_params', 0)

            ax3.plot(layers, params, 'o-', color=Theme.ACCENT, linewidth=2, markersize=5)
            if mean_p > 0:
                ax3.axhline(y=mean_p, color=Theme.WARNING, linestyle='--', linewidth=1.5,
                           label=f"Moyenne: {mean_p:.0f}")
                ax3.fill_between(layers, max(0, mean_p - std_p), mean_p + std_p,
                                alpha=0.15, color=Theme.WARNING)
            ax3.set_xlabel("Couche", color=Theme.TEXT_SECONDARY)
            ax3.set_ylabel("Parametres", color=Theme.TEXT_SECONDARY)
            ax3.set_title("Progression des Parametres par Couche", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax3.tick_params(colors=Theme.TEXT_SECONDARY)
            ax3.spines['bottom'].set_color(Theme.BORDER)
            ax3.spines['left'].set_color(Theme.BORDER)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.legend(facecolor=Theme.BG_CARD, edgecolor=Theme.BORDER, labelcolor=Theme.TEXT_PRIMARY)
            ax3.grid(True, alpha=0.2, color=Theme.BORDER)
            self._embed_figure(fig3)

        # 4. Scores de Qualite
        quality = adv_res.get('quality_assessment', {})
        if quality:
            fig4 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax4 = fig4.add_subplot(111)
            ax4.set_facecolor(Theme.BG_CARD)
            cats = ['Quantification', 'Architecture', 'Optimisation', 'Global']
            scores = [
                quality.get('quantization_quality', {}).get('weighted_score', 0),
                quality.get('architecture_coherence', {}).get('score', 0),
                quality.get('optimization_level', {}).get('score', 0),
                quality.get('overall_score', {}).get('score', 0)
            ]
            bar_colors = [Theme.SUCCESS if s >= 80 else Theme.WARNING if s >= 60 else Theme.DANGER for s in scores]
            bars = ax4.bar(cats, scores, color=bar_colors, alpha=0.85, edgecolor=Theme.BG_DARK, linewidth=1)
            ax4.set_ylabel("Score (/100)", color=Theme.TEXT_SECONDARY)
            ax4.set_ylim(0, 105)
            ax4.set_title("Evaluation de la Qualite", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            ax4.tick_params(colors=Theme.TEXT_SECONDARY)
            ax4.spines['bottom'].set_color(Theme.BORDER)
            ax4.spines['left'].set_color(Theme.BORDER)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            for i, (bar, s) in enumerate(zip(bars, scores)):
                ax4.text(bar.get_x() + bar.get_width()/2., s + 2, str(int(s)),
                        ha='center', fontweight='bold', color=Theme.TEXT_PRIMARY, fontsize=11)
            self._embed_figure(fig4)

        # 5. Repartition par composant (Pie)
        param_dist = arch_res.get('parameter_distribution', {})
        if param_dist.get('by_component'):
            fig5 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax5 = fig5.add_subplot(111)
            ax5.set_facecolor(Theme.BG_CARD)
            comp = param_dist['by_component']
            labels = list(comp.keys())
            values = list(comp.values())
            if sum(values) > 0:
                ax5.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=colors[:len(labels)], textprops={'color': Theme.TEXT_PRIMARY, 'fontsize': 9})
                ax5.set_title("Repartition des Parametres par Composant", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
            self._embed_figure(fig5)

        # 6. Compression par type
        qa = tens_res.get('quantization_analysis', {})
        if qa:
            fig6 = Figure(figsize=(7, 4), dpi=100, facecolor=Theme.BG_CARD)
            ax6 = fig6.add_subplot(111)
            ax6.set_facecolor(Theme.BG_CARD)
            types = [k for k, v in qa.items() if isinstance(v, dict) and 'compression_ratio' in v]
            ratios = [qa[t]['compression_ratio'] for t in types]
            if types:
                bars = ax6.barh(types, ratios, color=Theme.ACCENT, alpha=0.8, edgecolor=Theme.BG_DARK)
                ax6.set_xlabel("Ratio de Compression (x)", color=Theme.TEXT_SECONDARY)
                ax6.set_title("Efficacite de Compression", color=Theme.TEXT_PRIMARY, fontsize=12, fontweight='bold')
                ax6.tick_params(colors=Theme.TEXT_SECONDARY)
                ax6.spines['bottom'].set_color(Theme.BORDER)
                ax6.spines['left'].set_color(Theme.BORDER)
                ax6.spines['top'].set_visible(False)
                ax6.spines['right'].set_visible(False)
                for bar, r in zip(bars, ratios):
                    ax6.text(r + 0.1, bar.get_y() + bar.get_height()/2., f"{r:.1f}x",
                            va='center', color=Theme.TEXT_PRIMARY, fontsize=10)
            self._embed_figure(fig6)

        self.plot_frame.update_idletasks()

    def _embed_figure(self, fig):
        frame = tk.Frame(self.plot_frame, bg=Theme.BG_CARD, bd=1, relief="solid")
        frame.pack(fill="x", pady=10, padx=10)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        toolbar.configure(bg=Theme.BG_CARD)

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.plot_canvases.append(canvas)

    def _compare_models(self):
        if len(self.analyzers_history) < 2:
            messagebox.showinfo("Comparaison", "Analysez au moins 2 modeles pour comparer.")
            return

        self.compare_text.config(state='normal')
        self.compare_text.delete('1.0', tk.END)

        r = ["=" * 80, "COMPARAISON DE MODELES GGUF", "=" * 80, ""]

        for i, a in enumerate(self.analyzers_history[-3:], 1):
            s = a.analysis_results.get('structure', {})
            mi = s.get('model_info', {})
            q = s.get('quantization_summary', {})
            r.append(f"Modele {i}: {a.model_path.name}")
            r.append(f"  Architecture: {mi.get('architecture', 'N/A')}")
            r.append(f"  Taille: {q.get('total_model_size_mb', 0):.1f} MB")
            r.append(f"  Tenseurs: {s.get('tensor_count', 0)}")

            t = a.analysis_results.get('tensors', {}).get('compression_analysis', {})
            r.append(f"  Compression: {t.get('global_compression_ratio', 0):.2f}x")

            ar = a.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
            r.append(f"  Parametres: {safe_div(ar.get('num_layers', 0) * ar.get('hidden_size', 0) * 12, 1e9):.2f}B (est)")
            r.append("")

        self.compare_text.insert(tk.END, "\n".join(r))
        self.compare_text.config(state='disabled')
        self.notebook.select(3)

    def _export_results(self):
        if not self.current_analyzer:
            messagebox.showwarning("Export", "Aucun resultat a exporter.")
            return

        fmt = messagebox.askquestion("Format", "Exporter en JSON? (Non = CSV)")
        path = filedialog.asksaveasfilename(
            defaultextension=".json" if fmt == 'yes' else ".csv",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("Markdown", "*.md")]
        )
        if not path:
            return

        try:
            if path.endswith('.json'):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_analyzer.analysis_results, f, indent=2, default=str)
            elif path.endswith('.csv'):
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(["Nom", "Type", "Shape", "Taille_MB", "Elements", "Quantifie"])
                    for name, info in self.current_analyzer.tensors_info.items():
                        w.writerow([name, info['type_name'], str(info['shape']),
                                   f"{info['size_mb']:.4f}", info['element_count'],
                                   "Oui" if info['is_quantized'] else "Non"])
            else:
                report = self.current_analyzer.generate_comprehensive_report()
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(report)

            messagebox.showinfo("Export", f"Exporte vers:\n{path}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))


if __name__ == "__main__":
    app = GGUFInspectorUltra()
    app.mainloop()
