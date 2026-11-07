#!/usr/bin/env python3
"""
Tema 9 - Despacho Económico de Carga Computacional en Cloud (AWS vs Azure)
Cálculo Diferencial - Proyecto Final - Universidad de La Salle

Modelo
------
    C1(P1) = a1*P1 + b1*P1^2        (AWS EC2 T3a.medium)
    C2(P2) = a2*P2 + b2*P2^2        (Azure Standard_B2als_v2)
    Restricción: P1 + P2 = PD,  0 <= P1 <= PD

Sustituyendo P2 = PD - P1:
    C(P1)   = C1(P1) + C2(PD - P1)
    dC/dP1  = C1'(P1) - C2'(PD - P1) = 0   =>   dC1/dP1 = dC2/dP2   (costo marginal)
    d2C/dP1^2 = 2*b1 + 2*b2 > 0            =>   mínimo (convexa)

Solución analítica interior:
    P1* = (2*b2*PD + a2 - a1) / (2*(b1 + b2))

Si P1* cae fuera de [0, PD], el óptimo factible está en la frontera (solución de esquina).

Salidas (carpeta ./resultados):
    tabla_escenarios.csv, tabla_discreta.csv, tabla_incertidumbre.csv,
    fig_costos.png, fig_p1_vs_demanda.png, fig_discreto.png

Uso:
    pip install numpy matplotlib sympy pandas
    python despacho_economico_cloud.py

Probado con Python 3.12+ (compatible con 3.13).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: reproducible en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

# --------------------------------------------------------------------------
# 1. PARÁMETROS DEL MODELO
# --------------------------------------------------------------------------
HORAS_MES = 730  # horas de operación mensual

# Coeficientes lineales [USD/mes por vCPU equivalente], estimados de precios
# On-Demand Linux (US East) y capacidad base de CPU de cada instancia.
A1, A2 = 68.62, 45.7467
# Coeficientes cuadráticos: parámetros HIPOTÉTICOS de penalización operacional
# (no son tarifas oficiales de los proveedores).
B1, B2 = 0.8, 1.5

CAP_AWS = 0.4  # vCPU equivalentes por instancia AWS (capacidad base)
CAP_AZURE = 0.6  # vCPU equivalentes por instancia Azure (capacidad base)

ESCENARIOS = {"Pequeña": 0.4, "Mediana": 2.0, "Grande": 10.0}

SEMILLA = 42
N_MONTECARLO = 20_000
INCERTIDUMBRE_REL = 0.05  # ±5 % (uniforme) sobre a1, a2, b1, b2 (supuesto)

SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 2. FUNCIONES DE COSTO Y ÓPTIMO ANALÍTICO
# --------------------------------------------------------------------------
def c1(p1, a1=A1, b1=B1):
    return a1 * p1 + b1 * p1**2


def c2(p2, a2=A2, b2=B2):
    return a2 * p2 + b2 * p2**2


def costo_total(p1, pd_, a1=A1, a2=A2, b1=B1, b2=B2):
    return c1(p1, a1, b1) + c2(pd_ - p1, a2, b2)


def p1_critico(pd_, a1=A1, a2=A2, b1=B1, b2=B2):
    """Punto crítico (sin restricciones) de dC/dP1 = 0."""
    return (2 * b2 * pd_ + a2 - a1) / (2 * (b1 + b2))


def p1_optimo(pd_, a1=A1, a2=A2, b1=B1, b2=B2):
    """Óptimo factible: el punto crítico recortado a [0, PD] (C es convexa)."""
    return np.clip(p1_critico(pd_, a1, a2, b1, b2), 0.0, pd_)


# --------------------------------------------------------------------------
# 3. VERIFICACIÓN SIMBÓLICA (SymPy) - paso a paso del apéndice
# --------------------------------------------------------------------------
def derivacion_simbolica() -> dict:
    P1, PD = sp.symbols("P1 P_D", real=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", positive=True)
    C = a1 * P1 + b1 * P1**2 + a2 * (PD - P1) + b2 * (PD - P1) ** 2
    dC = sp.diff(C, P1)
    d2C = sp.diff(C, P1, 2)
    sol = sp.solve(sp.Eq(dC, 0), P1)[0]
    return {
        "C(P1)": sp.simplify(C),
        "dC/dP1": sp.simplify(dC),
        "d2C/dP1^2": sp.simplify(d2C),
        "P1*": sp.simplify(sol),
    }


# --------------------------------------------------------------------------
# 4. ESCENARIOS CONTINUOS
# --------------------------------------------------------------------------
def tabla_escenarios() -> pd.DataFrame:
    filas = []
    for nombre, pd_ in ESCENARIOS.items():
        pc = p1_critico(pd_)
        p1 = float(p1_optimo(pd_))
        p2 = pd_ - p1
        d2 = 2 * (B1 + B2)
        # Costo marginal de cada proveedor en el óptimo
        mc1 = A1 + 2 * B1 * p1
        mc2 = A2 + 2 * B2 * p2
        costo_opt = float(costo_total(p1, pd_))
        costo_aws = float(c1(pd_))  # toda la carga en AWS
        costo_azure = float(c2(pd_))  # toda la carga en Azure
        filas.append(
            {
                "Escenario": nombre,
                "PD [vCPU eq.]": pd_,
                "P1 crítico (sin restr.)": pc,
                "P1* [AWS]": p1,
                "P2* [Azure]": p2,
                "Tipo de solución": "interior" if 0 < pc < pd_ else "esquina",
                "d²C/dP1² [USD/vCPU²]": d2,
                "dC1/dP1 [USD/vCPU]": mc1,
                "dC2/dP2 [USD/vCPU]": mc2,
                "C* [USD/mes]": costo_opt,
                "C todo AWS": costo_aws,
                "C todo Azure": costo_azure,
                "Ahorro vs mejor único [%]": 100
                * (min(costo_aws, costo_azure) - costo_opt)
                / min(costo_aws, costo_azure),
            }
        )
    return pd.DataFrame(filas)


def umbral_demanda_p1_positivo() -> float:
    """Demanda PD a partir de la cual AWS recibe carga (P1* > 0)."""
    return (A1 - A2) / (2 * B2)


# --------------------------------------------------------------------------
# 5. COMPARACIÓN DISCRETA (instancias completas)
# --------------------------------------------------------------------------
def optimo_discreto(pd_: float, max_inst: int = 40) -> dict:
    """
    Instancias completas: con n1 instancias AWS y n2 Azure se instalan
    P1 = 0.4*n1 y P2 = 0.6*n2 vCPU equivalentes (se paga la capacidad completa).
    Se busca el par (n1, n2) de menor costo con capacidad instalada >= PD,
    evaluando el MISMO modelo cuadrático C1 + C2 del caso continuo, de modo
    que ambos costos sean directamente comparables.
    """
    mejor = None
    for n1, n2 in product(range(max_inst + 1), repeat=2):
        p1, p2 = CAP_AWS * n1, CAP_AZURE * n2
        if p1 + p2 + 1e-12 < pd_:
            continue
        costo = float(c1(p1) + c2(p2))
        cand = {"n_AWS": n1, "n_Azure": n2, "P1 instalado": p1,
                "P2 instalado": p2, "Costo discreto [USD/mes]": costo}
        if mejor is None or (costo, n1 + n2) < (
            mejor["Costo discreto [USD/mes]"], mejor["n_AWS"] + mejor["n_Azure"]
        ):
            mejor = cand
    return mejor


def tabla_discreta(cont: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, r in cont.iterrows():
        d = optimo_discreto(r["PD [vCPU eq.]"])
        d["Escenario"] = r["Escenario"]
        d["PD [vCPU eq.]"] = r["PD [vCPU eq.]"]
        d["P1* continuo"] = r["P1* [AWS]"]
        d["P2* continuo"] = r["P2* [Azure]"]
        d["Costo continuo [USD/mes]"] = r["C* [USD/mes]"]
        d["Sobrecosto discreto [%]"] = (
            100 * (d["Costo discreto [USD/mes]"] - r["C* [USD/mes]"]) / r["C* [USD/mes]"]
        )
        filas.append(d)
    cols = [
        "Escenario", "PD [vCPU eq.]", "P1* continuo", "P2* continuo",
        "n_AWS", "n_Azure", "P1 instalado", "P2 instalado",
        "Costo continuo [USD/mes]", "Costo discreto [USD/mes]",
        "Sobrecosto discreto [%]",
    ]
    return pd.DataFrame(filas)[cols]


# --------------------------------------------------------------------------
# 6. PROPAGACIÓN DE INCERTIDUMBRES (Monte Carlo + propagación analítica)
# --------------------------------------------------------------------------
def incertidumbre() -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    u = INCERTIDUMBRE_REL

    def muestra(valor):
        return valor * rng.uniform(1 - u, 1 + u, N_MONTECARLO)

    a1, a2, b1, b2 = muestra(A1), muestra(A2), muestra(B1), muestra(B2)
    filas = []
    for nombre, pd_ in ESCENARIOS.items():
        p1 = p1_optimo(pd_, a1, a2, b1, b2)
        cst = costo_total(p1, pd_, a1, a2, b1, b2)
        filas.append(
            {
                "Escenario": nombre,
                "PD [vCPU eq.]": pd_,
                "P1* media": p1.mean(),
                "P1* desv. est.": p1.std(ddof=1),
                "P1* p2.5": np.percentile(p1, 2.5),
                "P1* p97.5": np.percentile(p1, 97.5),
                "C* media [USD/mes]": cst.mean(),
                "C* desv. est. [USD/mes]": cst.std(ddof=1),
            }
        )
    return pd.DataFrame(filas)


def propagacion_analitica(pd_: float) -> float:
    """
    Propagación de primer orden de P1* (interior) respecto a (a1, a2, b1, b2),
    asumiendo cada parámetro uniforme en ±u*valor => sigma = u*valor/sqrt(3).
    P1* = (2 b2 PD + a2 - a1) / (2 (b1 + b2))
    """
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", positive=True)
    PD = sp.Symbol("PD", positive=True)
    f = (2 * b2 * PD + a2 - a1) / (2 * (b1 + b2))
    vals = {a1: A1, a2: A2, b1: B1, b2: B2, PD: pd_}
    var = 0.0
    for s, v in [(a1, A1), (a2, A2), (b1, B1), (b2, B2)]:
        grad = float(sp.diff(f, s).subs(vals))
        sigma = INCERTIDUMBRE_REL * v / np.sqrt(3)
        var += (grad * sigma) ** 2
    return float(np.sqrt(var))


# --------------------------------------------------------------------------
# 7. FIGURAS
# --------------------------------------------------------------------------
def figura_costos(cont: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (nombre, pd_) in zip(axes, ESCENARIOS.items()):
        p1 = np.linspace(0, pd_, 400)
        ax.plot(p1, costo_total(p1, pd_), lw=2, label="C(P₁)")
        fila = cont[cont["Escenario"] == nombre].iloc[0]
        ax.plot(fila["P1* [AWS]"], fila["C* [USD/mes]"], "ro", label="Óptimo factible")
        pc = fila["P1 crítico (sin restr.)"]
        if 0 <= pc <= pd_:
            ax.axvline(pc, ls="--", c="gray", lw=1)
        ax.set_title(f"{nombre} (P_D = {pd_:g})")
        ax.set_xlabel("P₁ en AWS [vCPU eq.]")
        ax.set_ylabel("Costo total [USD/mes]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_costos.png", dpi=200)
    plt.close(fig)


def figura_p1_vs_demanda(inc: pd.DataFrame):
    rng = np.random.default_rng(SEMILLA)
    u = INCERTIDUMBRE_REL
    a1 = A1 * rng.uniform(1 - u, 1 + u, N_MONTECARLO)
    a2 = A2 * rng.uniform(1 - u, 1 + u, N_MONTECARLO)
    b1 = B1 * rng.uniform(1 - u, 1 + u, N_MONTECARLO)
    b2 = B2 * rng.uniform(1 - u, 1 + u, N_MONTECARLO)

    pds = np.linspace(0.1, 20, 120)
    media, desv = [], []
    for pd_ in pds:
        p = p1_optimo(pd_, a1, a2, b1, b2)
        media.append(p.mean())
        desv.append(p.std(ddof=1))
    media, desv = np.array(media), np.array(desv)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(pds, p1_optimo(pds), "k-", lw=2, label="P₁* (parámetros nominales)")
    ax.fill_between(pds, np.maximum(media - desv, 0), media + desv, alpha=0.3,
                    label="±1σ (Monte Carlo, ±5 % en parámetros)")
    for nombre, pd_ in ESCENARIOS.items():
        r = inc[inc["Escenario"] == nombre].iloc[0]
        ax.errorbar(pd_, r["P1* media"], yerr=r["P1* desv. est."], fmt="o",
                    capsize=4, label=f"{nombre}")
    ax.axvline(umbral_demanda_p1_positivo(), ls="--", c="gray", lw=1)
    ax.text(umbral_demanda_p1_positivo() + 0.2, 0.3,
            f"P_D = {umbral_demanda_p1_positivo():.2f}", fontsize=8)
    ax.set_xlabel("Demanda total P_D [vCPU eq.]")
    ax.set_ylabel("P₁* asignado a AWS [vCPU eq.]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_p1_vs_demanda.png", dpi=200)
    plt.close(fig)


def figura_discreta(cont: pd.DataFrame, disc: pd.DataFrame):
    x = np.arange(len(cont))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(x - w / 2, cont["P1* [AWS]"], w, label="AWS continuo")
    axes[0].bar(x + w / 2, disc["P1 instalado"], w, label="AWS discreto")
    axes[0].set_xticks(x, cont["Escenario"])
    axes[0].set_ylabel("P₁ [vCPU eq.]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x - w / 2, cont["C* [USD/mes]"], w, label="Continuo (modelo)")
    axes[1].bar(x + w / 2, disc["Costo discreto [USD/mes]"], w,
                label="Discreto (instancias completas)")
    axes[1].set_xticks(x, cont["Escenario"])
    axes[1].set_ylabel("Costo [USD/mes]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_discreto.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. PRINCIPAL
# --------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    print("=== Derivación simbólica ===")
    for k, v in derivacion_simbolica().items():
        print(f"{k:12s} = {v}")

    print(f"\nUmbral: AWS recibe carga cuando P_D > (a1 - a2)/(2 b2) = "
          f"{umbral_demanda_p1_positivo():.4f} vCPU eq.")

    cont = tabla_escenarios()
    print("\n=== Escenarios continuos ===")
    print(cont.T.to_string(header=False))
    cont.to_csv(SALIDA / "tabla_escenarios.csv", index=False)

    disc = tabla_discreta(cont)
    print("\n=== Comparación discreta ===")
    print(disc.T.to_string(header=False))
    disc.to_csv(SALIDA / "tabla_discreta.csv", index=False)

    inc = incertidumbre()
    print("\n=== Incertidumbre (Monte Carlo) ===")
    print(inc.T.to_string(header=False))
    inc.to_csv(SALIDA / "tabla_incertidumbre.csv", index=False)
    print(f"\nσ analítica de P1* (primer orden), P_D = 10: "
          f"{propagacion_analitica(10.0):.4f} vCPU eq.")

    figura_costos(cont)
    figura_p1_vs_demanda(inc)
    figura_discreta(cont, disc)
    print(f"\nTablas y figuras guardadas en: {SALIDA}")


if __name__ == "__main__":
    main()
