import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D

# ============================================================
# Constants
# ============================================================
kB = 1.380649e-23  # J / K
eV_to_J = 1.602176634e-19  # J / eV
kB_eV = 8.617333262e-5  # eV / K

# ============================================================
# Eurofer parameters
# ============================================================
b = 0.2737e-9  # m
mu = 80e9  # Pa
Tm = 1811  # K
Tf = 0.9
T0 = Tf * Tm

Bk = 6.6e-5  # Pa s
h = 2.0 * np.sqrt(2.0) / 3.0
w = 25.0
Bkink = Bk * w / (2.0 * h)

tau_p = 0.36e9  # Pa
dH0_eV = 2.2
DeltaH0 = dH0_eV * eV_to_J
p = 0.65
q = 1.7

target_gdot_array = np.array([3e-5, 3e-4, 3e-3]) / 0.333
T_list = np.linspace(300, 900, 51)

alpha_a = 0.35
k1 = 7.8e10  # /m
k20 = 6750  # -
q2 = -0.007  # eV

d_lath = 5e-7  # m
c_block = 0.3
d_block = 3.1e-6  # m
c_PAG = 0.1
d_PAG = 2.1e-5
cmx = 0.3
dmx = 1.5e-6
cm23c6 = 0.2
dm23c6 = 4.56e-7

# Dynamic Strain Aging
tau_ss = 130e6  # Pa
p_ss = 0.37  # -
ta0_ss = 1e-9  # s
Qa_ss = 1.6 * eV_to_J  # J
gamma_ss_exp = 1

beta = 0.1

rho_dict = {}
for T in T_list:
    # temperature dependent recovery coefficient
    k2 = k20 * np.exp(q2 / (kB_eV * T))
    # solve steady-state rho
    sqrt_rho = (k1 + np.sqrt(k1**2 + 4 * k1 * k2 * (1/d_lath + c_block/d_block + c_PAG/d_PAG + cmx/dmx + cm23c6/dm23c6))) / (2 * k2)
    rho_ss = sqrt_rho**2
    rho_dict[T] = rho_ss

plt.figure(figsize=(6, 4))
plt.plot(T_list, [rho_dict[T] for T in T_list], 'o-')
plt.xlabel("Temperature (K)")
plt.ylabel(r"Steady-state $\rho$ (m$^{-2}$)")
plt.yscale("log")
plt.grid(alpha=0.3)


def mu_of_T(T):
    mu0 = 80e9  # Pa at 300 K
    T_ref = 300.0  # K
    Tm = 1811.0  # K
    a_mu = 0.35  # fractional drop by Tm
    mu_T = mu0 * (1.0 - a_mu * (T - T_ref) / (Tm - T_ref))
    return max(mu_T, 0.45 * mu0)


# ============================================================
# BCC screw mobility law
# ============================================================
def Lambda_micro():
    return (1/d_lath + c_block/d_block + c_PAG/d_PAG + cmx/dmx + cm23c6/dm23c6)


def aging_time(T):
    return ta0_ss * np.exp(Qa_ss / (kB * T))


def waiting_time(rho_m, gamma_dot):
    return (1 / (Lambda_micro() + np.sqrt(rho_m))) * rho_m * b / (gamma_dot**gamma_ss_exp)


def ss_strengthening(T, gamma_dot):
    t_aging = aging_time(T)
    t_wait = waiting_time(rho_of_T(T), gamma_dot)
    X = (t_wait / t_aging)**p_ss
    aging_on = np.exp(-X)
    return tau_ss * aging_on


def rho_of_T(T):
    return rho_dict[T] * beta


def tau_a_of_T_gdot(T, gamma_dot):
    rho_T = rho_of_T(T)
    tau_micro = alpha_a * mu_of_T(T) * b * (np.sqrt(rho_T) + Lambda_micro())
    tau_dsa = ss_strengthening(T, gamma_dot)
    return tau_micro + tau_dsa


def dg1_barrier(tau_eff, T):
    x = tau_eff / tau_p
    x = np.clip(x, 0.0, 0.999999)
    dg = (1.0 - x**p)**q - T / T0
    return max(dg, 0.0)


def screw_velocity(tau_total, T, gamma_dot_target, use_athermal=True):
    tau_a_T = tau_a_of_T_gdot(T, gamma_dot_target)
    tau_drive = tau_total - tau_a_T if use_athermal else tau_total
    if tau_drive <= 0:
        return 0.0
    dg1 = dg1_barrier(tau_drive, T)
    expCoeff = np.exp(-DeltaH0 * dg1 / (2.0 * kB * T))
    return tau_drive * b / Bkink * expCoeff


def gamma_dot(tau_total, T, gamma_dot_target, use_athermal=True):
    rho_T = rho_of_T(T)
    return rho_T * b * screw_velocity(tau_total, T, gamma_dot_target, use_athermal=use_athermal)


def solve_tau_for_gdot(T, gdot_target, use_athermal=True):
    tau_a_T = tau_a_of_T_gdot(T, gdot_target)
    tau_min = tau_a_T + 1e-6 if use_athermal else 1e-6
    tau_max = tau_a_T + 0.999 * tau_p if use_athermal else 0.999 * tau_p

    def residual(tau_total):
        return gamma_dot(tau_total, T, gdot_target, use_athermal=use_athermal) - gdot_target

    f_min = residual(tau_min)
    f_max = residual(tau_max)

    if np.isnan(f_min) or np.isnan(f_max):
        return np.nan
    if f_min * f_max > 0:
        return np.nan
    return brentq(residual, tau_min, tau_max, xtol=1e-12, rtol=1e-10, maxiter=500)


# ============================================================
# Compute solved tau(T), local m(T), and Vapp(T)
# ============================================================
# BUG FIX: the "success" branch used to live after a `continue`, nested
# inside the `if not np.isfinite(tau_sol)` block, so it never ran and
# `rows` only ever received NaN entries. Dedented it to its own branch.
rows = []
for use_athermal in [True, False]:
    label = "with_tau_a" if use_athermal else "no_tau_a"
    for target_gdot in target_gdot_array:
        for T in T_list:
            tau_sol = solve_tau_for_gdot(T, target_gdot, use_athermal=use_athermal)
            if not np.isfinite(tau_sol):
                rows.append({
                    "model": label,
                    "gdot": target_gdot,
                    "T": T,
                    "tau": np.nan,
                    "tau_eff": np.nan,
                })
                continue
            tau_eff = tau_sol - tau_a_of_T_gdot(T, target_gdot) if use_athermal else tau_sol
            rows.append({
                "model": label,
                "gdot": target_gdot,
                "T": T,
                "tau": tau_sol,
                "tau_eff": tau_eff,
            })

# ============================================================
# Convert to arrays
# ============================================================
def get_array(model, gdot, key):
    return np.array([r[key] for r in rows if r["model"] == model and r["gdot"] == gdot])


# ============================================================
# Compute global m from log(tau) vs log(gdot)
# ============================================================
# BUG FIX: tau_vals/gd_vals were (1) converted list->ndarray mid-loop, so the
# next .append() call raised AttributeError, (2) never reset per-T, so points
# from every earlier T leaked into later fits, and (3) the row-append call was
# nested inside the `else: m_global = np.nan` branch, so a row was only ever
# stored for the (rare) failing case. Restructured to reset per T and append
# exactly once per (model, T).
global_m_rows = []
for use_athermal in [True, False]:
    label = "with_tau_a" if use_athermal else "no_tau_a"
    for T in T_list:
        tau_vals = []
        gd_vals = []
        for gd in target_gdot_array:
            tau_val = get_array(label, gd, "tau")
            T_vals = get_array(label, gd, "T")
            idx = np.where(np.isclose(T_vals, T))[0]
            if len(idx) > 0:
                tau_here = tau_val[idx[0]]
                if np.isfinite(tau_here) and tau_here > 0:
                    tau_vals.append(tau_here)
                    gd_vals.append(gd)
        if len(tau_vals) >= 3:
            fit = np.polyfit(np.log(gd_vals), np.log(tau_vals), 1)
            m_global = fit[0]
        else:
            m_global = np.nan
        global_m_rows.append({
            "model": label,
            "T": T,
            "m_global": m_global,
        })

# ============================================================
# Compute activation volume V from tau vs kTlog(gdot)
# ============================================================
# BUG FIX: same three issues as above, fixed the same way.
activation_rows = []
for use_athermal in [True, False]:
    label = "with_tau_a" if use_athermal else "no_tau_a"
    for T in T_list:
        tau_vals = []
        x_vals_eV = []
        for gd in target_gdot_array:
            tau_arr = get_array(label, gd, "tau")
            T_vals = get_array(label, gd, "T")
            idx = np.where(np.isclose(T_vals, T))[0]
            if len(idx) > 0:
                tau_here = tau_arr[idx[0]]
                if np.isfinite(tau_here) and tau_here > 0:
                    # x = kB T ln(gdot), in eV
                    x_here_eV = kB_eV * T * np.log(gd)
                    tau_vals.append(tau_here)  # Pa
                    x_vals_eV.append(x_here_eV)  # eV
        if len(tau_vals) >= 3:
            # Fit tau = slope*x + intercept  (tau in Pa, x in eV)
            slope_Pa_per_eV, intercept_Pa = np.polyfit(x_vals_eV, tau_vals, 1)
            slope_Pa_per_J = slope_Pa_per_eV / eV_to_J
            Vstar_m3 = 1.0 / slope_Pa_per_J
            Vstar_b3 = Vstar_m3 / b**3
        else:
            slope_Pa_per_eV = np.nan
            intercept_Pa = np.nan
            Vstar_m3 = np.nan
            Vstar_b3 = np.nan
        activation_rows.append({
            "model": label,
            "T": T,
            "slope_Pa_per_eV": slope_Pa_per_eV,
            "intercept_Pa": intercept_Pa,
            "Vstar_m3": Vstar_m3,
            "Vstar_b3": Vstar_b3,
        })

# Vanaja et al. (JNM 424, 2012) — sigma_s, MPa; rates EQUAL the swept rates
VANAJA_T_K = np.array([300, 373, 423, 473, 523, 573, 623, 673, 723, 773, 823, 873])
VANAJA_RATES = np.array([3e-5, 3e-4, 3e-3])
VANAJA_SIGMA_S = np.array([
    [655, 720, 750],
    [620, 665, 670],
    [np.nan, 660, np.nan],
    [565, 585, 585],
    [560, 600, 580],
    [520, 570, 535],
    [515, 540, 530],
    [505, 515, 500],
    [450, 465, 480],
    [400, 440, 470],
    [350, 400, 440],
    [235, 305, 365],
])

# ============================================================
# Plot solved total tau and effective tau
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(target_gdot_array)))

for i, gd in enumerate(target_gdot_array):
    T_a = get_array("with_tau_a", gd, "T")
    tau_a_total = get_array("with_tau_a", gd, "tau")
    tau_a_eff = get_array("with_tau_a", gd, "tau_eff")
    # athermal component tau_a(T), added so legend1's dash-dot entry is real
    tau_a_athermal = np.array([tau_a_of_T_gdot(T, gd) for T in T_a])

    ax.plot(T_a, tau_a_total / 1e6, "-", color=colors[i],
            label=fr"total $\tau$, $\dot{{\gamma}}={gd:.0e}$")
    ax.plot(T_a, tau_a_athermal / 1e6, "-.", color=colors[i])

# ============================================================
# Vanaja et al. JNM 424, 2012 data
# sigma_s is axial MPa, so divide by 3 to compare to shear stress
# ============================================================
vanaja_markers = ["o", "s", "D"]

for j, sr in enumerate(VANAJA_RATES):
    y = VANAJA_SIGMA_S[:, j] / 3.0
    mask = ~np.isnan(y)

    # match Vanaja strain rate to nearest model color
    idx = np.argmin(np.abs(np.log10(target_gdot_array) - np.log10(sr)))
    c = colors[idx]

    ax.plot(
        VANAJA_T_K[mask],
        y[mask],
        linestyle="None",
        marker=vanaja_markers[j],
        markersize=9,
        markeredgewidth=2.0,
        color=c,
        label=fr"Vanaja, $\dot{{\epsilon}}={sr:.0e}$ s$^{{-1}}$"
    )

# ============================================================
# Legends
# ============================================================

# Legend 1: line styles
style_handles = [
    Line2D([0], [0], color="k", linestyle="-", label=r"total $\tau$"),
    Line2D([0], [0], color="k", linestyle="-.", label=r"$\tau_a(T)$")
]
legend1 = ax.legend(
    handles=style_handles,
    loc="upper right",
    fontsize=9,
    title="Stress components"
)
ax.add_artist(legend1)

# Legend 2: strain-rate colors
color_handles = [
    Line2D(
        [0], [0],
        color=colors[i],
        lw=2,
        label=fr"$\dot{{\gamma}}={gd:.0e}$ s$^{{-1}}$"
    )
    for i, gd in enumerate(target_gdot_array)
]
legend2 = ax.legend(
    handles=color_handles,
    loc="center right",
    fontsize=9,
    title="Model strain rate"
)
ax.add_artist(legend2)

# Legend 3: experimental datasets
exp_handles = []
for j, sr in enumerate(VANAJA_RATES):
    idx = np.argmin(np.abs(np.log10(target_gdot_array) - np.log10(sr)))
    c = colors[idx]

    exp_handles.append(
        Line2D(
            [0], [0],
            color=c,
            marker=vanaja_markers[j],
            linestyle="None",
            markersize=9,
            markeredgewidth=2.0,
            label=fr"Vanaja, $\dot{{\epsilon}}={sr:.0e}$ s$^{{-1}}$"
        )
    )

legend3 = ax.legend(
    handles=exp_handles,
    loc="lower left",
    fontsize=9,
    title="Experimental data"
)
ax.add_artist(legend3)

ax.set_xlabel("Temperature [K]", fontsize=14)
ax.set_ylabel("Stress [MPa]", fontsize=14)
ax.grid(True, alpha=0.3)

fig.tight_layout()


# ============================================================
# Plot local and global m(T)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(target_gdot_array)))
T_global_a = np.array([r["T"] for r in global_m_rows if r["model"] == "with_tau_a"])
m_global_a = np.array([r["m_global"] for r in global_m_rows if r["model"] == "with_tau_a"])
ax.plot(T_global_a, m_global_a, "k-", linewidth=3, label=r"global fit $m$")
ax.set_xlabel("Temperature [K]", fontsize=14)
ax.set_ylabel(r"m", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()


# ============================================================
# Plot global V(T)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(target_gdot_array)))
T_global_a = np.array([r["T"] for r in activation_rows if r["model"] == "with_tau_a"])
V_global_a = np.array([r["Vstar_b3"] for r in activation_rows if r["model"] == "with_tau_a"])
ax.plot(T_global_a, V_global_a, "k-", linewidth=3, label=r"global fit $V$")
ax.set_xlabel("Temperature [K]", fontsize=14)
ax.set_ylabel(r"V", fontsize=14)
ax.set_ylim(0, 300)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()