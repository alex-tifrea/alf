"""ICMD schema and names for the frozen acquisition benchmark."""

FEATURES = [
    "cu_wt_percent",
    "mg_wt_percent",
    "zn_wt_percent",
    "solution_temperature_c",
    "aging_temperature_c",
    "aging_duration_h",
]
PRIMARY = "yield_strength_mpa"
ALF_REVISION = "2002936c623d11bf58f33b8cfb198ecc59b4c4aa"

METHODS = [
    "uniform",
    "coreset",
    "bootstrap",
    "gp_std",
    "gp_pv",
    "conformal_width",
    "activation_pv",
    "activation_frozen_pv",
    "activation_random_pv",
]

NAMES = {
    "uniform": "Uniform",
    "coreset": "CoreSet (inputs)",
    "bootstrap": "Bootstrap uncertainty",
    "gp_std": "GP uncertainty",
    "gp_pv": "GP batch PV",
    "conformal_width": "Conformal width (CQR)",
    "activation_pv": "Activation-PV (refit MLP)",
    "activation_frozen_pv": "Activation-PV (seed-frozen MLP)",
    "activation_random_pv": "Activation-PV (random MLP)",
}
