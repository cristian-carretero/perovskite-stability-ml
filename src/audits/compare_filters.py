import pandas as pd

audit = pd.read_parquet("data/filtered/outdoor/02_filtering_audit.parquet")

print("Columnas:", audit.columns.tolist())
print("Total curvas:", len(audit))
print("Válidas:", int(audit["is_curve_valid"].sum()))
print()

print("=== Rechazos por regla ===")
for c in ["rej_night", "rej_unphysical", "rej_vspan_low", "rej_snr_low", "rej_spike"]:
    print(f"  {c:20s}  {audit[c].sum():>8,}  ({audit[c].mean():>6.2%})")

print("\n=== Distribución de scores (todas las curvas) ===")
print(audit[["v_span_ratio", "snr_i", "spike_score", "v_span", "i_span", "hysteresis_index"]].describe().T)

print("\n=== Scores separados por validez ===")
for c in ["v_span_ratio", "snr_i", "spike_score"]:
    print(f"\n{c}:")
    print(audit.groupby("is_curve_valid")[c].describe().round(4))

print("\n=== Por célula (rechazos por regla) ===")
per_cell = audit.groupby("cell_name").agg(
    total=("is_curve_valid", "count"),
    valid=("is_curve_valid", "sum"),
    rej_night=("rej_night", "sum"),
    rej_unphysical=("rej_unphysical", "sum"),
    rej_vspan=("rej_vspan_low", "sum"),
    rej_snr=("rej_snr_low", "sum"),
    rej_spike=("rej_spike", "sum"),
    v_span_ratio_med=("v_span_ratio", "median"),
    snr_i_med=("snr_i", "median"),
    spike_score_med=("spike_score", "median"),
)
print(per_cell.round(4).to_string())