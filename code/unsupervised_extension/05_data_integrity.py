"""Step 5: data-integrity checks on the patient linkage across files.

For the same patient and tumour, the segmented volume should agree across the
T1, T2 and ADC segmentations, the radiomic tumour volume should track the
radiologist's maximum diameter, and the whole-tumour mean ADC should track the
radiologist-measured ADC value. Spearman correlations are reported within
each acquisition batch (run after 01_harmonization.py, which defines batch).
"""

import pandas as pd

from common import DATA, TAB, load_modalities


def main():
    raw, ids = load_modalities()
    batch = pd.read_csv(TAB / "acquisition_batch.csv", index_col=0)["batch"]
    ps = pd.read_excel(DATA / "post_survey.xlsx").set_index("case").sort_index()
    vol = {m: raw[m]["VolumeCount"] for m in raw}
    rows = []
    for b in ["C1", "C2a", "C2b", "All"]:
        idx = batch.index if b == "All" else batch.index[batch == b]

        def rho(x, y):
            return x.loc[idx].corr(y.loc[idx], method="spearman")

        rows.append({
            "batch": b, "n": len(idx),
            "volume T1 vs T2": rho(vol["T1"], vol["T2"]),
            "volume T1 vs ADC": rho(vol["T1"], vol["ADC"]),
            "volume T2 vs ADC": rho(vol["T2"], vol["ADC"]),
            "T2 volume vs max diameter": rho(vol["T2"], ps["最大径"]),
            "ADC mean vs radiologist ADC": rho(raw["ADC"]["MeanValue"],
                                               ps["ADC值"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "data_integrity.csv", index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
