"""Compact markdown summary of the conditional-generation reports with the floor from VALIDATION neurons (chain AH, 2026-11-20).

Reads only the small csv tables of the report folders (no recomputation):
  <dir>/<date>_conditional-report-v4-mouse-valfloor   binary type, Minnie65 (v4, v3, real-pw8val)
  <dir>/<date>_conditional-report-v4-types-valfloor   detailed types, Minnie65 (v4 x 4 types, v3 x 2 types, real-pw8val)
  <dir>/<date>_conditional-report-v4-human-valfloor   binary type, H01
  <dir>/<date>_species-generation-v4-valfloor          species table (compare_species_generation.py)
Differences are given in % (exp(log) - 1): the log ratio of medians stays in the csv.

    python scripts/neuron_model/summarize_valfloor_reports.py --dir legacy/output_win/2026-11-20-valfloor \
        --out docs/neuron-model/sprints/21.11-conditions-valfloor-tables.md
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["OpenAngle", "CVD", "AverageDistance", "Length", "LengthVolumeRatio", "LengthAreaRatio", "Area",
           "JunctionArea", "Volume", "ConvexHullVolume", "ConvexHullRatio"]
STATUS_RU = {"reproduced": "воспроизведено", "consistent": "согласуется", "wrong_sign": "неверный знак",
             "amplitude_over": "амплитуда завышена", "amplitude_under": "амплитуда занижена", "not_reproduced": "не воспроизведено",
             "underpowered": "мало образцов", "spurious_effect": "лишний эффект", "outside_real_ci": "вне интервала реальных", "no_data": "нет данных"}
VERDICT_RU = {"differs": "отличаются", "equivalent": "эквивалентны", "indistinguishable_equivalence_not_shown": "не отличимы", "no_data": "нет данных"}
SET_RU = {"v4-pw8": "v4", "v3-pw8": "v3", "real-pw8val": "реальные через цепочку (пол)"}


def read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def order(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("metric").reindex([m for m in METRICS if m in set(df["metric"])]).reset_index()


def pc(x) -> str:
    """log ratio -> signed percent string."""
    if x is None or not np.isfinite(x):
        return ""
    return f"{100 * np.expm1(x):+.0f}"


def num(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}g}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def find(root: Path, suffix: str) -> Path | None:
    hits = sorted(root.glob(f"*{suffix}"))
    return hits[-1] if hits else None


def a_counts(c: pd.DataFrame) -> dict:
    vc = c["status"].value_counts()
    real_sig = int(c["real_sig"].sum())
    rep = int(vc.get("reproduced", 0))
    return {"значимых реальных": real_sig, "воспроизведено": rep,
            "не воспроизведено": int(vc.get("not_reproduced", 0)), "неверный знак": int(vc.get("wrong_sign", 0)),
            "амплитуда не та": int(vc.get("amplitude_over", 0) + vc.get("amplitude_under", 0)),
            "лишний эффект": int(vc.get("spurious_effect", 0) + vc.get("outside_real_ci", 0)),
            "мало образцов": int(vc.get("underpowered", 0)), "согласовано из 11": int(c["concordant"].sum())}


def b_counts(s: pd.DataFrame) -> dict:
    vc = s["verdict"].value_counts()
    d = order(s)
    big = d.loc[d["log_ratio"].abs().idxmax()]
    return {"отличаются": int(vc.get("differs", 0)), "эквивалентны": int(vc.get("equivalent", 0)),
            "не отличимы": int(vc.get("indistinguishable_equivalence_not_shown", 0)),
            "медиана |d|, %": f"{100 * np.expm1(np.median(np.abs(d['log_ratio']))):.0f}",
            "наибольшее отклонение": f"{big['metric']} {pc(big['log_ratio'])} %"}


def binary_section(folder: Path, title: str, labels: list[str], real_name: str) -> list[str]:
    out = [f"## {title}", "", f"Папка `{folder.name}`. a = тормозные, b = возбуждающие. Реальные шипики: {real_name}.", ""]
    r = read(folder / "real_effect.csv")
    if r is not None:
        r = order(r)
        out += [f"Реальных: тормозные {int(r['n_neurons_a'].iloc[0])} нейронов / {int(r['n_spines_a'].iloc[0])} шипиков, "
                f"возбуждающие {int(r['n_neurons_b'].iloc[0])} / {int(r['n_spines_b'].iloc[0])}; значимы {int(r['significant'].sum())} из {len(r)} метрик.", ""]
    rows = []
    for lab in labels:
        c = read(folder / f"concordance_{lab}.csv")
        if c is not None:
            rows.append({"набор": SET_RU.get(lab, lab), **a_counts(c)})
    if rows:
        out += ["**А. Воспроизводится ли различие тормозных и возбуждающих** (число метрик):", "", md(pd.DataFrame(rows)), ""]
    rows = []
    for typ in ("excitatory", "inhibitory"):
        for lab in labels:
            s = read(folder / f"same_type_{typ}@{lab}_vs_{typ}.csv")
            if s is None:
                s = read(folder / f"same_type_{typ}@{lab}_vs_mixture.csv")
                ref = "смесь типов"
            else:
                ref = typ
            if s is not None:
                rows.append({"тип": typ, "набор": SET_RU.get(lab, lab), "эталон": ref, **b_counts(s)})
    if rows:
        out += ["**Б. Совпадают ли сгенерированные шипики с реальными того же типа** (число метрик из 11; d = отношение медиан сгенерированные / реальные − 1):",
                "", md(pd.DataFrame(rows)), ""]
    # per metric
    if r is not None:
        t = pd.DataFrame({"метрика": r["metric"], "реальные: торм. / возб., %": r["log_ratio"].map(pc), "q реальных": r["q_bh"].map(num)})
        for lab in labels:
            c = read(folder / f"concordance_{lab}.csv")
            if c is None:
                continue
            c = order(c)
            t[f"{SET_RU.get(lab, lab)}: торм. / возб., %"] = c["lr_gen"].map(pc).values
            t[f"{SET_RU.get(lab, lab)}: статус"] = c["status"].map(STATUS_RU).fillna(c["status"]).values
        for typ in ("excitatory", "inhibitory"):
            for lab in labels:
                s = read(folder / f"same_type_{typ}@{lab}_vs_{typ}.csv")
                if s is not None:
                    s = order(s)
                    t[f"d {typ[:4]}. {SET_RU.get(lab, lab).split(' ')[0]}, %"] = [f"{pc(x)}{'*' if v == 'differs' else ''}" for x, v in zip(s["log_ratio"], s["verdict"])]
        out += ["**По метрикам.** «торм. / возб.» = отношение медиан − 1. d = медиана сгенерированных / медиана реальных того же типа − 1, * = отличаются (q < 0,05).", "", md(t), ""]
    return out


def types_section(folder: Path) -> list[str]:
    out = ["## 2. Мышь, подробный тип (4P, 6P-CT, 23P, 6P-IT; усиление 1,5)", "",
           f"Папка `{folder.name}`. Реальные шипики проверочных нейронов `splits_v1_h01`. v3 есть только для 4P и 6P-CT.", ""]
    types = ["23P", "4P", "6P-CT", "6P-IT"]
    neurons = {}
    rows = []
    for a, b in itertools.combinations(types, 2):
        r = read(folder / f"real_effect_{a}_{b}.csv")
        if r is None:
            continue
        neurons[a], neurons[b] = int(r["n_neurons_a"].iloc[0]), int(r["n_neurons_b"].iloc[0])
        row = {"пара": f"{a} / {b}", "значимых реальных": int(r["significant"].sum())}
        for lab in ("real-pw8val", "v4-pw8", "v3-pw8"):
            c = read(folder / f"concordance_{lab}_{a}_{b}.csv")
            if c is None:
                row[SET_RU[lab].split(" ")[0]] = ""
                continue
            k = a_counts(c)
            row[SET_RU[lab].split(" ")[0]] = f"{k['воспроизведено']} воспр. / {k['согласовано из 11']} согл. / {k['неверный знак']} неверн. знак"
        rows.append(row)
    out += ["Нейронов в проверочной части: " + ", ".join(f"{t} {n}" for t, n in neurons.items()) + ".", "",
            "**А. Воспроизводятся ли различия пар типов.** Ячейка: воспроизведено реальных эффектов / согласовано метрик из 11 / неверный знак. "
            "Столбец «реальные» = реальные шипики через ту же цепочку (пол).", "", md(pd.DataFrame(rows)), ""]
    rows = []
    for typ in types:
        for lab in ("v4-pw8", "v3-pw8", "real-pw8val"):
            s = read(folder / f"same_type_{typ}@{lab}_vs_{typ}.csv")
            if s is not None:
                rows.append({"тип": typ, "набор": SET_RU[lab], **b_counts(s)})
    out += ["**Б. Совпадают ли сгенерированные шипики с реальными того же типа** (число метрик из 11):", "", md(pd.DataFrame(rows)), ""]
    for a, b in (("23P", "6P-CT"), ("4P", "6P-CT")):
        r = read(folder / f"real_effect_{a}_{b}.csv")
        if r is None:
            continue
        r = order(r)
        t = pd.DataFrame({"метрика": r["metric"], f"реальные: {a} / {b}, %": r["log_ratio"].map(pc), "q реальных": r["q_bh"].map(num)})
        for lab in ("real-pw8val", "v4-pw8", "v3-pw8"):
            c = read(folder / f"concordance_{lab}_{a}_{b}.csv")
            if c is None:
                continue
            c = order(c)
            nm = SET_RU[lab].split(" ")[0]
            t[f"{nm}: {a} / {b}, %"] = c["lr_gen"].map(pc).values
            t[f"{nm}: статус"] = c["status"].map(STATUS_RU).fillna(c["status"]).values
        out += [f"**Пара {a} / {b} по метрикам** (отношение медиан − 1):", "", md(t), ""]
    return out


def species_section(folder: Path) -> list[str]:
    out = ["## 4. Вид (мышь / человек) внутри двоичного типа", "", f"Папка `{folder.name}`. Только описательно: вид смешан с датасетом. "
           "Реальные: проверочные нейроны (мышь возб. 65 / торм. 8, человек возб. 6 / торм. 3).", ""]
    ps = read(folder / "per_set.csv")
    if ps is not None:
        sh = ps.groupby(["label", "value"])["closer_to_own"].mean().unstack(0)
        sh = sh.rename(columns=lambda c: SET_RU.get(c, c)).map(lambda x: f"{100 * x:.0f} %")
        out += ["**Доля метрик, по которым набор ближе (W₁/σ) к реальным своей группы, чем к остальным:**", "", md(sh.reset_index().rename(columns={"value": "группа"})), ""]
        size = ps[ps["metric"].isin(["Length", "Area", "Volume", "JunctionArea"])]
        t = size.pivot_table(index=["value", "metric"], columns="label", values="gen_over_real").rename(columns=lambda c: SET_RU.get(c, c))
        t = t.map(lambda x: f"×{x:.2f}").reset_index().rename(columns={"value": "группа", "metric": "метрика"})
        out += ["**Размер: медиана сгенерированных / медиана реальных той же группы:**", "", md(t), ""]
    pr = read(folder / "pairs.csv")
    if pr is not None:
        rows = []
        for (lab, a, b), g in pr.groupby(["label", "a", "b"]):
            s = g["share_of_real_difference"].replace([np.inf, -np.inf], np.nan).dropna()
            same = np.sign(np.log(g["ratio_gen"])) == np.sign(np.log(g["ratio_real"]))
            rows.append({"набор": SET_RU.get(lab, lab), "a / b": f"{a} / {b}", "знак совпал": f"{int(same.sum())} из {len(g)}",
                         "медиана доли реальной разницы": f"{s.median():.2f}"})
        out += ["**Различия между группами: воспроизведён ли знак и какая доля реальной разницы получена** (1 = вся, 0 = нет):", "",
                md(pd.DataFrame(rows)), ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L = ["# Условная генерация шипиков: отчёты с полом из проверочных нейронов (цепочка AH, 20.11.2026)", "",
         "Собрано `scripts/neuron_model/summarize_valfloor_reports.py` из готовых таблиц. Все сгенерированные и «реальные через цепочку» наборы "
         "прошли одну цепочку Poisson d8_pw8; реальные шипики, с которыми идёт сравнение, взяты у проверочных нейронов `splits_v1_h01` "
         "(генератор их не видел). Пол = реальные шипики тех же проверочных нейронов через ту же цепочку: лучше пола генератор быть не может. "
         "Тест А: перестановки между нейронами и бутстрэп нейронов, BH по 11 метрикам. Тест Б: d = ln медиан, кластерный бутстрэп по нейронам, "
         "BH, TOST с границей 0,2 σ. Одно зерно генерации, по 300 сгенерированных и 500 реальных через цепочку на набор.", ""]
    f = find(a.dir, "conditional-report-v4-mouse-valfloor")
    if f:
        L += binary_section(f, "1. Мышь (Minnie65), двоичный тип", ["v4-pw8", "v3-pw8", "real-pw8val"], "проверочные нейроны Minnie65")
    f = find(a.dir, "conditional-report-v4-types-valfloor")
    if f:
        L += types_section(f)
    f = find(a.dir, "conditional-report-v4-human-valfloor")
    if f:
        L += binary_section(f, "3. Человек (H01), двоичный тип", ["v4-pw8", "real-pw8val"],
                            "проверочные нейроны H01 без шипиков без шейки; тормозных нейронов 3, поэтому пул тормозных отчёт пропускает, "
                            "тормозные сравниваются со смесью типов, а тест А неинформативен")
    f = find(a.dir, "species-generation-v4-valfloor")
    if f:
        L += species_section(f)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
