"""Collects ready condition tables (real and generated spines) from a folder of analysis runs into ONE markdown file.

Reads only small csv tables (no recomputation). The analysis folder is the ``analysis`` folder of the transfer archive
(``transfer_2026-11-19/analysis``) or ``runs/analysis/neuron-model``. LabID (health / treatment) is left out on purpose.

    python scripts/neuron_model/summarize_condition_tables.py --analysis legacy/output_win/transfer_2026-11-19/analysis \
        --out docs/neuron-model/sprints/19.11-conditions-tables.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["OpenAngle", "CVD", "AverageDistance", "Length", "LengthVolumeRatio", "LengthAreaRatio", "Area",
           "JunctionArea", "Volume", "ConvexHullVolume", "ConvexHullRatio"]


def fmt(x, nd=3):
    if pd.isna(x):
        return ""
    if isinstance(x, (bool,)) or str(x) in ("True", "False"):
        return "да" if str(x) == "True" else "нет"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer() and abs(x) > 99):
        return f"{int(x)}"
    if isinstance(x, float):
        a = abs(x)
        return f"{x:.2e}" if (a != 0 and (a < 1e-3 or a >= 1e5)) else f"{x:.{nd}g}"
    return str(x)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def order(df: pd.DataFrame, col: str = "metric") -> pd.DataFrame:
    key = {m: i for i, m in enumerate(METRICS)}
    return df.assign(_o=df[col].map(key).fillna(99)).sort_values("_o").drop(columns="_o")


def pct(x):
    return None if pd.isna(x) else round(100 * x, 1)


def read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def real_pair(df: pd.DataFrame, a: str, b: str, label_a: str, label_b: str) -> pd.DataFrame:
    d = order(df)
    return pd.DataFrame({"метрика": d["metric"], f"медиана {label_a}": d["median_a"], f"медиана {label_b}": d["median_b"],
                         "отличие, %": d["rel_diff_a_vs_b"].map(pct), "δ Клиффа": d["cliffs_delta"],
                         "q (BH)": d["q_bh"], "значимо": d["significant"]})


def generated_block(folder: Path, labels: list[str], types: list[str], real_label_for: dict[str, str]) -> list[str]:
    out = []
    # A: reproduced differences
    rows = []
    for lab in labels:
        c = read(folder / f"concordance_{lab}.csv")
        if c is None:
            continue
        vc = c["status"].value_counts()
        rows.append({"набор": lab, "воспроизведено": int(vc.get("reproduced", 0)), "не воспроизведено": int(vc.get("not_reproduced", 0)),
                     "неверный знак": int(vc.get("wrong_sign", 0)), "лишний эффект": int(vc.get("spurious_effect", 0)),
                     "мало образцов": int(vc.get("underpowered", 0)), "значимых реальных": int(c["real_sig"].sum()), "всего метрик": len(c)})
    if rows:
        out += ["**А. Воспроизводятся ли различия типов** (счёт метрик по статусам, реальная разница против сгенерированной):", "", md(pd.DataFrame(rows)), ""]
    # B: same type
    rows = []
    for typ in types:
        for lab in labels:
            f = folder / f"same_type_{typ}@{lab}_vs_{typ}.csv"
            s = read(f)
            if s is None:
                continue
            vc = s["verdict"].value_counts()
            rows.append({"тип": typ, "набор": lab, "отличается": int(vc.get("differs", 0)),
                         "эквивалентность доказана": int(vc.get("equivalent", 0)),
                         "не отличим, эквивалентность не доказана": int(vc.get("indistinguishable_equivalence_not_shown", 0)),
                         "всего метрик": len(s)})
    if rows:
        out += ["**Б. Совпадают ли сгенерированные и реальные шипики одного типа** (число метрик из 11 по вердикту):", "", md(pd.DataFrame(rows)), ""]
    return out


def per_metric_generated(folder: Path, lab: str, ctrl: str, types: list[str]) -> list[str]:
    c = read(folder / f"concordance_{lab}.csv")
    cc = read(folder / f"concordance_{ctrl}.csv")
    if c is None:
        return []
    d = order(c)
    t = pd.DataFrame({"метрика": d["metric"], "реальная разница, log": d["lr_real"], "q реальных": d["real_q"],
                      f"сгенерированная ({lab}), log": d["lr_gen"], f"статус ({lab})": d["status"]})
    if cc is not None:
        t[f"статус (реальные через цепочку, {ctrl})"] = order(cc)["status"].values
    for typ in types:
        s = read(folder / f"same_type_{typ}@{lab}_vs_{typ}.csv")
        if s is not None:
            t[f"{typ}: реальные против сгенерированных, вердикт"] = order(s)["verdict"].str.replace("indistinguishable_equivalence_not_shown", "не отличим, экв. не доказана").values
    return [f"Подробно по метрикам (набор {lab}):", "", md(t), ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--global-tables", default="2026-11-17_042234_global-tables")
    ap.add_argument("--by-condition", default="2026-11-01_184009_morphology-by-condition")
    ap.add_argument("--report-binary", default="2026-11-16_conditional-report-pw8-binary")
    ap.add_argument("--report-types", default="2026-11-16_conditional-report-pw8-types")
    a = ap.parse_args()
    A = a.analysis
    L = ["# Различия морфометрик шипиков по условиям: реальные и сгенерированные",
         "", "Сводка собрана скриптом `scripts/neuron_model/summarize_condition_tables.py` из готовых таблиц (пересчёта нет). "
         "LabID (здоровье, лечение) не включён. Единица вывода всегда нейрон, перестановочные тесты между нейронами, "
         "поправка Бенджамини–Хохберга (q). Метрики: " + ", ".join(METRICS) + ".", ""]

    # ---- real, Minnie65
    bc = A / a.by_condition
    L += ["## 1. Реальные шипики Minnie65 (папка " + a.by_condition + ")", ""]
    b = read(bc / "cell_type_binary.csv")
    if b is not None:
        L += [f"### 1.1. Двоичный тип клетки: тормозные ({int(b['n_neurons_a'].iloc[0])} нейронов, {int(b['n_spines_a'].iloc[0])} шипиков) против возбуждающих "
              f"({int(b['n_neurons_b'].iloc[0])} нейронов, {int(b['n_spines_b'].iloc[0])} шипиков)", "",
              f"Значимы {int(b['significant'].sum())} из {len(b)} метрик.", "", md(real_pair(b, 'inhibitory', 'excitatory', 'тормозные', 'возбуждающие')), ""]
    o = read(bc / "cell_type_omnibus.csv")
    p = read(bc / "cell_type.csv")
    if o is not None:
        d = order(o)
        t = pd.DataFrame({"метрика": d["metric"], "типов": d["n_types"], "нейронов": d["n_neurons"], "q (общий тест)": d["q_bh"], "значимо": d["significant"]})
        if p is not None:
            cnt = p.groupby("metric")["significant"].agg(["sum", "size"])
            t["значимых пар типов"] = t["метрика"].map(lambda m: f"{int(cnt.loc[m, 'sum'])} из {int(cnt.loc[m, 'size'])}" if m in cnt.index else "")
        L += ["### 1.2. Подробный тип клетки (типы с ≥ 10 нейронов, без Unsure)", "",
              f"Общий тест значим по {int(o['significant'].sum())} из {len(o)} метрик"
              + (f", значимых пар «метрика × пара типов»: {int(p['significant'].sum())} из {len(p)}" if p is not None else "") + ".", "", md(t), ""]
    c = read(bc / "compartment.csv")
    if c is not None:
        rows = []
        for (x, y), g in c.groupby(["a", "b"]):
            if not g["n_neurons_paired"].max() > 0:
                continue
            rows.append({"сравнение": f"{x} против {y}", "нейронов в паре": int(g["n_neurons_paired"].max()), "значимых метрик": f"{int(g['significant'].sum())} из {len(g)}",
                         "медианное парное отличие, % (по модулю, мин–макс)": f"{pct(g['median_paired_rel_diff'].abs().min())}–{pct(g['median_paired_rel_diff'].abs().max())}"})
        L += ["### 1.3. Компартмент (внутри нейрона, только возбуждающие)", "", md(pd.DataFrame(rows)), ""]
        ab = c[(c["a"] == "apical") & (c["b"] == "basal")]
        if len(ab):
            d = order(ab)
            L += ["Апикальные против базальных по метрикам:", "", md(pd.DataFrame({"метрика": d["metric"], "парное отличие, %": d["median_paired_rel_diff"].map(pct),
                                                                              "доля нейронов, где апикальные больше": d["share_neurons_a_gt_b"], "q (BH)": d["q_bh"], "значимо": d["significant"]})), ""]

    # ---- global tables: species, H01 within
    gt = read(A / a.global_tables / "results_spine.csv")
    L += ["## 2. Вид и условия внутри H01 (глобальный отчёт, папка " + a.global_tables + ")", ""]
    if gt is not None:
        sp = gt[(gt["dataset"] == "h01 vs minnie65") & (gt["factor"] == "species")]
        if len(sp):
            d = order(sp)
            L += ["### 2.1. Вид: человек (H01) против мыши (Minnie65)", "",
                  "**Только описательно:** вид смешан с датасетом, областью коры и способом съёмки (причинный вывод невозможен). "
                  f"Значимы {int(d['significant'].sum())} из {len(d)} метрик. Нейронов: человек {int(d['n_neurons_a'].iloc[0])} ({int(d['n_spines_a'].iloc[0])} шипиков), мышь {int(d['n_neurons_b'].iloc[0])} ({int(d['n_spines_b'].iloc[0])} шипиков).", "",
                  md(pd.DataFrame({"метрика": d["metric"], "медиана человека": d["median_a"], "медиана мыши": d["median_b"],
                                   "отличие, %": d["rel_diff_a_vs_b"].map(pct), "δ Клиффа": d["cliffs_delta"], "q (BH)": d["q_bh"], "значимо": d["significant"]})), ""]
        hb = gt[(gt["dataset"] == "h01") & (gt["factor"] == "cell_type_binary")]
        if len(hb):
            d = order(hb)
            L += [f"### 2.2. H01: двоичный тип (тормозные {int(d['n_neurons_a'].iloc[0])} нейронов, {int(d['n_spines_a'].iloc[0])} шипиков, против возбуждающих "
                  f"{int(d['n_neurons_b'].iloc[0])} нейронов, {int(d['n_spines_b'].iloc[0])} шипиков)", "",
                  f"Значимы {int(d['significant'].sum())} из {len(d)} метрик.", "",
                  md(pd.DataFrame({"метрика": d["metric"], "медиана тормозных": d["median_a"], "медиана возбуждающих": d["median_b"],
                                   "отличие, %": d["rel_diff_a_vs_b"].map(pct), "δ Клиффа": d["cliffs_delta"], "q (BH)": d["q_bh"], "значимо": d["significant"]})), ""]
        hc = gt[(gt["dataset"] == "h01") & (gt["factor"] == "compartment")]
        if len(hc):
            d = order(hc)
            L += [f"### 2.3. H01: компартмент ({hc['a'].iloc[0]} против {hc['b'].iloc[0]}, внутри нейрона; {int(d['n_neurons_paired'].max())} нейронов в паре, "
                  f"шипиков {int(d['n_spines_a'].iloc[0])} и {int(d['n_spines_b'].iloc[0])})", "",
                  f"Значимы {int(hc['significant'].sum())} из {len(hc)} метрик, медианное парное отличие по модулю "
                  f"{pct(hc['median_paired_rel_diff'].abs().min())}–{pct(hc['median_paired_rel_diff'].abs().max())} %.", "",
                  md(pd.DataFrame({"метрика": d["metric"], "парное отличие, %": d["median_paired_rel_diff"].map(pct),
                                   "доля нейронов, где апикальные больше": d["share_neurons_a_gt_b"], "q (BH)": d["q_bh"], "значимо": d["significant"]})), ""]
        cons = read(A / a.global_tables / "consistency_summary_spine.csv")
        if cons is not None:
            L += ["### 2.4. Согласованность Minnie65 и H01 (одинаков ли знак и порядок эффектов)", "", md(cons), ""]

    # ---- generated
    L += ["## 3. Сгенерированные шипики (MoGen v3, цепочка Poisson d8_pw8)", "",
          "Наборы: `g1.25-pw0` и `g1.25-pw8` — сгенерированные с усилением условия 1,25 без цепочки Poisson (pw0) и с ней (pw8, финальная); "
          "`real-pw0` и `real-pw8` — **реальные** шипики тех же нейронов через те же цепочки (нижняя граница: лучше этого генератор быть не может). "
          "Вердикты теста Б: `differs` — отличаются, `equivalent` — эквивалентность в границах 0,2 σ доказана, "
          "«не отличим, эквивалентность не доказана» — отличий не найдено, но и доказать их отсутствие не удалось.", ""]
    rb = A / a.report_binary
    L += ["### 3.1. Двоичный тип (возбуждающие и тормозные)", ""]
    L += generated_block(rb, ["g1.25-pw0", "g1.25-pw8", "real-pw0", "real-pw8"], ["excitatory", "inhibitory"], {})
    L += per_metric_generated(rb, "g1.25-pw8", "real-pw8", ["excitatory", "inhibitory"])
    rt = A / a.report_types
    L += ["### 3.2. Подробный тип (4P против 6P-CT; усиление 1,5)", "",
          "В отчёте оставлена пара 4P / 6P-CT, BPC (4 проверочных нейрона) исключён из интерпретации, у MC нет проверочных нейронов.", ""]
    L += generated_block(rt, ["g1.5-pw0", "g1.5-pw8", "real-pw0", "real-pw8"], ["4P", "6P-CT"], {})
    L += per_metric_generated(rt, "g1.5-pw8", "real-pw8", ["4P", "6P-CT"])
    L += ["## 4. Что не выполнено", "",
          "- Компартмент у сгенерированных шипиков: модель на него не обучалась, реальный эффект 1–2 %.",
          "- Вид у сгенерированных шипиков: v4 обучена, но генерация по виду не запускалась (цепочка AE).",
          "- Здоровье и лечение (LabID): исключены из этой сводки по решению пользователя, прежний глобальный прогон устарел (единица папка, поправка z 0,7).", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
