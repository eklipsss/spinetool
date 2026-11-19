"""Collects ready condition tables (real and generated spines) from a folder of analysis runs into ONE markdown file.

Reads only small csv tables (no recomputation). The analysis folder is the ``analysis`` folder of the transfer archive
(``transfer_2026-11-19/analysis``) or ``runs/analysis/neuron-model``. LabID (health / treatment) is left out on purpose.

    python scripts/neuron_model/summarize_condition_tables.py --analysis legacy/output_win/transfer_2026-11-19/analysis \
        --out docs/neuron-model/sprints/19.11-conditions-tables.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ["OpenAngle", "CVD", "AverageDistance", "Length", "LengthVolumeRatio", "LengthAreaRatio", "Area",
           "JunctionArea", "Volume", "ConvexHullVolume", "ConvexHullRatio"]


STATUS_RU = {
    "reproduced": "воспроизведено", "consistent": "согласуется", "wrong_sign": "неверный знак",
    "amplitude_over": "амплитуда завышена", "amplitude_under": "амплитуда занижена", "not_reproduced": "не воспроизведено",
    "underpowered": "мало образцов", "spurious_effect": "лишний эффект", "outside_real_ci": "вне интервала реальных", "no_data": "нет данных",
}
VERDICT_RU = {
    "differs": "отличаются", "equivalent": "эквивалентны (доказано)",
    "indistinguishable_equivalence_not_shown": "не отличимы, эквивалентность не доказана", "no_data": "нет данных",
}

LEGEND = """**Как читать таблицу «Подробно по метрикам».** Сравниваются два типа клеток (a и b: в двоичном блоке a = тормозные, b = возбуждающие, в подробных типах a = 4P, b = 6P-CT).

| Столбец | Что это и как считается |
| --- | --- |
| метрика | одна из 11 морфометрик шипика |
| реальная разница, log и % | log = ln(медиана метрики у реальных шипиков типа a / медиана у реальных шипиков типа b), % = exp(log) − 1 (тот же показатель в процентах: 0,25 → +28 %, −0,33 → −28 %, 0,69 → +100 %). В таблицах разделов 1 и 2 «отличие, %» считается иначе: по медианам нейронных медиан, поэтому числа немного отличаются от этого столбца. Медианы по всем шипикам типа (не по нейронам); 0 = различия нет, +0,25 = у типа a метрика примерно на 28 % больше, −0,33 = примерно на 28 % меньше. Доверительный интервал (в таблицу не вынесен, используется в статусе) получен бутстрэпом по нейронам |
| q реальных | значимость реальной разницы. Тест на уровне нейронов: у каждого нейрона берётся медиана метрики, затем 10 000 раз метки типа случайно переставляются между нейронами и сравнивается разность медиан нейронных медиан; поправка Бенджамини–Хохберга по 11 метрикам. q < 0,05 значимо |
| сгенерированная (набор), log | то же, ln(медиана сгенерированных шипиков a / медиана сгенерированных b), только по сгенерированным наборам (или по реальным через цепочку, для набора real). Доверительный интервал получен бутстрэпом (1000 повторов) |
| статус | сравнение реальной и сгенерированной разницы (правила ниже) |
| статус (реальные через цепочку) | тот же статус, но «генератором» служат реальные шипики тех же типов, прошедшие ту же цепочку Poisson. Показывает, сколько несовпадений создаёт сама цепочка, а не генератор: лучше этого значения генератор быть не может |
| d, тип T | сама разность d (ln медианы сгенерированных − ln медианы реальных того же типа T); d > 0: сгенерированные больше реальных |
| \|d\|, σ (тип T) | точечная оценка смещения в стандартных отклонениях логарифма метрики (σ восстановлена из границы): |d| / sd. Не включает неопределённость; разность «граница без поправки − \|d\|, σ» = 1,645·SE / sd — вклад неопределённости оценки |
| граница без поправки, σ (тип T) | наименьшая граница эквивалентности (в стандартных отклонениях логарифма метрики по реальным шипикам), при которой эквивалентность метрики была бы доказана БЕЗ поправки на 11 метрик: (\\|d\\| + 1,645·SE) / sd. Это односторонняя оценка |d| с 95 % уверенностью, выраженная в σ |
| граница с поправкой, σ (тип T) | то же, но с поправкой Бенджамини–Хохберга по всем 11 метрикам (как в вердикте). Эквивалентность доказана при принятой границе 0,2 σ, если это значение ≤ 0,2. Читать нужно именно этот столбец вместе с вердиктом: значение между «без поправки» и 0,2 означает, что без поправки метрика прошла бы, а с поправкой нет |
| вердикт для типа a / типа b | другой вопрос: совпадают ли сгенерированные шипики данного типа с реальными шипиками того же типа. d = ln(медиана сгенерированных) − ln(медиана реальных того же типа). Бутстрэп по нейронам реальных шипиков (нейроны перевыбираются с возвращением) и независимый бутстрэп сгенерированной выборки; двусторонний p по процентилям, поправка Бенджамини–Хохберга по метрикам |

**Статусы** (колонки «статус», проверка воспроизведения реальной разницы):
- **воспроизведено**: реальная разница значима, у сгенерированных значима с тем же знаком, интервалы пересекаются;
- **согласуется**: реальная разница не значима, а сгенерированная лежит внутри интервала реальной (или тоже не значима и интервалы пересекаются), то есть шум генератора не наказывается;
- **неверный знак**: реальная разница значима, а у сгенерированных значима с противоположным знаком;
- **амплитуда завышена / занижена**: знак верный и значима, но интервалы не пересекаются, сгенерированная разница по модулю больше (меньше) реальной;
- **не воспроизведено**: реальная разница значима и достаточно велика, а у сгенерированных её нет или знак противоположный без значимости;
- **мало образцов**: реальная разница значима, но по модулю меньше минимально обнаружимой для этой сгенерированной выборки (80 % мощности при пороге 5 %), поэтому отсутствие эффекта у сгенерированных ничего не доказывает;
- **лишний эффект**: у реальных различия нет (не значимо), а у сгенерированных значимое различие вне интервала реальных;
- **вне интервала реальных**: у реальных различия нет, у сгенерированных нет значимости, но их оценка и интервал не согласуются с интервалом реальных.

**Вердикты** (колонки «реальные против сгенерированных»):
- **отличаются**: q < 0,05 для разности d;
- **эквивалентны (доказано)**: не отличаются и доказана эквивалентность тестом TOST: |d| меньше границы 0,2 стандартного отклонения логарифма метрики по реальным шипикам (q TOST < 0,05);
- **не отличимы, эквивалентность не доказана**: различия не найдено, но и доказать, что их нет в пределах 0,2 σ, не удалось (это утверждение о мощности, а не о совпадении).

"""

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
    esc = lambda text: str(text).replace("|", "\\|")  # noqa: E731  (a pipe inside a cell would split the markdown table)
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(esc(fmt(r[c])) for c in cols) + " |")
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


def effect_tables(folder: Path, lab: str, a: str, b: str) -> list[str]:
    """The tables of section 1.1 (median a, median b, relative difference, Cliff's delta, q, significance) for the REAL reference set of the
    comparison (validation neurons) and for the generated set ``lab``."""
    r = read(folder / "real_effect.csv")
    g = read(folder / f"generated_effect_{lab}.csv")
    if r is None or g is None:
        return []
    r, g = order(r).reset_index(drop=True), order(g).reset_index(drop=True)
    real = pd.DataFrame({"метрика": r["metric"], f"медиана {a}": r["median_a"], f"медиана {b}": r["median_b"], "отличие, %": r["rel_diff_a_vs_b"].map(pct),
                         "δ Клиффа": r["cliffs_delta"], "q (BH)": r["q_bh"], "значимо": r["significant"]})
    gen = pd.DataFrame({"метрика": g["metric"], f"медиана {a}": g["median_a"], f"медиана {b}": g["median_b"],
                        "отличие, %": (g["median_a"] / g["median_b"] - 1).map(pct), "δ Клиффа": g["cliffs_delta"], "q (BH)": g["q_bh"], "значимо": g["significant"]})
    return [f"**Реальные шипики, с которыми сравниваются сгенерированные** (проверочные нейроны `splits_v1`: {int(r['n_neurons_a'].iloc[0])} нейронов типа {a}, "
            f"{int(r['n_spines_a'].iloc[0])} шипиков; {int(r['n_neurons_b'].iloc[0])} нейронов типа {b}, {int(r['n_spines_b'].iloc[0])} шипиков). Таблица такая же, как в разделе 1: "
            "медиана здесь медиана по нейронам (у каждого нейрона берётся медиана метрики по его шипикам), тест на уровне нейронов, "
            f"значимы {int(r['significant'].sum())} из {len(r)} метрик.", "", md(real), "",
            f"**Сгенерированные шипики** (набор {lab}: {int(g['n_a'].max())} шипиков типа {a} и {int(g['n_b'].max())} типа {b}). Те же столбцы, но единица здесь шипик, а не нейрон: "
            "медиана по сгенерированным шипикам, значимость по перестановке меток между сгенерированными шипиками (у сгенерированных нет нейронов), "
            "поэтому q двух таблиц напрямую не сопоставимы: у сгенерированных выборка больше и тест мощнее. "
            f"Значимы {int(g['significant'].sum())} из {len(g)} метрик.", "", md(gen), ""]


def source_values(folder: Path, lab: str, a: str, b: str) -> list[str]:
    """The numbers the comparison is made of: pooled medians of real and generated spines of both types and their log ratios."""
    r = read(folder / "real_effect.csv")
    g = read(folder / f"generated_effect_{lab}.csv")
    if r is None or g is None:
        return []
    r, g = order(r).reset_index(drop=True), order(g).reset_index(drop=True)
    t = pd.DataFrame({"метрика": r["metric"],
                      f"реальные {a}: медиана": r["pooled_median_a"], f"реальные {b}: медиана": r["pooled_median_b"], "реальная разница, log": r["log_ratio"],
                      f"сгенерированные {a}: медиана": g["median_a"], f"сгенерированные {b}: медиана": g["median_b"], "сгенерированная разница, log": g["log_ratio"]})
    head = (f"Исходные значения для сравнения (набор {lab}). Реальные: шипики проверочных нейронов `splits_v1` ({int(r['n_neurons_a'].iloc[0])} нейронов типа {a}, "
            f"{int(r['n_spines_a'].iloc[0])} шипиков; {int(r['n_neurons_b'].iloc[0])} нейронов типа {b}, {int(r['n_spines_b'].iloc[0])} шипиков; таблица морфометрик этапа 8 полного Minnie65). "
            f"Сгенерированные: {int(g['n_a'].max())} шипиков типа {a} и {int(g['n_b'].max())} типа {b} (валидные сетки с найденным основанием; для объёма и оболочки все сетки), "
            "условие задавалось при генерации. Генератор нигде не видел проверочные нейроны.")
    return [head, "", md(t), ""]


def corrected_margin(tab: pd.DataFrame, grid=None) -> pd.Series:
    """Smallest equivalence margin (in sd of the log metric) at which the TOST equivalence of a metric is proven AFTER the
    Benjamini-Hochberg correction over all metrics of the table (the correction the verdict uses). min_margin_sd of the report is the
    same quantity WITHOUT the correction. sd of a metric is recovered from min_margin_sd = (|d| + 1.645 SE) / sd."""
    from scipy.stats import norm

    from src.neuron_model.spine_generation.evaluation.conditional_tests import bh_qvalues

    d, se = tab["log_ratio"].abs().to_numpy(float), tab["se"].to_numpy(float)
    sd = (d + 1.645 * se) / tab["min_margin_sd"].to_numpy(float)
    grid = np.arange(0.02, 3.0001, 0.005) if grid is None else grid
    out = np.full(len(tab), np.nan)
    for delta in grid:
        p = norm.cdf((d - delta * sd) / se)
        q = bh_qvalues(p)
        newly = np.isnan(out) & (q < 0.05)
        out[newly] = delta
    return pd.Series(out, index=tab.index)


def per_metric_generated(folder: Path, lab: str, ctrl: str, types: list[str]) -> list[str]:
    c = read(folder / f"concordance_{lab}.csv")
    cc = read(folder / f"concordance_{ctrl}.csv")
    if c is None:
        return []
    d = order(c)
    t = pd.DataFrame({"метрика": d["metric"], "реальная разница, log": d["lr_real"], "реальная разница, % (= exp(log) − 1)": d["lr_real"].map(lambda x: pct(np.expm1(x))),
                      "q реальных": d["real_q"], f"сгенерированная ({lab}), log": d["lr_gen"],
                      f"сгенерированная ({lab}), %": d["lr_gen"].map(lambda x: pct(np.expm1(x))), f"статус ({lab})": d["status"].map(STATUS_RU).fillna(d["status"])})
    if cc is not None:
        t[f"статус (реальные через цепочку, {ctrl})"] = order(cc)["status"].map(STATUS_RU).fillna(order(cc)["status"]).values
    for typ in types:
        s = read(folder / f"same_type_{typ}@{lab}_vs_{typ}.csv")
        if s is not None:
            t[f"вердикт, тип {typ}"] = order(s)["verdict"].map(VERDICT_RU).fillna(order(s)["verdict"]).values
            t[f"d, тип {typ}"] = order(s)["log_ratio"].values
            so = order(s).reset_index(drop=True)
            sd_log = (so["log_ratio"].abs() + 1.645 * so["se"]) / so["min_margin_sd"]
            t[f"|d|, σ (тип {typ})"] = (so["log_ratio"].abs() / sd_log).round(3).values
            t[f"граница без поправки, σ (тип {typ})"] = order(s)["min_margin_sd"].values
            t[f"граница с поправкой, σ (тип {typ})"] = corrected_margin(order(s).reset_index(drop=True)).round(3).values
    return [f"Подробно по метрикам (набор {lab}):", "", md(t), ""]


METHOD_B = """## 5. Метод теста Б (совпадают ли сгенерированные и реальные шипики одного типа)

**Вопрос.** Можно ли отличить сгенерированный набор типа T от реальных шипиков того же типа T по типичному значению метрики?

**Статистика.** Для каждой метрики d = ln(медиана сгенерированных) − ln(медиана реальных шипиков типа T). d = 0 означает, что медианы совпадают, d = 0,3 означает, что сгенерированная медиана в e^0,3 ≈ 1,35 раза больше.

**Почему медиана.** Метрики шипика асимметричны (длинный правый хвост: у части шипиков объём в разы больше типичного), среднее определяется хвостом и выбросами, медиана устойчива и описывает «типичный» шипик.

**Почему логарифм.** (1) Метрики измеряются в разных единицах (нм, нм², нм³, безразмерные), в логарифме разность d безразмерна и сопоставима между метриками. (2) d задаёт относительную ошибку: «в 1,35 раза больше» значит одно и то же для шипиков в 1 раз и в 10 раз крупнее. (3) Разность логарифмов симметрична: ошибка в обе стороны (в 2 раза больше и в 2 раза меньше) даёт одну величину с разным знаком. (4) Распределения метрик близки к логнормальным, в логарифме ошибки оценки близки к нормальным, и доверительные интервалы корректнее.

**Откуда ошибка оценки.** Реальные шипики сгруппированы по нейронам (шипики одного нейрона не независимы, нейроны похожи «стилем»). Поэтому бутстрэп устроен в два слоя: в каждой из 1000 реплик реальные нейроны перевыбираются с возвращением, затем из выбранных нейронов набирается 5000 реальных шипиков (нейрон выбирается с равной вероятностью, чтобы один нейрон не доминировал), а сгенерированная выборка независимо перевыбирается с возвращением. Получается распределение d* (1000 значений). Из него: 95 % интервал по процентилям, стандартная ошибка (SE), двусторонний p по инверсии процентилей: p = 2·min(доля d* ≤ 0, доля d* ≥ 0). Поправка Бенджамини–Хохберга по 11 метрикам (q). Вердикт «отличаются»: q < 0,05.

**Эквивалентность (TOST).** «Нет значимого различия» не означает «совпадают». Для доказательства совпадения задаётся граница δ = 0,2 стандартного отклонения логарифма метрики по реальным шипикам («малый» эффект) и проверяется, что |d| < δ: p_TOST = Φ((|d| − δ)/SE), поправка по метрикам. Если q_TOST < 0,05, вердикт «эквивалентны (доказано)». Если различия нет, а эквивалентность не доказана, вердикт «не отличимы, эквивалентность не доказана» (это утверждение о мощности). В таблицах есть min_margin_sd = (|d| + 1,645·SE)/sd: наименьшая граница, при которой эквивалентность была бы доказана.

**Почему не KS-тест по шипикам.** Шипиков десятки тысяч, и тест по шипикам видит различия на уровне нейронного «стиля» там, где различий нет: на симуляции наивный KS отвергал равенство в 55 % случаев при истинном равенстве, тест с нейроном как единицей 0,5 %. Симуляция самого теста Б (генератор = свежая выборка из истинной совокупности, 300 повторов × 5 метрик): ошибка I рода 3,8–3,9 % при 60 и 200 нейронах, мощность при сдвиге 0,3σ 90 % и 99 %.

**Что тест не проверяет.** Он сравнивает только положение (медиану). Форму распределения, хвосты и корреляции между метриками он не видит. Для них считаются описательные показатели: W₁/σ (расстояние между распределениями в долях σ), энергетическое расстояние стандартизованных логарифмов, сравнение матриц корреляций, всё это относительно «шумового пола» (два непересекающихся половинных набора реальных нейронов). Пол шире истинного нуля, поэтому он снисходителен к генератору, и эти показатели не доказывают совпадения.
"""


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
    if p is not None:
        pair = p[(p["a"] == "4P") & (p["b"] == "6P-CT")]
        if len(pair):
            L += [f"### 1.2а. Пара 4P против 6P-CT (все нейроны: {int(pair['n_neurons_a'].iloc[0])} нейронов 4P, {int(pair['n_spines_a'].iloc[0])} шипиков; "
                  f"{int(pair['n_neurons_b'].iloc[0])} нейронов 6P-CT, {int(pair['n_spines_b'].iloc[0])} шипиков)", "",
                  f"Та же таблица, что в 1.1, для пары типов, которая используется в оценке сгенерированных шипиков (раздел 3.2). Значимы {int(pair['significant'].sum())} из {len(pair)} метрик. "
                  "Поправка Бенджамини–Хохберга считается внутри всей таблицы пар подробных типов (495 тестов), поэтому она строже, чем для одной пары.", "",
                  md(real_pair(pair, '4P', '6P-CT', '4P', '6P-CT')), ""]
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
    L += [LEGEND]
    L += effect_tables(rb, "g1.25-pw8", "тормозные", "возбуждающие")
    L += source_values(rb, "g1.25-pw8", "тормозные", "возбуждающие")
    L += per_metric_generated(rb, "g1.25-pw8", "real-pw8", ["excitatory", "inhibitory"])
    rt = A / a.report_types
    L += ["### 3.2. Подробный тип (4P против 6P-CT; усиление 1,5)", "",
          "В отчёте оставлена пара 4P / 6P-CT, BPC (4 проверочных нейрона) исключён из интерпретации, у MC нет проверочных нейронов.", ""]
    L += generated_block(rt, ["g1.5-pw0", "g1.5-pw8", "real-pw0", "real-pw8"], ["4P", "6P-CT"], {})
    L += ["Столбцы и статусы те же, что в 3.1 (легенда выше); a = 4P, b = 6P-CT.", ""]
    L += effect_tables(rt, "g1.5-pw8", "4P", "6P-CT")
    L += source_values(rt, "g1.5-pw8", "4P", "6P-CT")
    L += per_metric_generated(rt, "g1.5-pw8", "real-pw8", ["4P", "6P-CT"])
    L += ["## 4. Что не выполнено", "",
          "- Компартмент у сгенерированных шипиков: модель на него не обучалась, реальный эффект 1–2 %.",
          "- Вид у сгенерированных шипиков: v4 обучена, но генерация по виду не запускалась (цепочка AE).",
          "- Здоровье и лечение (LabID): исключены из этой сводки по решению пользователя, прежний глобальный прогон устарел (единица папка, поправка z 0,7).", ""]
    L += [METHOD_B]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
