
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA Agent (CSV/NF-e) com memória e PDF
--------------------------------------
- Leitura robusta de CSV (encoding/delimitador automáticos, números BR, datas).
- Normalização de colunas (snake_case, sem acento).
- Perfil, checagens de qualidade e heurísticas de NF-e.
- Perguntas em PT-BR e suporte a SQL (DuckDB).
- Geração de gráficos, HTML e PDF.
- Conclusões automáticas + memória (JSON).

Autor: ChatGPT (GPT-5 Thinking) | Licença: MIT
"""

from __future__ import annotations
import os, io, re, csv, json, unicodedata
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Iterable
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

# opcionais
try:
    import chardet
except Exception:
    chardet = None
try:
    import duckdb
except Exception:
    duckdb = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Utilitários -------------------------

def strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def slugify_col(s: str) -> str:
    s = strip_accents(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def human_bytes(num: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024: return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"

def detect_encoding(path: Path, sample_size: int = 200_000) -> str:
    try:
        raw = Path(path).open('rb').read(sample_size)
    except Exception:
        return "utf-8"
    if chardet is not None:
        try:
            res = chardet.detect(raw)
            return res.get("encoding") or "utf-8"
        except Exception:
            pass
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            raw.decode(enc); return enc
        except Exception:
            continue
    return "utf-8"

def sniff_delimiter(path: Path, encoding: str) -> Optional[str]:
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            sample = f.read(100_000)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return None

_numeric_comma_regex = re.compile(r"^\s*-?\d{1,3}(\.\d{3})*,\d+\s*$")
_numeric_dot_regex = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")

def looks_like_brazilian_number(x: str) -> bool:
    return isinstance(x, str) and bool(_numeric_comma_regex.match(x))

def normalize_brazilian_number(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if _numeric_comma_regex.match(s):
            s = s.replace(".", "").replace(",", ".")
            try: return float(s)
            except Exception: return np.nan
    return x

def try_parse_date_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind == "M": return s
    if s.dtype == "O":
        sample = s.dropna().astype(str).head(200)
        like = 0
        for v in sample:
            v2 = v.strip()
            if re.search(r"\d{4}-\d{2}-\d{2}", v2) or re.search(r"\d{2}/\d{2}/\d{4}", v2):
                like += 1
        ratio = like / max(1, len(sample))
        if ratio < 0.25: return s
        try: return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        except Exception:
            try: return pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
            except Exception: return s
    return s

def is_valid_cnpj(cnpj: str) -> bool:
    if not isinstance(cnpj, str): return False
    c = re.sub(r"\D", "", cnpj)
    if len(c) != 14 or c == c[0]*14: return False
    def dv(digs: str) -> int:
        w = [5,4,3,2,9,8,7,6,5,4,3,2] if len(digs)==12 else [6,5,4,3,2,9,8,7,6,5,4,3,2]
        s = sum(int(d)*w[i] for i,d in enumerate(digs))
        r = s % 11
        return 0 if r < 2 else 11 - r
    d1 = dv(c[:12]); d2 = dv(c[:12] + str(d1))
    return c[-2:] == f"{d1}{d2}"

def is_valid_cpf(cpf: str) -> bool:
    if not isinstance(cpf, str): return False
    c = re.sub(r"\D", "", cpf)
    if len(c) != 11 or c == c[0]*11: return False
    def dv(digs: str) -> int:
        s = sum(int(d)*w for d,w in zip(digs, range(len(digs)+1,1,-1)))
        r = (s*10) % 11
        return 0 if r == 10 else r
    d1 = dv(c[:9]); d2 = dv(c[:9] + str(d1))
    return c[-2:] == f"{d1}{d2}"

def autowidth_fig():
    return plt.figure(figsize=(9, 4.5), dpi=140)


# ------------------------- Núcleo do Agente -------------------------

@dataclass
class AgentConfig:
    prefer_value_cols: Tuple[str,...] = (
        "valor_total","valor","valor_empenhado","valor_liquidado","valor_pago","valor_da_nota","vl_total","vlr","preco","amount"
    )
    prefer_date_cols: Tuple[str,...] = (
        "data_emissao","data","emissao","competencia","data_documento","data_pagamento","data_liquidacao","transaction_date"
    )
    prefer_cnpj_cols: Tuple[str,...] = ("cnpj_emitente","cnpj","cnpj_fornecedor","cnpj_cpf","cnpj_cpf_fornecedor")
    prefer_cpf_cols: Tuple[str,...] = ("cpf","cpf_fornecedor","cpf_cnpj","cpf_cnpj_fornecedor")
    prefer_desc_cols: Tuple[str,...] = ("descricao","descricao_item","historico","item","objeto","natureza")
    max_charts_per_type: int = 12
    rounding: int = 2
    known_thresholds_brl: Tuple[float,...] = (8000.0, 17600.0, 50000.0)

class CSVAgent:
    def __init__(self, config: AgentConfig|None = None):
        self.config = config or AgentConfig()
        self.df: Optional[pd.DataFrame] = None
        self.original_columns: Optional[List[str]] = None
        self.colmap: Dict[str,str] = {}
        self.meta: Dict[str, Any] = {}

    # ----------------- Leitura e normalização -----------------
    def load(self, path_or_buffer) -> "CSVAgent":
        if isinstance(path_or_buffer, (str, Path)):
            path = Path(path_or_buffer)
            enc = detect_encoding(path)
            delim = sniff_delimiter(path, enc)
            try:
                df = pd.read_csv(path, sep=delim if delim else None, engine="python", encoding=enc, dtype=str)
            except Exception:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        else:
            try:
                path_or_buffer.seek(0)
            except Exception: pass
            df = pd.read_csv(path_or_buffer, sep=None, engine="python", dtype=str)

        self.original_columns = list(df.columns)
        new_cols, self.colmap = [], {}
        for c in df.columns:
            sc = slugify_col(str(c))
            if sc in self.colmap:
                k=2; sc2=f"{sc}_{k}"
                while sc2 in self.colmap: k+=1; sc2=f"{sc}_{k}"
                sc = sc2
            self.colmap[sc] = c; new_cols.append(sc)
        df.columns = new_cols

        for col in df.columns:
            if df[col].dtype == "O":
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                sample = df[col].dropna().astype(str).head(200)
                frac = sum(looks_like_brazilian_number(v) for v in sample) / max(1, len(sample))
                if frac >= 0.5:
                    df[col] = df[col].map(normalize_brazilian_number)
                else:
                    frac_dot = sum(bool(_numeric_dot_regex.match(v)) for v in sample) / max(1, len(sample))
                    if frac_dot >= 0.5:
                        df[col] = pd.to_numeric(df[col], errors="ignore")
        for col in df.columns:
            df[col] = try_parse_date_series(df[col])
        df = df.convert_dtypes()

        self.df = df
        self.meta["num_rows"] = len(df)
        self.meta["num_cols"] = len(df.columns)
        self.meta["memory_usage"] = df.memory_usage(deep=True).sum()
        return self

    # ----------------- Helpers -----------------
    def find_first(self, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in self.df.columns: return c
        return None
    def pick_value_col(self) -> Optional[str]: return self.find_first(self.config.prefer_value_cols)
    def pick_date_col(self) -> Optional[str]:  return self.find_first(self.config.prefer_date_cols)
    def pick_cnpj_col(self) -> Optional[str]:  return self.find_first(self.config.prefer_cnpj_cols)
    def pick_cpf_col(self) -> Optional[str]:   return self.find_first(self.config.prefer_cpf_cols)
    def pick_desc_col(self) -> Optional[str]:  return self.find_first(self.config.prefer_desc_cols)

    # ----------------- Perfil -----------------
    def profile(self) -> Dict[str, Any]:
        assert self.df is not None
        df = self.df
        info = {
            "linhas": len(df),
            "colunas": len(df.columns),
            "uso_de_memoria": human_bytes(df.memory_usage(deep=True).sum()),
            "tipos": {c: str(df[c].dtype) for c in df.columns},
            "faltantes_por_coluna_pct": {c: float(df[c].isna().mean()*100) for c in df.columns},
            "exemplo_de_linhas": df.head(5).to_dict(orient="records"),
        }
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            info["resumo_numerico"] = df[num_cols].describe().to_dict()
        cat_cols = [c for c in df.columns if df[c].dtype == "string" or pd.api.types.is_object_dtype(df[c])]
        topcats = {}
        for c in cat_cols:
            vc = df[c].value_counts(dropna=True).head(10)
            if len(vc): topcats[c] = vc.to_dict()
        info["top_categorias"] = topcats
        dcol = self.pick_date_col()
        if dcol is not None and pd.api.types.is_datetime64_any_dtype(df[dcol]):
            info["intervalo_datas"] = {"min": str(pd.to_datetime(df[dcol]).min()),
                                       "max": str(pd.to_datetime(df[dcol]).max())}
        return info

    # ----------------- Qualidade -----------------
    def quality_checks(self) -> Dict[str, Any]:
        assert self.df is not None
        df = self.df; issues = {}
        dup_all = df.duplicated().sum()
        if dup_all > 0: issues["linhas_duplicadas"] = int(dup_all)
        miss = {c: float(df[c].isna().mean()*100) for c in df.columns}
        high_miss = {c:p for c,p in miss.items() if p >= 50.0}
        if high_miss: issues["colunas_com_muitos_faltantes_pct"] = high_miss
        vcol = self.pick_value_col()
        if vcol and pd.api.types.is_numeric_dtype(df[vcol]):
            s = df[vcol].dropna()
            if len(s) >= 10:
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lim_low, lim_high = q1 - 1.5*iqr, q3 + 1.5*iqr
                outliers = df[(df[vcol] < lim_low) | (df[vcol] > lim_high)]
                issues["outliers_iqr_count"] = int(len(outliers))
                issues["outliers_iqr_limits"] = {"low": float(lim_low), "high": float(lim_high)}
        return issues

    # ----------------- NF-e -----------------
    def nfe_checks(self) -> Dict[str, Any]:
        assert self.df is not None
        df = self.df; report = {}
        cnpj_col = self.pick_cnpj_col(); cpf_col = self.pick_cpf_col()
        date_col = self.pick_date_col(); value_col = self.pick_value_col()

        if cnpj_col:
            invalid = df[cnpj_col].dropna().astype(str).apply(lambda x: not is_valid_cnpj(x)).sum()
            report["cnpj_invalidos"] = int(invalid)
        if cpf_col:
            invalid = df[cpf_col].dropna().astype(str).apply(lambda x: not is_valid_cpf(x)).sum()
            report["cpf_invalidos"] = int(invalid)

        NotaCols = [c for c in df.columns if any(k in c for k in ("numero_nota","n_nota","nota","num_nota"))]
        nota_col = NotaCols[0] if NotaCols else None
        if nota_col and cnpj_col:
            subset = [x for x in (nota_col, cnpj_col, date_col) if x is not None]
            if subset:
                dups = df.duplicated(subset=subset).sum()
                report["notas_potencialmente_duplicadas"] = int(dups)

        if all([date_col, cnpj_col, value_col]) and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            grp = df.groupby([cnpj_col, df[date_col].dt.date]).size().reset_index(name="qtd_lancamentos")
            suspeitos = grp[grp["qtd_lancamentos"] >= 5]
            report["fornecedor_dias_com_muitos_lancamentos"] = int(len(suspeitos))
            suspicious_ranges = []
            values = df[value_col].dropna()
            if len(values) >= 50:
                for th in self.config.known_thresholds_brl:
                    near = values[(values >= 0.9*th) & (values <= 1.01*th)]
                    if len(near) >= max(5, 0.05*len(values)):
                        suspicious_ranges.append(float(th))
                if suspicious_ranges:
                    report["bunching_perto_de_thresholds_brl"] = suspicious_ranges
        return report

    # ----------------- Gráficos -----------------
    def _save_plot(self, fig, outdir: Path, name: str) -> str:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{name}.png"
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        return str(path)

    def generate_charts(self, outdir: str|Path = "charts") -> Dict[str,List[str]]:
        assert self.df is not None
        df = self.df; outdir = Path(outdir)
        results = {"numeric": [], "categorical": [], "time_series": []}
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for i, c in enumerate(num_cols[: self.config.max_charts_per_type]):
            fig = autowidth_fig(); ax = fig.gca()
            df[c].dropna().hist(bins=30, ax=ax)
            ax.set_title(f"Distribuição de {c}"); ax.set_xlabel(c); ax.set_ylabel("Frequência")
            results["numeric"].append(self._save_plot(fig, outdir, f"hist_{i:02d}_{c}"))
        cat_cols = [c for c in df.columns if df[c].dtype == "string" or pd.api.types.is_object_dtype(df[c])]
        for i, c in enumerate(cat_cols[: self.config.max_charts_per_type]):
            vc = df[c].value_counts(dropna=True).head(15)
            if len(vc) == 0: continue
            fig = autowidth_fig(); ax = fig.gca()
            vc.plot(kind="bar", ax=ax); ax.set_title(f"Top categorias em {c}"); ax.set_xlabel(c); ax.set_ylabel("Contagem")
            results["categorical"].append(self._save_plot(fig, outdir, f"bar_{i:02d}_{c}"))
        dcol = self.pick_date_col(); vcol = self.pick_value_col()
        if dcol and vcol and pd.api.types.is_datetime64_any_dtype(df[dcol]) and pd.api.types.is_numeric_dtype(df[vcol]):
            tmp = df[[dcol, vcol]].dropna().copy()
            if len(tmp) > 0:
                tmp["periodo"] = tmp[dcol].dt.to_period("M").dt.to_timestamp()
                ts = tmp.groupby("periodo")[vcol].sum()
                fig = autowidth_fig(); ax = fig.gca()
                ts.plot(ax=ax); ax.set_title(f"Série mensal de {vcol}"); ax.set_xlabel("Período"); ax.set_ylabel(vcol)
                results["time_series"].append(self._save_plot(fig, outdir, f"timeseries_{vcol}"))
        return results

    # ----------------- Conclusões & Memória -----------------
    def compute_conclusions(self, max_items: int = 5) -> Dict[str, Any]:
        assert self.df is not None
        df = self.df
        prof = self.profile()
        qual = self.quality_checks()
        nfe = self.nfe_checks()

        findings = []
        details = {}

        findings.append(f"O dataset possui {prof['linhas']:,} linhas e {prof['colunas']} colunas; uso de memória ~{prof['uso_de_memoria']}.")

        high_missing = [(c, pct) for c, pct in prof["faltantes_por_coluna_pct"].items() if pct >= 50.0]
        if high_missing:
            high_missing = sorted(high_missing, key=lambda x: x[1], reverse=True)[:max_items]
            findings.append("Há colunas com muitos faltantes (≥50%), o que pode comprometer análises e modelos.")
            details["colunas_muitos_faltantes"] = [{ "coluna": c, "pct": round(pct, 2) } for c, pct in high_missing]

        vcol = self.pick_value_col(); dcol = self.pick_date_col()
        if "outliers_iqr_count" in qual and qual["outliers_iqr_count"] > 0:
            findings.append(f"Foram detectados {qual['outliers_iqr_count']} outliers (IQR) na coluna de valor, fora dos limites {qual.get('outliers_iqr_limits', {})}.")
        if dcol and vcol and pd.api.types.is_datetime64_any_dtype(df[dcol]) and pd.api.types.is_numeric_dtype(df[vcol]):
            tmp = df[[dcol, vcol]].dropna().copy()
            if len(tmp) > 0:
                tmp["periodo"] = tmp[dcol].dt.to_period("M").dt.to_timestamp()
                ts = tmp.groupby("periodo")[vcol].sum()
                if len(ts) >= 3:
                    trend = "crescente" if ts.iloc[-1] > ts.iloc[0] else "decrescente"
                    findings.append(f"A soma mensal de '{vcol}' apresenta tendência {trend} do primeiro ao último período observado.")

        if nfe.get("cnpj_invalidos", 0) > 0:
            findings.append(f"Existem {nfe['cnpj_invalidos']} CNPJs com dígitos inválidos — revisar cadastro/ingestão.")
        if nfe.get("notas_potencialmente_duplicadas", 0) > 0:
            findings.append(f"Foram encontradas {nfe['notas_potencialmente_duplicadas']} notas potencialmente duplicadas (mesmo nº + emitente + data).")
        if nfe.get("fornecedor_dias_com_muitos_lancamentos", 0) > 0:
            findings.append("Há dias com muitos lançamentos para o mesmo fornecedor — verificar risco de fracionamento.")
        if nfe.get("bunching_perto_de_thresholds_brl"):
            findings.append(f"Concentração de valores próximo a limites regulatórios: {nfe['bunching_perto_de_thresholds_brl']} — possível 'bunching'.")

        if "class" in df.columns:
            class_counts = df["class"].value_counts(dropna=False)
            total = class_counts.sum()
            fraud = int(class_counts.get(1, 0)); normal = int(class_counts.get(0, 0))
            fraud_rate = fraud / total * 100 if total else 0
            findings.append(f"Detecção de coluna 'class' (fraude): {fraud} transações (≈{fraud_rate:.3f}%) positivas e {normal} negativas — forte desbalanceamento.")
            details["fraud_rate_pct"] = round(fraud_rate, 3)

            cand_amt = [c for c in (vcol, "amount", "valor", "valor_total") if c and c in df.columns]
            amt_col = cand_amt[0] if cand_amt else None
            if amt_col and pd.api.types.is_numeric_dtype(df[amt_col]):
                g = df.groupby("class", dropna=False)[amt_col].agg(["count", "mean", "median"]).reset_index()
                details["valores_por_classe"] = g.to_dict(orient="records")
                findings.append(f"Comparando '{amt_col}' por classe, há diferenças nas medidas de tendência central que podem indicar padrões distintos.")

        text = " ".join(findings) if findings else "Sem conclusões relevantes no momento."
        return {"texto": text, "detalhes": details, "perfil": prof, "qualidade": qual, "nfe": nfe}

    def save_memory(self, memory_path: str|Path = "agent_memory.json", dataset_label: Optional[str] = None) -> str:
        assert self.df is not None
        memo_path = Path(memory_path)
        try:
            memo = json.loads(memo_path.read_text(encoding="utf-8")) if memo_path.exists() else {}
        except Exception:
            memo = {}
        label = dataset_label or f"dataset_{self.meta.get('num_rows','?')}x{self.meta.get('num_cols','?')}"
        concl = self.compute_conclusions()
        memo[label] = {
            "conclusoes": concl,
            "colunas_originais": self.original_columns,
            "colunas_normalizadas": list(self.df.columns),
            "momento": datetime.now().isoformat(timespec="seconds")
        }
        memo_path.write_text(json.dumps(memo, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(memo_path)

    # ----------------- HTML -----------------
    def save_html_report(self, filepath: str|Path = "eda_report.html") -> str:
        assert self.df is not None
        prof = self.profile(); qual = self.quality_checks(); nfe = self.nfe_checks()
        charts = self.generate_charts(outdir=Path(filepath).with_suffix("").name + "_charts")

        def dict_to_html(d: Dict[str, Any]) -> str:
            return f"<pre>{json.dumps(d, indent=2, ensure_ascii=False)}</pre>"

        imgs_html = ""
        for kind, paths in charts.items():
            if not paths: continue
            imgs_html += f"<h3>Gráficos: {kind}</h3>\n"
            for p in paths:
                imgs_html += f'<img src="{os.path.basename(p)}" style="max-width:100%;height:auto;display:block;margin-bottom:10px;">\n'

        from datetime import datetime as _dt
        html = f"""
<!DOCTYPE html><html lang="pt-br"><head><meta charset="utf-8"><title>Relatório EDA Agent</title>
<style> body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:20px; line-height:1.3}} h1,h2,h3{{margin-top:1.2em}} pre{{background:#111;color:#eee;padding:12px;border-radius:8px;overflow:auto}}</style>
</head><body>
<h1>Relatório EDA Agent</h1>
<p><strong>Linhas:</strong> {prof['linhas']} &nbsp;|&nbsp; <strong>Colunas:</strong> {prof['colunas']} &nbsp;|&nbsp; <strong>Memória:</strong> {prof['uso_de_memoria']}</p>
<h2>Tipos e Faltantes</h2>{dict_to_html({"tipos": prof["tipos"], "faltantes_por_coluna_pct": prof["faltantes_por_coluna_pct"]})}
<h2>Resumo numérico</h2>{dict_to_html(prof.get("resumo_numerico", {}))}
<h2>Top categorias</h2>{dict_to_html(prof.get("top_categorias", {}))}
<h2>Intervalo de datas</h2>{dict_to_html(prof.get("intervalo_datas", {}))}
<h2>Checagens de Qualidade</h2>{dict_to_html(qual)}
<h2>Checagens específicas NF-e</h2>{dict_to_html(nfe)}
<h2>Exemplos de linhas</h2>{dict_to_html({"exemplo_de_linhas": prof["exemplo_de_linhas"]})}
<h2>Gráficos</h2><p>As imagens abaixo ficam na pasta <code>{Path(filepath).with_suffix("").name}_charts</code>.</p>
{imgs_html}
<hr><p style="color:#666">Gerado por EDA Agent · {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body></html>
"""
        out = Path(filepath); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f: f.write(html)
        return str(out)

    # ----------------- PDF -----------------
    def save_pdf_report(self, filepath: str|Path = "Agentes Autônomos – Relatório da Atividade Extra.pdf") -> str:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        except Exception as e:
            raise RuntimeError("Para gerar PDF, instale 'reportlab' (pip install reportlab).") from e

        charts = self.generate_charts(outdir=Path("pdf_charts"))
        concl = self.compute_conclusions()
        prof = concl["perfil"]; qual = concl["qualidade"]; nfe  = concl["nfe"]

        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Agentes Autônomos – Relatório da Atividade Extra", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Linhas: {prof['linhas']:,}  |  Colunas: {prof['colunas']}  |  Memória: {prof['uso_de_memoria']}", styles["Normal"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Tipos & Faltantes (%)</b>", styles["Heading2"]))
        story.append(Paragraph(json.dumps({'tipos': prof['tipos'], 'faltantes_por_coluna_pct': prof['faltantes_por_coluna_pct']}, ensure_ascii=False, indent=2).replace('\n','<br/>'), styles["Code"]))
        if 'resumo_numerico' in prof:
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Resumo numérico</b>", styles["Heading2"]))
            story.append(Paragraph(json.dumps(prof['resumo_numerico'], ensure_ascii=False, indent=2).replace('\n','<br/>'), styles["Code"]))
        if prof.get("intervalo_datas"):
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Intervalo de datas</b>", styles["Heading2"]))
            story.append(Paragraph(json.dumps(prof['intervalo_datas'], ensure_ascii=False, indent=2).replace('\n','<br/>'), styles["Code"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Checagens de Qualidade</b>", styles["Heading2"]))
        story.append(Paragraph(json.dumps(qual, ensure_ascii=False, indent=2).replace('\n','<br/>'), styles["Code"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Checagens específicas NF-e</b>", styles["Heading2"]))
        story.append(Paragraph(json.dumps(nfe, ensure_ascii=False, indent=2).replace('\n','<br/>'), styles["Code"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Conclusões do Agente</b>", styles["Heading2"]))
        story.append(Paragraph(concl["texto"], styles["BodyText"]))
        story.append(Spacer(1, 12))

        added = 0
        for kind, paths in charts.items():
            for p in paths:
                if added >= 6: break
                story.append(Paragraph(f"Gráfico: {kind}", styles["Heading3"]))
                try:
                    story.append(Image(p, width=420, height=210))
                except Exception:
                    pass
                story.append(Spacer(1, 6)); added += 1

        doc.build(story)
        return str(filepath)

    # ----------------- Perguntas -----------------
    def ask(self, question: str) -> Tuple[pd.DataFrame, str]:
        assert self.df is not None
        df = self.df
        q = strip_accents(question.lower()).strip()

        if q.startswith("sql:"):
            if duckdb is None:
                return pd.DataFrame(), "DuckDB não está instalado. Instale com: pip install duckdb"
            sql = question[4:].strip()
            con = duckdb.connect(); con.register("df", df)
            try: ans = con.execute(sql).fetchdf()
            except Exception as e: return pd.DataFrame(), f"Erro SQL: {e}"
            return ans, "Consulta SQL executada."

        vcol = self.pick_value_col(); dcol = self.pick_date_col()

        def extract_cols(text: str) -> List[str]:
            parts = [slugify_col(p.strip()) for p in re.split(r",| e ", text)]
            return [c for c in parts if c in df.columns]

        m = re.search(r"total(izar|)?\s+por\s+(.+)", q)
        if m and vcol and vcol in df.columns and pd.api.types.is_numeric_dtype(df[vcol]):
            cols = extract_cols(m.group(2))
            if cols:
                g = df.groupby(cols, dropna=False)[vcol].sum().reset_index().sort_values(vcol, ascending=False)
                return g, f"Soma de '{vcol}' por {cols}."

        m = re.search(r"top\s+(\d+)\s+(de\s+)?(.+)", q)
        if m:
            n = int(m.group(1)); coltxt = m.group(3).strip()
            cols = extract_cols(coltxt)
            if cols:
                c = cols[0]; vc = df[c].value_counts(dropna=True).head(n).reset_index()
                vc.columns = [c, "contagem"]
                return vc, f"Top {n} valores em '{c}'."

        m = re.search(r"(media|mediana|min|max)\s+de\s+(.+)", q)
        if m:
            op = m.group(1); rest = m.group(2); parts = rest.split(" por ")
            col = slugify_col(parts[0].strip())
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if len(parts) == 2:
                    grpcol = slugify_col(parts[1].strip())
                    if grpcol in df.columns:
                        g = df.groupby(grpcol, dropna=False)[col]
                        res = g.mean().reset_index() if op=="media" else g.median().reset_index() if op=="mediana" else g.min().reset_index() if op=="min" else g.max().reset_index()
                        return res, f"{op.title()} de '{col}' por '{grpcol}'."
                else:
                    s = df[col]
                    val = s.mean() if op=="media" else s.median() if op=="mediana" else s.min() if op=="min" else s.max()
                    return pd.DataFrame({op:[val]}), f"{op.title()} de '{col}'."

        if q.startswith("filtrar "):
            expr = q[len("filtrar "):]
            tokens = [t.strip() for t in re.split(r"\se\s+", expr)]
            mask = pd.Series(True, index=df.index); used = []
            for tok in tokens:
                m = re.match(r"([a-z0-9_]+)\s*=\s*'([^']+)'", tok)
                if not m: m = re.match(r"([a-z0-9_]+)\s*=\s*(.+)", tok)
                if m:
                    col = slugify_col(m.group(1)); val = m.group(2).strip()
                    if col in df.columns:
                        used.append(col)
                        if pd.api.types.is_numeric_dtype(df[col]):
                            try: v = float(val.replace(",", ".").replace(" ", ""))
                            except Exception: v = np.nan
                            mask &= (pd.to_numeric(df[col], errors="coerce") == v)
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            try:
                                v = pd.to_datetime(val, dayfirst=True, errors="coerce")
                                mask &= (pd.to_datetime(df[col]) == v)
                            except Exception: pass
                        else:
                            mask &= (df[col].astype(str).str.lower() == val.lower())
            ans = df[mask].copy()
            return ans, f"Filtro aplicado nas colunas: {used}."

        if dcol and pd.api.types.is_datetime64_any_dtype(df[dcol]):
            m = re.search(r"entre\s+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+e\s+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", q)
            if m:
                d1 = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
                d2 = pd.to_datetime(m.group(2), dayfirst=True, errors="coerce")
                if pd.notna(d1) and pd.notna(d2):
                    mask = (pd.to_datetime(df[dcol]) >= d1) & (pd.to_datetime(df[dcol]) <= d2)
                    return df[mask].copy(), f"Registro entre {d1.date()} e {d2.date()} (coluna {dcol})."

        mentioned = [c for c in df.columns if c in q]
        if mentioned:
            c = mentioned[0]; vc = df[c].value_counts(dropna=True).head(20).reset_index()
            vc.columns = [c, "contagem"]
            return vc, f"Contagem dos top valores em '{c}'."

        return pd.DataFrame(), "Não entendi a pergunta. Exemplos: 'total por fornecedor', 'top 20 de orgao', 'média de valor por mês' ou 'sql:'."


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="EDA Agent - Analisador de CSV genérico (com foco em NF-e)")
    p.add_argument("--file", "-f", required=True, help="Caminho do CSV")
    p.add_argument("--report", action="store_true", help="Gera relatório HTML (eda_report.html)")
    p.add_argument("--pdf", action="store_true", help="Gera relatório PDF com o nome exigido pela atividade")
    p.add_argument("--ask", "-q", help="Pergunta em PT-BR (ex: 'total por fornecedor')")
    args = p.parse_args()

    ag = CSVAgent().load(args.file)
    print(json.dumps(ag.profile(), indent=2, ensure_ascii=False))

    if args.ask:
        ans, msg = ag.ask(args.ask)
        print("\n# Resposta:", msg)
        print(ans.head(50).to_string(index=False))

    if args.report:
        out = ag.save_html_report("eda_report.html")
        print(f"\nRelatório salvo em: {out}")
    if args.pdf:
        out = ag.save_pdf_report("Agentes Autônomos – Relatório da Atividade Extra.pdf")
        print(f"PDF salvo em: {out}")

if __name__ == "__main__":
    _cli()
