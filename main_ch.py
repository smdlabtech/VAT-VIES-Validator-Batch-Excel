# app.py (avec expanders)
# -*- coding: utf-8 -*-

import io
import time
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, field_validator

API_URL = "https://ec.europa.eu/taxation_customs/vies/rest-api/check-vat-number"
REQUIRED_COLS = ["MS Code", "VAT Number"]

# --------------------------- Pydantic ---------------------------
class VATRecord(BaseModel):
    ms_code: str = Field(...)
    vat_number: str = Field(...)
    requester_ms: str = Field(...)
    requester_vat: str = Field("")

    @field_validator("ms_code", "requester_ms")
    @classmethod
    def normalize_country(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError("Code pays invalide (2 lettres)")
        return v

    @field_validator("vat_number", "requester_vat")
    @classmethod
    def normalize_vat(cls, v: str) -> str:
        return (v or "").strip().replace(" ", "").replace(".", "").replace("-", "").upper()


class AppSettings(BaseModel):
    requester_ms_default: str = Field("FR")
    requester_vat_default: str = Field("")
    delay: float = Field(1.5, ge=0.0)
    retries: int = Field(2, ge=1, le=5)
    chunk_size: int = Field(10, ge=1, le=500)
    chunk_pause: float = Field(3.0, ge=0.0)
    valid_only: bool = Field(False)
    timeout: float = Field(10.0, ge=1.0, le=60.0)
    demo_mock_api: bool = Field(False)

    @field_validator("requester_ms_default")
    @classmethod
    def normalize_req_ms(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError("Code pays demandeur par défaut invalide")
        return v

    @field_validator("requester_vat_default")
    @classmethod
    def normalize_req_vat(cls, v: str) -> str:
        return (v or "").strip().replace(" ", "").replace(".", "").replace("-", "").upper()

# --------------------------- Utils ---------------------------
def _clean_country(v) -> str:
    if pd.isna(v): return ""
    return str(v).strip().upper()

def _clean_vat(v) -> str:
    if pd.isna(v): return ""
    return str(v).replace(" ", "").replace(".", "").replace("-", "").upper()

def _ordered_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "MS Code","VAT Number","valid","name","address",
        "Requester MS Code","Requester VAT Number",
        "Attempts","timestamp","error",
    ]
    cols = [c for c in preferred if c in df.columns]
    others = [c for c in df.columns if c not in cols]
    return df[cols + others] if cols else df

def _mock_response(ms_code: str, vat_number: str) -> dict:
    valid = (hash(vat_number) + hash(ms_code)) % 10 < 7
    return {
        "valid": bool(valid),
        "name": "Mock Company SA" if valid else None,
        "address": "1 Rue de la Paix, 75000 Paris, FR" if valid else None,
        "requestIdentifier": "MOCK-REQ-123",
        "timestamp": datetime.now().isoformat(),
    }

def check_vat(ms_code: str, vat_number: str, requester_code: str, requester_vat: str,
              timeout: float = 10.0, demo_mock_api: bool = False) -> dict:
    if demo_mock_api:
        time.sleep(0.1)
        return _mock_response(ms_code, vat_number)
    payload = {
        "countryCode": ms_code,
        "vatNumber": vat_number,
        "requesterCountryCode": requester_code,
        "requesterVatNumber": requester_vat,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            data["timestamp"] = datetime.now().isoformat()
            return data
        return {"valid": False, "error": f"HTTP {r.status_code}", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"valid": False, "error": str(e), "timestamp": datetime.now().isoformat()}

def validate_and_prepare_df(df: pd.DataFrame, settings: AppSettings) -> Tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Colonnes manquantes : {', '.join(missing)}")
        return pd.DataFrame(), errors

    df = df.rename(columns=lambda x: str(x).strip())
    if "Requester MS Code" not in df.columns:
        df["Requester MS Code"] = settings.requester_ms_default
    if "Requester VAT Number" not in df.columns:
        df["Requester VAT Number"] = settings.requester_vat_default

    df["MS Code"] = df["MS Code"].apply(_clean_country)
    df["VAT Number"] = df["VAT Number"].apply(_clean_vat)
    df["Requester MS Code"] = df["Requester MS Code"].apply(_clean_country)
    df["Requester VAT Number"] = df["Requester VAT Number"].apply(_clean_vat)

    valid_df = df.dropna(subset=["MS Code", "VAT Number"])
    valid_df = valid_df[(valid_df["MS Code"] != "") & (valid_df["VAT Number"] != "")]
    dropped = len(df) - len(valid_df)
    if dropped > 0:
        errors.append(f"{dropped} ligne(s) ignorée(s) (code pays ou n° TVA manquant).")

    for idx, row in valid_df.iterrows():
        try:
            VATRecord(
                ms_code=row["MS Code"],
                vat_number=row["VAT Number"],
                requester_ms=row["Requester MS Code"],
                requester_vat=row["Requester VAT Number"],
            )
        except Exception as e:
            errors.append(f"Ligne {idx}: {e}")
    return valid_df, errors

def process_batch(df: pd.DataFrame, settings: AppSettings, status_cb=None, log_cb=None) -> pd.DataFrame:
    total = len(df)
    results = []
    chunks = [df[i:i + settings.chunk_size] for i in range(0, total, settings.chunk_size)]
    processed = 0
    progress = st.progress(0, text="Initialisation…")

    for cidx, chunk in enumerate(chunks, start=1):
        msg = f"Lot {cidx}/{len(chunks)} – {len(chunk)} élément(s)"
        if status_cb: status_cb(msg)
        if log_cb: log_cb(msg)

        for _, row in chunk.iterrows():
            ms = row["MS Code"]; vat = row["VAT Number"]
            req_ms = row["Requester MS Code"]; req_vat = row["Requester VAT Number"]

            attempt = 0; res = None
            while attempt < max(1, int(settings.retries)):
                res = check_vat(ms, vat, req_ms, req_vat, timeout=settings.timeout, demo_mock_api=settings.demo_mock_api)
                if ("error" not in res) or res.get("valid", False):
                    break
                attempt += 1
                time.sleep(max(settings.delay * 1.5, 0.05))

            res = res or {}
            res["MS Code"] = ms; res["VAT Number"] = vat
            res["Requester MS Code"] = req_ms; res["Requester VAT Number"] = req_vat
            res["Attempts"] = attempt + 1
            results.append(res)

            processed += 1
            progress.progress(min(processed / total, 1.0), text=f"Progression {processed}/{total}")
            time.sleep(max(settings.delay, 0.0))

        if cidx < len(chunks) and settings.chunk_pause > 0:
            pause_msg = f"Pause {settings.chunk_pause:.1f}s entre lots…"
            if status_cb: status_cb(pause_msg)
            if log_cb: log_cb(pause_msg)
            time.sleep(settings.chunk_pause)

    out_df = pd.DataFrame(results)
    if settings.valid_only and "valid" in out_df.columns:
        out_df = out_df[out_df["valid"] == True].copy()
    try:
        out_df = _ordered_columns(out_df)
    except Exception:
        pass
    return out_df

def export_bytes(out_df: pd.DataFrame, base_name: str, valid_only: bool) -> Tuple[bytes, bytes, str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_valides" if valid_only else ""
    xlsx_name = f"{base_name}_resultats_TVA{suffix}_{stamp}.xlsx"
    csv_name = f"{base_name}_resultats_TVA{suffix}_{stamp}.csv"

    xbio = io.BytesIO()
    with pd.ExcelWriter(xbio, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False)
    xlsx_bytes = xbio.getvalue()

    cbio = io.BytesIO()
    out_df.to_csv(cbio, index=False)
    csv_bytes = cbio.getvalue()

    return xlsx_bytes, csv_bytes, xlsx_name, csv_name

# --------------------------- UI ---------------------------
st.set_page_config(page_title="VIES TVA Validator", page_icon="🇪🇺", layout="wide", initial_sidebar_state="expanded")
st.title("🇪🇺 VIES TVA Validator – Batch Excel")
st.caption("Vérifie en masse des numéros de TVA via l'API VIES (UE) avec retries, throttling, et export des résultats.")

# journal en session
if "_logs" not in st.session_state: st.session_state["_logs"] = []
def log(msg: str): st.session_state["_logs"].append(f"{datetime.now().strftime('%H:%M:%S')}  |  {msg}")

# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Paramètres")

    # Paramètres essentiels
    with st.expander("Paramètres essentiels", expanded=True):
        requester_ms_default = st.text_input("Requester MS (défaut 🇫🇷)", "FR")
        requester_vat_default = st.text_input("Requester VAT (défaut)", "")

    # Options avancées
    with st.expander("Options avancées (lots & cadence)", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            delay = st.number_input("Delay (sec)", min_value=0.0, value=1.5, step=0.1, help="Délai entre requêtes")
            retries = st.number_input("Retries", min_value=1, max_value=5, value=2, step=1, help="Tentatives en cas d’échec")
            valid_only = st.checkbox("Garder uniquement les valides", value=False)
        with colB:
            chunk_size = st.number_input("Chunk size", min_value=1, max_value=500, value=10, step=1, help="Taille d’un lot")
            chunk_pause = st.number_input("Chunk pause (sec)", min_value=0.0, value=3.0, step=0.5, help="Pause entre lots")
            timeout = st.number_input("Timeout (sec)", min_value=1.0, max_value=60.0, value=10.0, step=1.0)

    # Validation Pydantic des paramètres
    try:
        settings = AppSettings(
            requester_ms_default=requester_ms_default,
            requester_vat_default=requester_vat_default,
            delay=delay, retries=retries,
            chunk_size=chunk_size, chunk_pause=chunk_pause,
            valid_only=valid_only, timeout=timeout 
        )
    except Exception as e:
        st.error(f"Paramètres invalides : {e}")
        st.stop()

# ----- Guide rapide -----
with st.expander("📘 Guide rapide (à lire une fois)", expanded=False):
    st.markdown(
        """
1. **Charge un Excel** contenant au minimum par défaut `MS Code` (FR, DE, …) et `VAT Number`.  
2. Vérifie l’**aperçu** puis lance **Démarrer**.  
3. Consulte les **statistiques**, les **résultats** et **télécharge** Excel/CSV.  
4. Ajuste les **Options avancées** (sidebar) si tu rencontres des erreurs 429/timeout.

**Optionnelles**  
- `Requester MS Code`  
- `Requester VAT Number`  

Si absentes, les valeurs par défaut de la barre latérale seront utilisées.

API_URL = "https://ec.europa.eu/taxation_customs/vies/rest-api/check-vat-number"
        """
    )


# ----- Upload -----
st.subheader("1. Chargement du fichier Excel")
uploaded = st.file_uploader("Charger un fichier .xlsx contenant les Codes TVA à vérifier", type=["xlsx"])


if uploaded is not None:
    try:
        df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Impossible de lire le fichier Excel : {e}")
        st.stop()

    with st.expander("Aperçu du fichier (top 50)", expanded=True):
        st.dataframe(df_raw.head(50), use_container_width=True)

    valid_df, prep_errors = validate_and_prepare_df(df_raw, settings)

    if prep_errors:
        with st.expander("Erreurs / Notes de préparation", expanded=True):
            for msg in prep_errors:
                st.info(f"• {msg}")

    if valid_df.empty:
        st.error("Aucune ligne exploitable après nettoyage/validation.")
        st.stop()

    st.success(f"{len(valid_df)} ligne(s) prête(s) à être traitée(s).")

    st.subheader("2. Vérification des N° TVA dans la base VIES")
    if st.button("🚀 Lancer la vérification"):
        def _status(msg: str):
            st.session_state["_last_status"] = msg

        log("Début du traitement")
        with st.status("Traitement en cours…", expanded=False) as s:
            out_df = process_batch(valid_df, settings, status_cb=_status, log_cb=log)
            s.update(state="complete", label="Terminé ✅")
        log("Traitement terminé")

        # ----- Stats -----
        with st.expander("3) Statistiques globales", expanded=True):
            if "valid" in out_df.columns and len(out_df) > 0:
                total_check = len(out_df)
                valid_count = int(out_df["valid"].fillna(False).sum())
                invalid_count = total_check - valid_count
                pct = (valid_count / total_check * 100) if total_check else 0.0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total vérifiés", f"{total_check}")
                c2.metric("Valides", f"{valid_count}")
                c3.metric("Invalides", f"{invalid_count}")
                c4.metric("% Valides", f"{pct:.1f}%")
            else:
                st.write("Pas de métriques disponibles.")

        # ----- Résultats -----
        with st.expander("4) Résultats (tableau)", expanded=True):
            st.dataframe(out_df, use_container_width=True)

        # ----- Export -----
        with st.expander("5) Export (Excel / CSV)", expanded=True):
            base = (uploaded.name.rsplit(".", 1)[0]) if uploaded.name else "export"
            xlsx_bytes, csv_bytes, xlsx_name, csv_name = export_bytes(out_df, base, settings.valid_only)
            colx, coly = st.columns(2)
            with colx:
                st.download_button("⬇️ Télécharger Excel", data=xlsx_bytes, file_name=xlsx_name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with coly:
                st.download_button("⬇️ Télécharger CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")

        # ----- Journal -----
        with st.expander("🖥️ Journal d’exécution (logs)", expanded=False):
            if st.session_state["_logs"]:
                st.code("\n".join(st.session_state["_logs"]), language="text")
            else:
                st.write("Aucun log à afficher.")

else:
    st.info("Charge un fichier Excel pour commencer.")
