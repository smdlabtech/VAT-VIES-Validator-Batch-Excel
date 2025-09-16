#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import requests
import io
from enum import Enum


# ------------------------- Modèles Pydantic -------------------------
class VatRequest(BaseModel):
    """Modèle pour une requête de validation TVA"""
    country_code: str = Field(..., alias="MS Code")
    vat_number: str = Field(..., alias="VAT Number")
    requester_country_code: Optional[str] = Field(None, alias="Requester MS Code")
    requester_vat_number: Optional[str] = Field(None, alias="Requester VAT Number")
    
    @validator('country_code', 'requester_country_code')
    def format_country_code(cls, v):
        if v is None:
            return ""
        return str(v).strip().upper()
    
    @validator('vat_number', 'requester_vat_number')
    def clean_vat_number(cls, v):
        if v is None:
            return ""
        return str(v).replace(" ", "").replace(".", "").replace("-", "").upper()


class VatResponse(BaseModel):
    """Modèle pour la réponse de l'API VIES"""
    valid: bool
    name: Optional[str] = None
    address: Optional[str] = None
    error: Optional[str] = None
    timestamp: str


class ProcessingConfig(BaseModel):
    """Configuration du traitement"""
    requester_ms_default: str = "FR"
    requester_vat_default: str = ""
    delay: float = 1.5
    retries: int = 2
    chunk_size: int = 10
    chunk_pause: float = 3.0
    valid_only: bool = False


# ------------------------- Configuration Streamlit -------------------------
st.set_page_config(
    page_title="VIES TVA Validator",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "https://ec.europa.eu/taxation_customs/vies/rest-api/check-vat-number"


# ------------------------- Utilitaires -------------------------
def check_vat(ms_code: str, vat_number: str, requester_code: str, requester_vat: str, timeout: int = 10) -> Dict[str, Any]:
    """Vérifier un numéro TVA via l'API VIES"""
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
        else:
            return {
                "valid": False,
                "error": f"HTTP {r.status_code}",
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def ordered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ordonner les colonnes du DataFrame de résultat"""
    preferred = [
        "MS Code", "VAT Number", "valid", "name", "address",
        "Requester MS Code", "Requester VAT Number",
        "Attempts", "timestamp"
    ]
    cols = [c for c in preferred if c in df.columns]
    others = [c for c in df.columns if c not in cols]
    return df[cols + others]


def process_dataframe(
    df: pd.DataFrame,
    config: ProcessingConfig
) -> pd.DataFrame:
    """Traiter un DataFrame avec les numéros TVA"""
    # Vérification colonnes minimales
    required = ["MS Code", "VAT Number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {', '.join(missing)}")
        return pd.DataFrame()
    
    # Normalisation colonnes
    df = df.rename(columns=lambda x: str(x).strip())
    if "Requester MS Code" not in df.columns:
        df["Requester MS Code"] = config.requester_ms_default
    if "Requester VAT Number" not in df.columns:
        df["Requester VAT Number"] = config.requester_vat_default
    
    # Nettoyage champs
    df["MS Code"] = df["MS Code"].apply(lambda x: format_country_code(x))
    df["VAT Number"] = df["VAT Number"].apply(lambda x: clean_vat_number(x))
    df["Requester MS Code"] = df["Requester MS Code"].apply(lambda x: format_country_code(x))
    df["Requester VAT Number"] = df["Requester VAT Number"].apply(lambda x: clean_vat_number(x))
    
    # Lignes valides minimales
    valid_df = df.dropna(subset=["MS Code", "VAT Number"])
    dropped = len(df) - len(valid_df)
    if dropped > 0:
        st.warning(f"{dropped} lignes ignorées (code pays ou n° TVA manquant).")
    
    total = len(valid_df)
    if total == 0:
        st.info("Aucune ligne à traiter.")
        return pd.DataFrame()
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    chunks = [valid_df[i:i + config.chunk_size] for i in range(0, total, config.chunk_size)]
    processed = 0
    
    for cidx, chunk in enumerate(chunks, start=1):
        status_text.text(f"Traitement du lot {cidx}/{len(chunks)}")
        
        for i, row in chunk.iterrows():
            ms = row["MS Code"]
            vat = row["VAT Number"]
            req_ms = row["Requester MS Code"]
            req_vat = row["Requester VAT Number"]
            
            # Retries
            attempt = 0
            res = None
            while attempt < max(1, int(config.retries)):
                res = check_vat(ms, vat, req_ms, req_vat)
                # Succès si pas d'erreur OU si valid=True
                if ("error" not in res) or res.get("valid", False):
                    break
                attempt += 1
                time.sleep(config.delay * 1.5)
            
            # Ajouts
            res = res or {}
            res["MS Code"] = ms
            res["VAT Number"] = vat
            res["Requester MS Code"] = req_ms
            res["Requester VAT Number"] = req_vat
            res["Attempts"] = attempt + 1
            results.append(res)
            
            processed += 1
            progress_bar.progress(processed / total)
            
            time.sleep(config.delay)
        
        # Pause entre lots
        if cidx < len(chunks) and config.chunk_pause > 0:
            time.sleep(config.chunk_pause)
    
    # Résultats -> DataFrame
    out_df = pd.DataFrame(results)
    
    if config.valid_only and "valid" in out_df.columns:
        out_df = out_df[out_df["valid"] == True].copy()
        st.info(f"Filtrage : {len(out_df)} résultats valides conservés.")
    
    # Colonnes ordonnées si possible
    try:
        out_df = ordered_columns(out_df)
    except Exception:
        pass
    
    # Statistiques simples
    if "valid" in out_df.columns:
        total_check = len(out_df)
        valid_count = int(out_df["valid"].sum())
        invalid_count = total_check - valid_count
        pct = (valid_count / total_check * 100) if total_check else 0
        
        st.success(
            f"**Résultats:** Total: {total_check} | "
            f"Valides: {valid_count} | Invalides: {invalid_count} | "
            f"{pct:.1f}% valides"
        )
    
    progress_bar.empty()
    status_text.empty()
    
    return out_df


def format_country_code(code):
    """Formatter un code pays"""
    if pd.isna(code):
        return ""
    return str(code).strip().upper()


def clean_vat_number(vat):
    """Nettoyer un numéro TVA"""
    if pd.isna(vat):
        return ""
    return str(vat).replace(" ", "").replace(".", "").replace("-", "").upper()


# ------------------------- Interface Streamlit -------------------------
def main():
    st.title("🇪🇺 VIES TVA Validator")
    st.markdown("""
    Cette application permet de valider des numéros de TVA en masse via l'API VIES de l'Union Européenne.
    Téléchargez un fichier Excel avec les colonnes requises et lancez la validation.
    """)
    
    # Section d'information
    with st.expander("ℹ️ Instructions d'utilisation"):
        st.markdown("""
        ### Format du fichier Excel attendu:
        
        **Colonnes requises:**
        - `MS Code`: Code pays (ex: FR, DE, IT, ES)
        - `VAT Number`: Numéro de TVA à valider
        
        **Colonnes optionnelles:**
        - `Requester MS Code`: Code pays du demandeur
        - `Requester VAT Number`: Numéro TVA du demandeur
        
        Si les colonnes optionnelles ne sont pas présentes, les valeurs par défaut définies dans la sidebar seront utilisées.
        
        ### À propos de l'API VIES:
        - L'API VIES est fournie par la Commission Européenne
        - Les validations sont effectuées en temps réel
        - Un délai entre les requêtes est nécessaire pour respecter les limites de l'API
        """)
    
    # Sidebar avec les paramètres
    with st.sidebar:
        st.header("Paramètres de validation")
        
        requester_ms_default = st.text_input("Code pays demandeur par défaut", "FR")
        requester_vat_default = st.text_input("N° TVA demandeur par défaut", "")
        
        delay = st.slider("Délai entre requêtes (secondes)", 0.5, 5.0, 1.5, 0.5)
        retries = st.slider("Tentatives en cas d'échec", 1, 5, 2)
        chunk_size = st.slider("Taille des lots", 5, 50, 10)
        chunk_pause = st.slider("Pause entre lots (secondes)", 1.0, 10.0, 3.0, 0.5)
        valid_only = st.checkbox("Exporter uniquement les valides")
        
        config = ProcessingConfig(
            requester_ms_default=requester_ms_default,
            requester_vat_default=requester_vat_default,
            delay=delay,
            retries=retries,
            chunk_size=chunk_size,
            chunk_pause=chunk_pause,
            valid_only=valid_only
        )
    
    # Section de téléchargement du fichier
    st.header("1. Chargement du fichier")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel",
        type=["xlsx"],
        help="Le fichier doit contenir au moins les colonnes 'MS Code' et 'VAT Number'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Fichier chargé avec succès: {uploaded_file.name}")
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Détection des colonnes
            st.write("Colonnes détectées:")
            st.write(list(df.columns))
            
            # Bouton de lancement du traitement
            if st.button("🚀 Lancer la validation", type="primary"):
                with st.spinner("Validation en cours..."):
                    result_df = process_dataframe(df, config)
                
                if not result_df.empty:
                    st.subheader("Résultats de la validation")
                    st.dataframe(result_df)
                    
                    # Export des résultats
                    st.subheader("Télécharger les résultats")
                    
                    # Format Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Résultats VIES')
                    
                    st.download_button(
                        label="📥 Télécharger en Excel",
                        data=output.getvalue(),
                        file_name=f"resultats_vies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Format CSV
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger en CSV",
                        data=csv,
                        file_name=f"resultats_vies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {str(e)}")
    
    # Pied de page
    st.markdown("---")
    st.caption("VIES TVA Validator - Utilise l'API VIES de l'Union Européenne")


if __name__ == "__main__":
    main()