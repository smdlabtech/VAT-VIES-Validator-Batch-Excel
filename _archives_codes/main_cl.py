#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VIES TVA Validator (Streamlit App)
---------------------------------
Application web pour la vérification de numéros de TVA européens
via l'API VIES (Validation Information Exchange System)
"""

import io
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, validator


# ------------------------- Modèles Pydantic -------------------------

class VATValidationRequest(BaseModel):
    """Modèle pour une requête de validation TVA"""
    country_code: str = Field(..., min_length=2, max_length=2, description="Code pays (ex: FR, DE)")
    vat_number: str = Field(..., min_length=1, description="Numéro de TVA")
    requester_country_code: Optional[str] = Field(default="", description="Code pays du demandeur")
    requester_vat_number: Optional[str] = Field(default="", description="Numéro TVA du demandeur")
    
    @validator('country_code', 'requester_country_code')
    def validate_country_code(cls, v):
        if v and len(v.strip()) > 0:
            return v.strip().upper()
        return v
    
    @validator('vat_number', 'requester_vat_number')
    def clean_vat_number(cls, v):
        if not v:
            return ""
        return str(v).replace(" ", "").replace(".", "").replace("-", "").upper()


class VATValidationResponse(BaseModel):
    """Modèle pour une réponse de validation TVA"""
    ms_code: str
    vat_number: str
    valid: bool
    name: Optional[str] = None
    address: Optional[str] = None
    requester_ms_code: Optional[str] = None
    requester_vat_number: Optional[str] = None
    attempts: int = 1
    timestamp: str
    error: Optional[str] = None


class ProcessingConfig(BaseModel):
    """Configuration pour le traitement par lots"""
    requester_ms_default: str = Field(default="FR", description="Code pays par défaut")
    requester_vat_default: str = Field(default="", description="N° TVA par défaut")
    delay: float = Field(default=1.5, ge=0.1, le=10.0, description="Délai entre requêtes")
    retries: int = Field(default=2, ge=1, le=5, description="Nombre de tentatives")
    chunk_size: int = Field(default=10, ge=1, le=50, description="Taille des lots")
    chunk_pause: float = Field(default=3.0, ge=0.0, le=30.0, description="Pause entre lots")
    valid_only: bool = Field(default=False, description="Exporter uniquement les valides")
    timeout: int = Field(default=10, ge=5, le=30, description="Timeout des requêtes")


# ------------------------- Configuration -------------------------

API_URL = "https://ec.europa.eu/taxation_customs/vies/rest-api/check-vat-number"

# Configuration Streamlit
st.set_page_config(
    page_title="VIES TVA Validator",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------- Fonctions utilitaires -------------------------

@st.cache_data
def get_eu_countries():
    """Liste des codes pays de l'UE"""
    return {
        'AT': 'Autriche', 'BE': 'Belgique', 'BG': 'Bulgarie', 'CY': 'Chypre',
        'CZ': 'République tchèque', 'DE': 'Allemagne', 'DK': 'Danemark',
        'EE': 'Estonie', 'ES': 'Espagne', 'FI': 'Finlande', 'FR': 'France',
        'GR': 'Grèce', 'HR': 'Croatie', 'HU': 'Hongrie', 'IE': 'Irlande',
        'IT': 'Italie', 'LT': 'Lituanie', 'LU': 'Luxembourg', 'LV': 'Lettonie',
        'MT': 'Malte', 'NL': 'Pays-Bas', 'PL': 'Pologne', 'PT': 'Portugal',
        'RO': 'Roumanie', 'SE': 'Suède', 'SI': 'Slovénie', 'SK': 'Slovaquie'
    }


def check_vat_api(request: VATValidationRequest, timeout: int = 10) -> Dict[str, Any]:
    """Appel à l'API VIES pour vérifier un numéro de TVA"""
    payload = {
        "countryCode": request.country_code,
        "vatNumber": request.vat_number,
        "requesterCountryCode": request.requester_country_code or "",
        "requesterVatNumber": request.requester_vat_number or "",
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            data["timestamp"] = datetime.now().isoformat()
            return data
        else:
            return {
                "valid": False,
                "error": f"HTTP {response.status_code}",
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def process_dataframe(df: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    """Traite un DataFrame avec validation TVA"""
    # Vérification colonnes requises
    required_columns = ["MS Code", "VAT Number"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Colonnes manquantes : {', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # Normalisation des colonnes
    df = df.rename(columns=lambda x: str(x).strip())
    
    # Ajout des colonnes manquantes avec valeurs par défaut
    if "Requester MS Code" not in df.columns:
        df["Requester MS Code"] = config.requester_ms_default
    if "Requester VAT Number" not in df.columns:
        df["Requester VAT Number"] = config.requester_vat_default
    
    # Nettoyage des données
    df["MS Code"] = df["MS Code"].apply(lambda x: str(x).strip().upper() if pd.notna(x) else "")
    df["VAT Number"] = df["VAT Number"].apply(
        lambda x: str(x).replace(" ", "").replace(".", "").replace("-", "").upper() if pd.notna(x) else ""
    )
    df["Requester MS Code"] = df["Requester MS Code"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else ""
    )
    df["Requester VAT Number"] = df["Requester VAT Number"].apply(
        lambda x: str(x).replace(" ", "").replace(".", "").replace("-", "").upper() if pd.notna(x) else ""
    )
    
    # Filtrage des lignes valides
    valid_df = df.dropna(subset=["MS Code", "VAT Number"])
    valid_df = valid_df[(valid_df["MS Code"] != "") & (valid_df["VAT Number"] != "")]
    
    if len(valid_df) == 0:
        st.warning("Aucune ligne valide à traiter.")
        return pd.DataFrame()
    
    st.info(f"Traitement de {len(valid_df)} lignes valides sur {len(df)} au total.")
    
    return valid_df


def validate_batch(df: pd.DataFrame, config: ProcessingConfig) -> List[VATValidationResponse]:
    """Valide un lot de numéros de TVA"""
    results = []
    total = len(df)
    
    # Découpage en chunks
    chunks = [df[i:i + config.chunk_size] for i in range(0, total, config.chunk_size)]
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0
    
    for chunk_idx, chunk in enumerate(chunks, 1):
        status_text.text(f"Traitement du lot {chunk_idx}/{len(chunks)}")
        
        for i, row in chunk.iterrows():
            try:
                # Création de la requête
                request = VATValidationRequest(
                    country_code=row["MS Code"],
                    vat_number=row["VAT Number"],
                    requester_country_code=row.get("Requester MS Code", ""),
                    requester_vat_number=row.get("Requester VAT Number", "")
                )
                
                # Tentatives avec retry
                attempt = 0
                api_response = None
                
                while attempt < config.retries:
                    api_response = check_vat_api(request, config.timeout)
                    
                    # Succès si pas d'erreur ou si valide
                    if "error" not in api_response or api_response.get("valid", False):
                        break
                    
                    attempt += 1
                    if attempt < config.retries:
                        time.sleep(config.delay * 1.5)
                
                # Création de la réponse
                response = VATValidationResponse(
                    ms_code=request.country_code,
                    vat_number=request.vat_number,
                    valid=api_response.get("valid", False),
                    name=api_response.get("name"),
                    address=api_response.get("address"),
                    requester_ms_code=request.requester_country_code,
                    requester_vat_number=request.requester_vat_number,
                    attempts=attempt + 1,
                    timestamp=api_response.get("timestamp", datetime.now().isoformat()),
                    error=api_response.get("error")
                )
                
                results.append(response)
                
            except Exception as e:
                st.error(f"Erreur lors du traitement de la ligne {i}: {str(e)}")
                continue
            
            processed += 1
            progress_bar.progress(processed / total)
            
            # Délai entre les requêtes
            time.sleep(config.delay)
        
        # Pause entre les chunks
        if chunk_idx < len(chunks) and config.chunk_pause > 0:
            status_text.text(f"Pause de {config.chunk_pause}s entre les lots...")
            time.sleep(config.chunk_pause)
    
    progress_bar.progress(1.0)
    status_text.text("Traitement terminé !")
    
    return results


# ------------------------- Interface Streamlit -------------------------

def main():
    st.title("🇪🇺 VIES TVA Validator")
    st.markdown("**Validation de numéros de TVA européens via l'API VIES**")
    
    # Sidebar pour la configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Configuration par défaut
    config_data = {}
    config_data["requester_ms_default"] = st.sidebar.selectbox(
        "Code pays demandeur par défaut",
        options=list(get_eu_countries().keys()),
        index=list(get_eu_countries().keys()).index("FR")
    )
    config_data["requester_vat_default"] = st.sidebar.text_input(
        "N° TVA demandeur par défaut",
        value="",
        help="Laissez vide si non applicable"
    )
    
    st.sidebar.subheader("Paramètres de traitement")
    config_data["delay"] = st.sidebar.slider(
        "Délai entre requêtes (sec)",
        min_value=0.1, max_value=10.0, value=1.5, step=0.1
    )
    config_data["retries"] = st.sidebar.slider(
        "Nombre de tentatives",
        min_value=1, max_value=5, value=2
    )
    config_data["chunk_size"] = st.sidebar.slider(
        "Taille des lots",
        min_value=1, max_value=50, value=10
    )
    config_data["chunk_pause"] = st.sidebar.slider(
        "Pause entre lots (sec)",
        min_value=0.0, max_value=30.0, value=3.0, step=0.5
    )
    config_data["timeout"] = st.sidebar.slider(
        "Timeout requêtes (sec)",
        min_value=5, max_value=30, value=10
    )
    config_data["valid_only"] = st.sidebar.checkbox(
        "Exporter uniquement les valides",
        value=False
    )
    
    try:
        config = ProcessingConfig(**config_data)
    except Exception as e:
        st.sidebar.error(f"Erreur de configuration : {str(e)}")
        return
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📁 Upload Fichier", "🔍 Vérification individuelle", "📊 Résultats fichier excel chargé"])
    
    with tab1:
        # st.header("Upload du fichier Excel")
        with st.expander("📘 Guide rapide : charger votre fichier Excel contenant le N° TVA", expanded=False):
            st.markdown("""
            **Colonnes attendues dans votre fichier :**
            - `MS Code` : Code pays (FR, DE, IT, etc.)
            - `VAT Number` : Numéro de TVA
            - `Requester MS Code` (optionnel)
            - `Requester VAT Number` (optionnel)
            """)
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier Excel (.xlsx)",
            type=['xlsx'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier
                df = pd.read_excel(uploaded_file)
                st.success(f"Fichier chargé : {len(df)} lignes")
                
                # Aperçu des données
                with st.expander("Aperçu des données", expanded=False):
                    st.dataframe(df.head(10))
                    
                # Traitement
                if st.button("🚀 Démarrer la vérification", type="primary"):
                    processed_df = process_dataframe(df, config)
                    
                    if not processed_df.empty:
                        with st.spinner("Validation en cours..."):
                            results = validate_batch(processed_df, config)
                        
                        # Stockage des résultats dans la session
                        st.session_state['results'] = results
                        st.session_state['config'] = config
                        
                        st.success("Validation terminée ! Consultez l'onglet 'Résultats'.")
                        
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
    
    with tab2:
        # st.header("Vérification individuelle")
        col1, col2 = st.columns(2)
        
        with col1:
            country_options = get_eu_countries()
            selected_country = st.selectbox(
                "Code pays",
                options=list(country_options.keys()),
                format_func=lambda x: f"{x} - {country_options[x]}"
            )
            
            vat_number = st.text_input(
                "Numéro de TVA",
                value="",
                help="Saisissez le numéro sans le code pays"
            )
        
        with col2:
            req_country = st.selectbox(
                "Code pays demandeur (optionnel)",
                options=[""] + list(country_options.keys()),
                format_func=lambda x: f"{x} - {country_options[x]}" if x else "Non spécifié"
            )
            
            req_vat = st.text_input(
                "N° TVA demandeur (optionnel)",
                value=""
            )
        
        if st.button("Vérifier le N° TVA", type="primary"):
            if not vat_number:
                st.error("Veuillez saisir un numéro de TVA")
            else:
                try:
                    request = VATValidationRequest(
                        country_code=selected_country,
                        vat_number=vat_number,
                        requester_country_code=req_country,
                        requester_vat_number=req_vat
                    )
                    
                    with st.spinner("Vérification en cours..."):
                        result = check_vat_api(request, config.timeout)
                    
                    # Affichage du résultat
                    if result.get("valid", False):
                        st.success("✅ Numéro de TVA valide")
                        if result.get("name"):
                            st.info(f"**Nom :** {result['name']}")
                        if result.get("address"):
                            st.info(f"**Adresse :** {result['address']}")
                    else:
                        st.error("❌ Numéro de TVA invalide")
                        if result.get("error"):
                            st.warning(f"Erreur : {result['error']}")
                    
                except Exception as e:
                    st.error(f"Erreur lors de la validation : {str(e)}")
    
    with tab3:
        # st.header("Résultats de validation")
        
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            config_used = st.session_state.get('config', config)
            
            # Conversion en DataFrame pour affichage
            results_data = []
            for result in results:
                results_data.append({
                    "Code Pays": result.ms_code,
                    "N° TVA": result.vat_number,
                    "Valide": "✅" if result.valid else "❌",
                    "Nom": result.name or "",
                    "Adresse": result.address or "",
                    "Tentatives": result.attempts,
                    "Erreur": result.error or "",
                    "Timestamp": result.timestamp
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Statistiques
            total_count = len(results)
            valid_count = sum(1 for r in results if r.valid)
            invalid_count = total_count - valid_count
            valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", total_count)
            with col2:
                st.metric("Valides", valid_count)
            with col3:
                st.metric("Invalides", invalid_count)
            with col4:
                st.metric("% Valides", f"{valid_pct:.1f}%")
            
            # Filtrage des résultats
            if config_used.valid_only:
                filtered_df = results_df[results_df["Valide"] == "✅"]
                st.info(f"Affichage des {len(filtered_df)} résultats valides uniquement")
            else:
                filtered_df = results_df
            
            # Affichage des résultats
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export
            st.subheader("Export des résultats")
            
            # Préparation du fichier Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Résultats TVA')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultats_TVA_{timestamp}.xlsx"
            
            st.download_button(
                label="📥 Télécharger les résultats (Excel)",
                data=buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        else:
            st.info("Aucun résultat disponible. Uploadez un fichier et lancez la validation.")


if __name__ == "__main__":
    main()