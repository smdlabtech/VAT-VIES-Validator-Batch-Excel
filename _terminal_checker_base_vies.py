#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VIES TVA Validator (CLI)
------------------------
Usage basique :
    python vies_tva_validator.py --input mon_fichier.xlsx

Colonnes attendues dans le fichier Excel :
    - "MS Code" (ex: FR, DE, IT, ES)
    - "VAT Number"
Optionnel :
    - "Requester MS Code"
    - "Requester VAT Number"

Paramètres utiles :
    --requester-ms FR --requester-vat 12345678901   (défauts à utiliser si colonnes absentes)
    --delay 1.5                                      (délai entre appels API, en secondes)
    --retries 2                                      (tentatives en cas d’échec)
    --chunk-size 10 --chunk-pause 3.0                (traitement par lots)
    --valid-only                                     (exporter uniquement les numéros valides)
    --output resultats.xlsx                          (nom du fichier de sortie)
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests


API_URL = "https://ec.europa.eu/taxation_customs/vies/rest-api/check-vat-number"


# ------------------------- Utilitaires -------------------------
def clean_vat_number(vat):
    if pd.isna(vat):
        return ""
    return str(vat).replace(" ", "").replace(".", "").replace("-", "").upper()


def format_country_code(code):
    if pd.isna(code):
        return ""
    return str(code).strip().upper()


def check_vat(ms_code, vat_number, requester_code, requester_vat, timeout=10):
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


def ordered_columns(df):
    preferred = [
        "MS Code", "VAT Number", "valid", "name", "address",
        "Requester MS Code", "Requester VAT Number",
        "Attempts", "timestamp"
    ]
    cols = [c for c in preferred if c in df.columns]
    others = [c for c in df.columns if c not in cols]
    return df[cols + others]


# ------------------------- Traitement principal -------------------------
def process_file(
    input_path,
    requester_ms_default="FR",
    requester_vat_default="",
    delay=1.5,
    retries=2,
    chunk_size=10,
    chunk_pause=3.0,
    valid_only=False,
    output_path=None,
):
    # Chargement Excel
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier Excel : {e}")
        sys.exit(1)

    # Vérification colonnes minimales
    required = ["MS Code", "VAT Number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes : {', '.join(missing)}")
        sys.exit(1)

    # Normalisation colonnes
    df = df.rename(columns=lambda x: str(x).strip())
    if "Requester MS Code" not in df.columns:
        df["Requester MS Code"] = requester_ms_default
    if "Requester VAT Number" not in df.columns:
        df["Requester VAT Number"] = requester_vat_default

    # Nettoyage champs
    df["MS Code"] = df["MS Code"].apply(format_country_code)
    df["VAT Number"] = df["VAT Number"].apply(clean_vat_number)
    df["Requester MS Code"] = df["Requester MS Code"].apply(format_country_code)
    df["Requester VAT Number"] = df["Requester VAT Number"].apply(clean_vat_number)

    # Lignes valides minimales
    valid_df = df.dropna(subset=["MS Code", "VAT Number"])
    dropped = len(df) - len(valid_df)
    if dropped > 0:
        print(f"[INFO] {dropped} lignes ignorées (code pays ou n° TVA manquant).")

    total = len(valid_df)
    if total == 0:
        print("[INFO] Aucune ligne à traiter.")
        sys.exit(0)

    print(f"[INFO] {total} lignes à vérifier via VIES...")
    results = []

    # Découpage en lots
    chunks = [valid_df[i:i + chunk_size] for i in range(0, total, chunk_size)]
    processed = 0

    for cidx, chunk in enumerate(chunks, start=1):
        print(f"[LOT {cidx}/{len(chunks)}]")
        for i, row in chunk.iterrows():
            ms = row["MS Code"]
            vat = row["VAT Number"]
            req_ms = row["Requester MS Code"]
            req_vat = row["Requester VAT Number"]

            # Retries
            attempt = 0
            res = None
            while attempt < max(1, int(retries)):
                res = check_vat(ms, vat, req_ms, req_vat)
                # Succès si pas d'erreur OU si valid=True
                if ("error" not in res) or res.get("valid", False):
                    break
                attempt += 1
                time.sleep(delay * 1.5)

            # Ajouts
            res = res or {}
            res["MS Code"] = ms
            res["VAT Number"] = vat
            res["Requester MS Code"] = req_ms
            res["Requester VAT Number"] = req_vat
            res["Attempts"] = attempt + 1
            results.append(res)

            processed += 1
            if processed % 5 == 0 or processed == total:
                print(f"  - Progression : {processed}/{total}")

            time.sleep(delay)

        # Pause entre lots
        if cidx < len(chunks) and chunk_pause > 0:
            print(f"  Pause {chunk_pause}s entre lots...")
            time.sleep(chunk_pause)

    # Résultats -> DataFrame
    out_df = pd.DataFrame(results)

    if valid_only and "valid" in out_df.columns:
        out_df = out_df[out_df["valid"] == True].copy()
        print(f"[INFO] Filtrage : {len(out_df)} résultats valides conservés.")

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
        print(f"[STATS] Total: {total_check} | Valides: {valid_count} | Invalides: {invalid_count} | {pct:.1f}% valides")

    # Export Excel
    if not output_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(input_path))[0]
        suffix = "_valides" if valid_only else ""
        output_path = f"{base}_resultats_TVA{suffix}_{stamp}.xlsx"

    try:
        # Utilise openpyxl par défaut
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False)
        print(f"[OK] Résultats exportés : {output_path}")
    except Exception as e:
        print(f"[ERREUR] Export Excel échoué : {e}")
        # Fallback CSV
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        out_df.to_csv(csv_path, index=False)
        print(f"[OK] Export CSV de secours : {csv_path}")


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vérification VIES des numéros de TVA (batch Excel)")
    p.add_argument("--input", "-i", help="Chemin du fichier Excel à traiter (.xlsx)")
    p.add_argument("--requester-ms", default="FR", help="Code pays du demandeur par défaut (si colonne absente)")
    p.add_argument("--requester-vat", default="", help="N° TVA du demandeur par défaut (si colonne absente)")
    p.add_argument("--delay", type=float, default=1.5, help="Délai entre requêtes (secondes)")
    p.add_argument("--retries", type=int, default=2, help="Nombre de tentatives en cas d'échec")
    p.add_argument("--chunk-size", type=int, default=10, help="Taille des lots de traitement")
    p.add_argument("--chunk-pause", type=float, default=3.0, help="Pause entre lots (secondes)")
    p.add_argument("--valid-only", action="store_true", help="N’exporter que les résultats valides")
    p.add_argument("--output", "-o", help="Nom du fichier de sortie (.xlsx)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Si aucun fichier fourni, on demande via input() pour garder la simplicité
    input_path = args.input
    if not input_path:
        input_path = input("Chemin du fichier Excel (.xlsx) à traiter : ").strip()

    if not input_path or not os.path.exists(input_path):
        print("[ERREUR] Fichier introuvable. Utilise --input ou fournis un chemin valide.")
        sys.exit(1)

    process_file(
        input_path=input_path,
        requester_ms_default=args.requester_ms,
        requester_vat_default=args.requester_vat,
        delay=args.delay,
        retries=args.retries,
        chunk_size=args.chunk_size,
        chunk_pause=args.chunk_pause,
        valid_only=args.valid_only,
        output_path=args.output,
    )
