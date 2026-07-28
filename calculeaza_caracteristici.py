
import pandas as pd
import numpy as np
from supabase import create_client, Client

# 1. Conectare la Supabase (Opțional, dacă vrei să citești direct din DB)
# SUPABASE_URL = "https://your-project-ref.supabase.co"
# SUPABASE_KEY = "your-anon-or-service-role-key"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def calculeaza_caracteristici_ai(df):
    # Asigurăm ordonarea cronologică a meciurilor
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Dicționare pentru urmărirea istoricului fiecărei echipe
    echipa_meciuri = {}

    # Coloane noi de date pentru AI
    forma_gazda, forma_oaspete = [], []
    g_m_gazda, g_p_gazda = [], []
    g_m_oaspete, g_p_oaspete = [], []

    for idx, row in df.iterrows():
        home, away = row['Home'], row['Away']

        # Preluăm doar ultimele 5 meciuri jucate ÎNAINTE de meciul curent
        hist_home = echipa_meciuri.get(home, [])[-5:]
        hist_away = echipa_meciuri.get(away, [])[-5:]

        # Calculăm media de puncte și goluri
        if hist_home:
            pts_h = sum([m['puncte'] for m in hist_home]) / len(hist_home)
            gm_h = sum([m['gm'] for m in hist_home]) / len(hist_home)
            gp_h = sum([m['gp'] for m in hist_home]) / len(hist_home)
        else:
            pts_h, gm_h, gp_h = 1.0, 1.0, 1.0  # Valori neutre pentru meciurile de început

        if hist_away:
            pts_a = sum([m['puncte'] for m in hist_away]) / len(hist_away)
            gm_a = sum([m['gm'] for m in hist_away]) / len(hist_away)
            gp_a = sum([m['gp'] for m in hist_away]) / len(hist_away)
        else:
            pts_a, gm_a, gp_a = 1.0, 1.0, 1.0

        forma_gazda.append(pts_h)
        forma_oaspete.append(pts_a)
        g_m_gazda.append(gm_h)
        g_p_gazda.append(gp_h)
        g_m_oaspete.append(gm_a)
        g_p_oaspete.append(gp_a)

        # Calculăm rezultatul meciului curent și actualizăm istoricul
        hg, ag = row['HG'], row['AG']
        if hg > ag:
            pts_h_meci, pts_a_meci = 3, 0
        elif hg == ag:
            pts_h_meci, pts_a_meci = 1, 1
        else:
            pts_h_meci, pts_a_meci = 0, 3

        if home not in echipa_meciuri: echipa_meciuri[home] = []
        if away not in echipa_meciuri: echipa_meciuri[away] = []

        echipa_meciuri[home].append({'puncte': pts_h_meci, 'gm': hg, 'gp': ag})
        echipa_meciuri[away].append({'puncte': pts_a_meci, 'gm': ag, 'gp': hg})

    # Adăugăm noii indicatori în dataset
    df['forma_gazda_5m'] = forma_gazda
    df['forma_oaspete_5m'] = forma_oaspete
    df['g_marcate_gazda_medie'] = g_m_gazda
    df['g_primite_gazda_medie'] = g_p_gazda
    df['g_marcate_oaspete_medie'] = g_m_oaspete
    df['g_primite_oaspete_medie'] = g_p_oaspete

    return df

# Rulare directă pe fișierul ROU.csv
if __name__ == "__main__":
    df_raw = pd.read_csv("ROU.csv")
    df_prelucrat = calculeaza_caracteristici_ai(df_raw)
    df_prelucrat.to_csv("meciuri_prelucrate_ai.csv", index=False)
    print("Gata! Fișierul 'meciuri_prelucrate_ai.csv' a fost generat și este pregătit pentru antrenarea AI-ului.")
