
# Now let's handle the attack rate sweep summary files which have columns: p_vac, final_attack, csv, png
# We will extract attack rate as function of vaccination fraction

def extract_attackrate_sweep(df):
    # We just take p_vac and final_attack as is since they are already fractions
    return df[['p_vac', 'final_attack']].copy()