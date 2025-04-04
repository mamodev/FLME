def validate_int( v): # (Same as before)
    if v == "" or v == "-": return True
    try: int(v); return True
    except ValueError: return False

def validate_float( v): # (Same as before)
    if v in ["", "-", "."]: return True
    if v.count('.') > 1 or v.count('-') > 1: return False
    if v == "-.": return True
    if v.endswith('.') and v.count('.') == 1:
            if v.startswith('-') and len(v) == 2: return True
            if not v.startswith('-') and len(v) == 1: return True
            if v[:-1].replace('-','').isdigit(): return True
    try:
        if v != "-": float(v)
        return True
    except ValueError: return False
