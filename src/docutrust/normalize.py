FIELD_TYPES = {
    "merchant_name": "TEXT",
    "invoice_date": "DATE",
    "invoice_number": "IDENTIFIER",
    "total_amount": "AMOUNT",
    "cgst_amount": "AMOUNT",
    "sgst_amount": "AMOUNT",
}
def label(predicted, gold, field_name):
    # 1. dono None?
    # 2. gold hai, predicted None?
    # 3. gold None, predicted kuch hai?
    # 4. dono hain — type ke hisaab se compare
    if predicted is None and gold is None:
        return "CORRECT"
    if predicted is None:
        return "MISS"
    if gold is None:
        return "WRONG"
    ftype = FIELD_TYPES[field_name]
    if ftype == "AMOUNT":
        p, g = normalize_amount(predicted), normalize_amount(gold)
    else:
        p, g = normalize_text(predicted), normalize_text(gold)
    if p is None or g is None:
        return "WRONG"
    return "CORRECT" if p == g else "WRONG"

def normalize_text(s):
    return " ".join(s.lower().split())

def normalize_amount(s):
    s = s.replace("₹", "").replace("Rs", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None