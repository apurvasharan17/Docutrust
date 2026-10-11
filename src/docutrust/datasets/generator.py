"""Synthetic GST invoice generator.

make_fields(seed) -> gold field values for one document
render(fields, seed) -> PIL image of the invoice
main() -> writes data/synthetic/doc_XXX.png + doc_XXX.json

Same seed => same document, always (reproducibility).
"""
import json
import os
import random
from datetime import date, timedelta

from PIL import Image, ImageDraw, ImageFont

MERCHANTS = [
    "Sri Lakshmi Stores",
    "Balaji Super Mart",
    "Annapoorna Provisions",
    "Ganesh General Traders",
    "Kaveri Home Needs",
    "Nandini Fresh Mart",
    "Mysore Silk Emporium",
    "Venkateshwara Hardware",
    "Green Leaf Organics",
    "City Kitchen Appliances",
    "Royal Textiles",
    "Shree Krishna Sweets",
    "Metro Daily Needs",
    "Sai Medicals & General",
    "Udupi Grand Cafe",
]

ITEMS = [
    "Basmati Rice 5kg", "Toor Dal 1kg", "Sunflower Oil 1L", "Steel Tumbler Set",
    "Cotton Towel", "Pressure Cooker 3L", "LED Bulb 9W", "Detergent 2kg",
    "Tea Powder 500g", "Notebook Pack", "Non-stick Tawa", "Hand Wash 750ml",
]

GST_RATE = 0.09  # CGST 9% + SGST 9% = 18% total
START_DATE = date(2025, 4, 1)  # start of FY 2025-26
DATE_RANGE_DAYS = 540

WIDTH, HEIGHT = 600, 900
OUT_DIR = "data/synthetic"
N_DOCS = 50


# ---------------- Step 2: gold fields ----------------
def make_fields(seed):
    rng = random.Random(seed)

    merchant = rng.choice(MERCHANTS)

    d = START_DATE + timedelta(days=rng.randint(0, DATE_RANGE_DAYS))
    invoice_date = d.strftime("%d/%m/%Y")

    invoice_number = f"INV-{d.year}-{rng.randint(1, 99999):05d}"

    taxable = round(rng.uniform(100, 20000), 2)
    cgst = round(taxable * GST_RATE, 2)
    sgst = cgst
    total = round(taxable + cgst + sgst, 2)

    return {
        "doc_id": f"doc_{seed:03d}",
        "merchant_name": merchant,
        "invoice_date": invoice_date,
        "invoice_number": invoice_number,
        "total_amount": f"{total:.2f}",
        "cgst_amount": f"{cgst:.2f}",
        "sgst_amount": f"{sgst:.2f}",
    }


# ---------------- Step 3: render image ----------------
def split_amount(total_paise, n, rng):
    """Split an amount (in paise) into n positive parts that sum exactly."""
    cuts = sorted(rng.sample(range(1, total_paise), n - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [total_paise])]
    return parts


def render(fields, seed):
    # separate RNG stream so rendering never changes the gold fields
    rng = random.Random(seed * 1000 + 1)

    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    big = ImageFont.load_default(size=26)
    font = ImageFont.load_default(size=18)

    # taxable = total - cgst - sgst, done in paise to avoid float errors
    to_paise = lambda s: round(float(s) * 100)
    taxable_p = (to_paise(fields["total_amount"])
                 - to_paise(fields["cgst_amount"])
                 - to_paise(fields["sgst_amount"]))

    n_items = rng.randint(3, 5)
    names = rng.sample(ITEMS, n_items)
    prices = split_amount(taxable_p, n_items, rng)

    x, y, right = 40, 40, WIDTH - 40

    def line(text, f=font, gap=32):
        nonlocal y
        draw.text((x, y), text, fill="black", font=f)
        y += gap

    def row(label, value, f=font, gap=32):
        nonlocal y
        draw.text((x, y), label, fill="black", font=f)
        w = draw.textlength(value, font=f)
        draw.text((right - w, y), value, fill="black", font=f)
        y += gap

    def rule():
        nonlocal y
        draw.line((x, y, right, y), fill="black", width=1)
        y += 16

    line(fields["merchant_name"], big, 44)
    line("TAX INVOICE", font, 40)
    line(f"Invoice No: {fields['invoice_number']}")
    line(f"Date: {fields['invoice_date']}", gap=40)
    rule()

    for name, p in zip(names, prices):
        row(name, f"{p / 100:.2f}")
    rule()

    row("Taxable Amount", f"{taxable_p / 100:.2f}")
    row("CGST @ 9%", fields["cgst_amount"])
    row("SGST @ 9%", fields["sgst_amount"], gap=40)
    rule()
    row("TOTAL", fields["total_amount"], big, 44)

    line("Thank you! Visit again.")
    return img


# ---------------- Step 4: write dataset ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in range(1, N_DOCS + 1):
        fields = make_fields(seed)
        img = render(fields, seed)
        base = os.path.join(OUT_DIR, fields["doc_id"])
        img.save(base + ".png")
        with open(base + ".json", "w") as f:
            json.dump(fields, f, indent=2)
    print(f"wrote {N_DOCS} docs to {OUT_DIR}/")


if __name__ == "__main__":
    main()