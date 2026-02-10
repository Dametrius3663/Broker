import requests, os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# --- CONFIG ---
API_KEY = "52044214-913155381c2b0c709d0dc8f22"  # get from https://pixabay.com/api/docs/
OUTPUT_PDF = "forest_items.pdf"
IMAGE_DIR = "forest_images"
ITEMS = [
    "Pinecone","Owl","Acorn","Deer Track","Stick",
    "Fern","Mushroom","Spider Web","Squirrel","Feather",
    "Berry","River Rock","Tree Trunk","Bear Paw Print","Fox",
    "Leaf","Butterfly","Bird Nest","Tree Bark","Moss",
    "Ant Hill","Frog","Caterpillar","Flower","Forest Free Space"
]

# --- Ensure folder ---
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Download an image for each item ---
def fetch_image(query, filename):
    url = f"https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=illustration&per_page=3"
    r = requests.get(url).json()
    if r["hits"]:
        img_url = r["hits"][0]["webformatURL"]
        img_data = requests.get(img_url).content
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"Downloaded: {filename}")
    else:
        print(f"No image found for {query}")

for item in ITEMS:
    safe_name = item.replace(" ", "_").lower() + ".jpg"
    filepath = os.path.join(IMAGE_DIR, safe_name)
    if not os.path.exists(filepath):
        fetch_image(item, filepath)

# --- Create PDF with 1"×1" images ---
def make_pdf(items, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    margin = 0.5 * inch
    cell_size = 1.0 * inch
    spacing = 0.3 * inch
    page_w, page_h = letter

    x, y = margin, page_h - margin - cell_size
    for item in items:
        fname = item.replace(" ", "_").lower() + ".jpg"
        path = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(path):
            img = ImageReader(path)
            c.drawImage(img, x, y, width=cell_size, height=cell_size, preserveAspectRatio=True, anchor='c')
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + cell_size/2, y - 0.15*inch, item)

        x += cell_size + spacing
        if x + cell_size > page_w - margin:
            x = margin
            y -= cell_size + spacing + 0.25*inch
            if y < margin:
                c.showPage()
                x, y = margin, page_h - margin - cell_size
    c.save()

make_pdf(ITEMS, OUTPUT_PDF)
print(f"PDF created: {OUTPUT_PDF}")