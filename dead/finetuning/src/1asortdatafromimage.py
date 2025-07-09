from pdf2image import convert_from_path
import pytesseract
import os

pdf_file = "../dataset/peyote_dance_0.pdf"
output_file = "ocr_output.txt"
tmp_dir = "ocr_tmp"
os.makedirs(tmp_dir, exist_ok=True)

print("Converting PDF to images...")
images = convert_from_path(pdf_file, output_folder=tmp_dir)

print("Running OCR on images...")
text = ""
for i, img in enumerate(images):
    page_text = pytesseract.image_to_string(img, lang='eng')
    text += f"\n\n--- Page {i+1} ---\n\n{page_text}"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ OCR complete. Output saved to {output_file}")
