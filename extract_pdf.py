import pymupdf
import sys, os

sys.stdout.reconfigure(encoding='utf-8')

pdf_dir = r'C:\Users\dines\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\8F5E11ABFBCC7E3A2D00398175A05495F2553131\transfers\2026-16'

for f in os.listdir(pdf_dir):
    if f.endswith('.pdf') and 'CodeAlpha' in f:
        path = os.path.join(pdf_dir, f)
        doc = pymupdf.open(path)
        for page in doc:
            print(page.get_text())
        break
