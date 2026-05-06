import fitz
import pytesseract
from PIL import Image
import numpy as np
import cv2

doc = fitz.open("/home/morrty00/PROJECTS/nazli-EC/test_pdfs/EC_4_30.pdf") # I don't know the exact path. Wait, I should find the PDF file first.
