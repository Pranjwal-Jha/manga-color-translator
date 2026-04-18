#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import main

TRANSLATED_TEXT = [
    "She is the Demon of Famine.",
    "Are you serious?",
    "The real Kiga-chan!",
    "Heh.",
    "Oh?!",
    "Demon of Famine...",
    "And...",
    "Since I am the Demon of Death, I'm the Death's Intimate Demon, so... Se-chan!",
    "What?! Aya?!",
    "Where did Yor go?!",
    "Se-chan."
]

def get_font_path():
    font_name = "AnimeAce.ttf"
    if not os.path.exists(font_name):
        import urllib.request
        url = "https://raw.githubusercontent.com/localizator/ukrainian-fonts-pack/master/AnimeAcev02%20-%20Anime%20Ace%20v02%20-%20Regular.ttf"
        try:
            print("Downloading Manga font (Anime Ace)...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(font_name, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download font: {e}")
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return font_name

def get_bubble_bounds(img_bgr, x1, y1, x2, y2):
    """
    Extract exact speech bubble bounds using flood-fill from erased image
    as done in manga-image-translator.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Smooth a bit and use a strict threshold since eroded images usually have pure white (#FFFFFF)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
    
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    start_point = (cx, cy)
    if thresh[cy, cx] != 255:
        # Search locally for nearest white pixel to flood fill
        for r in range(1, 40):
            found = False
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and thresh[ny, nx] == 255:
                        start_point = (nx, ny)
                        found = True
                        break
                if found: break
            if found: break
            
    if thresh[start_point[1], start_point[0]] == 255:
        cv2.floodFill(thresh, mask, start_point, 128)
        bubble_mask = (thresh == 128).astype(np.uint8)
        coords = cv2.findNonZero(bubble_mask)
        
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            
            # If the floodfilled region is the entire page (unsealed bubble)
            # fallback to the original box.
            if bw * bh > (w * h * 0.8) or bw > (w * 0.5) or bh > (h * 0.5):
                return x1, y1, x2, y2
                
            return bx, by, bx + bw, by + bh
            
    return x1, y1, x2, y2

def wrap_text(text, font, max_width):
    words = text.split()
    if not words: return []
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = current_line + " " + word
        if font.getlength(test_line) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def fit_text(text, box_w, box_h, font_path):
    """Binary search font size, using a safe margin to avoid curved speech bubble edges."""
    min_size = 8
    max_size = 120
    best_size = min_size
    best_lines = [text]
    
    # 75% margin ensures text fits into elliptical bubbles without spilling out the corners
    safe_target_w = box_w * 0.75
    safe_target_h = box_h * 0.75
    
    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2
        try:
            font = ImageFont.truetype(font_path, mid_size)
        except IOError:
            font = ImageFont.load_default()
            return font, [text], 12
            
        lines = wrap_text(text, font, safe_target_w)
        
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        spacing = mid_size * 0.15
        
        total_h = len(lines) * line_height + max(0, len(lines) - 1) * spacing
        max_line_w = max([font.getlength(line) for line in lines]) if lines else 0
        
        if max_line_w <= safe_target_w and total_h <= safe_target_h:
            best_size = mid_size
            best_lines = lines
            min_size = mid_size + 1
        else:
            max_size = mid_size - 1
            
    try:
        best_font = ImageFont.truetype(font_path, best_size)
    except IOError:
        best_font = ImageFont.load_default()
        
    return best_font, best_lines, best_size

def run():
    print("Loading image and detecting boxes...")
    img_bgr = cv2.imread("chainsaw_man_1.png")
    img_h, img_w = img_bgr.shape[:2]
    
    erased_bgr = cv2.imread("erased.png")
    
    raw_boxes = main.detect_text_regions(img_bgr, gpu=True)
    boxes = main.filter_boxes(raw_boxes, img_w, img_h)
    boxes = main.nms(boxes, threshold=0.4)
    boxes = main.filter_furigana(boxes, img_h)
    boxes = main.merge_nearby_boxes(boxes, gap_ratio=0.05)
    boxes = main.sort_manga_order(boxes, img_h)
    
    font_path = get_font_path()
    print(f"Using font: {font_path}")
    
    target_img = Image.open("erased.png").convert("RGB")
    draw = ImageDraw.Draw(target_img)
    
    for i, (box, text) in enumerate(zip(boxes, TRANSLATED_TEXT)):
        ox1, oy1, ox2, oy2 = box
        
        # 1. Bubble Extraction (manga-image-translator style)
        bx1, by1, bx2, by2 = get_bubble_bounds(erased_bgr, ox1, oy1, ox2, oy2)
        bw = bx2 - bx1
        bh = by2 - by1
        
        # Safe fallback logic in case original OCR bound is somehow larger
        obw = ox2 - ox1
        obh = oy2 - oy1
        if bw < obw or bh < obh:
            bw, bh = obw, obh
            bx1, by1, bx2, by2 = ox1, oy1, ox2, oy2
        
        # 2. Fit Text inside precise bubble bounds
        font, lines, font_size = fit_text(text, bw, bh, font_path)
        
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        spacing = font_size * 0.15
        
        total_h = len(lines) * line_height + max(0, len(lines) - 1) * spacing
        current_y = by1 + (bh - total_h) / 2
        
        stroke_width = max(1, int(font_size * 0.05))
        
        for line in lines:
            line_w = font.getlength(line)
            current_x = bx1 + (bw - line_w) / 2
            
            draw.text(
                (current_x, current_y), 
                line, 
                fill="black", 
                font=font, 
                stroke_width=stroke_width, 
                stroke_fill="white"
            )
            current_y += line_height + spacing

    target_img.save("translated.png")
    print("Saved translated.png successfully!")

if __name__ == "__main__":
    run()
