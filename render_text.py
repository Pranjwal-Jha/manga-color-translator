import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import main

TRANSLATED_TEXT = [
    "Dash!!",
    "Ah...",
    "How could I not smell it before?!",
    "Run, run, run!",
    "Beam!!",
    "Eh?!",
    "That guy is dangerous, Chainsaw Man!",
    "Got an odor! It's a bomb, damn it!",
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

def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0):
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0: return [0, 0, 0, 0]
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w - 1)
    rect[1::2] = np.clip(rect[1::2], 0, im_h - 1)
    return rect.tolist()

def get_bubble_bounds(img_bgr, ox1, oy1, ox2, oy2):
    """
    Extract exact speech bubble bounds using Canny edges + flood-fill
    Stolen directly from manga-image-translator!
    """
    im_h, im_w = img_bgr.shape[:2]
    w = ox2 - ox1
    h = oy2 - oy1

    enlarge_ratio = 2.5
    aspect = h / max(1, w)
    x1, y1, x2, y2 = enlarge_window([ox1, oy1, ox2, oy2], im_w, im_h, enlarge_ratio, aspect_ratio=aspect)

    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ox1, oy1, ox2, oy2

    ch, cw = crop.shape[:2]
    crop_area = ch * cw

    blurred = cv2.GaussianBlur(crop, (3,3), cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blurred, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(edges, (0, 0), (cw-1, ch-1), (255,255,255), 1, cv2.LINE_8)

    cons, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros((ch, cw), np.uint8)
    seedpnt = (cw // 2, ch // 2)
    difres = 10

    best_mask = np.zeros((ch, cw), np.uint8)
    min_retval = np.inf

    for i in range(len(cons)):
        rect = cv2.boundingRect(cons[i])
        if rect[2]*rect[3] < crop_area * 0.4:
            continue

        mask = cv2.drawContours(mask, cons, i, (255,), 2)
        cpmask = mask.copy()
        cv2.rectangle(mask, (0, 0), (cw-1, ch-1), (255,), 1, cv2.LINE_8)

        retval, _, _, _ = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt, flags=4, newVal=(127,), loDiff=(difres,difres,difres), upDiff=(difres,difres,difres))

        if retval <= crop_area * 0.3:
            mask = cv2.drawContours(mask, cons, i, (0,), 2)
        elif retval < min_retval and retval > crop_area * 0.3:
            min_retval = retval
            best_mask = cpmask

    best_mask = 127 - best_mask
    best_mask = cv2.dilate(best_mask, np.ones((3,3), np.uint8), iterations=1)

    ballon_area, _, _, _ = cv2.floodFill(best_mask, mask=None, seedPoint=seedpnt, flags=4, newVal=(30,), loDiff=(difres,difres,difres), upDiff=(difres,difres,difres))
    best_mask = 30 - best_mask
    _, best_mask = cv2.threshold(best_mask, 1, 255, cv2.THRESH_BINARY)
    best_mask = cv2.bitwise_not(best_mask, best_mask)

    box_kernel = int(np.sqrt(ballon_area) / 30)
    if box_kernel > 1:
        kernel = np.ones((box_kernel, box_kernel), np.uint8)
        best_mask = cv2.dilate(best_mask, kernel, iterations=1)
        best_mask = cv2.erode(best_mask, kernel, iterations=1)

    coords = cv2.findNonZero(best_mask)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        return bx + x1, by + y1, bx + bw + x1, by + bh + y1

    return ox1, oy1, ox2, oy2

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
    parser = argparse.ArgumentParser(description="Render translated text onto erased manga image.")
    parser.add_argument("--image", required=True, help="Original image for generating boxes")
    parser.add_argument("--erased", required=True, help="Erased image to paste text onto")
    parser.add_argument("--output", default="translated.png", help="Output file path")
    args = parser.parse_args()

    print(f"Loading image '{args.image}' and detecting boxes...")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None: raise FileNotFoundError(f"Could not read {args.image}")
    img_h, img_w = img_bgr.shape[:2]

    erased_bgr = cv2.imread(args.erased)
    if erased_bgr is None: raise FileNotFoundError(f"Could not read {args.erased}")

    if img_bgr.shape[:2] != erased_bgr.shape[:2]:
        print(f"Warning: Dimensions mismatch! Image is {img_w}x{img_h} but erased is {erased_bgr.shape[1]}x{erased_bgr.shape[0]}. Resizing erased background.")
        erased_bgr = cv2.resize(erased_bgr, (img_w, img_h))

    raw_boxes = main.detect_text_regions(img_bgr, gpu=True)
    boxes = main.filter_boxes(raw_boxes, img_w, img_h)
    boxes = main.nms(boxes, threshold=0.4)
    boxes = main.filter_furigana(boxes, img_h)
    boxes = main.merge_nearby_boxes(boxes, gap_ratio=0.05)
    boxes = main.sort_manga_order(boxes, img_h)

    font_path = get_font_path()
    print(f"Using font: {font_path}")

    target_img = Image.fromarray(cv2.cvtColor(erased_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(target_img)

    for i, (box, text) in enumerate(zip(boxes, TRANSLATED_TEXT)):
        ox1, oy1, ox2, oy2 = box

        # 1. Bubble Extraction (manga-image-translator style)
        bx1, by1, bx2, by2 = get_bubble_bounds(img_bgr, ox1, oy1, ox2, oy2)
        bw = bx2 - bx1
        bh = by2 - by1

        # Safe fallback logic to ensure horizontal text fits properly (Japanese OCR text is vertical and narrow)
        obw = ox2 - ox1
        obh = oy2 - oy1
        if bw < obw or bh < obh:
            best_size = max(obw, obh)
            cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            bw, bh = best_size, best_size
            bx1, by1 = cx - bw // 2, cy - bh // 2
            bx2, by2 = bx1 + bw, by1 + bh

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

    target_img.save(args.output)
    print(f"Saved {args.output} successfully!")

if __name__ == "__main__":
    run()
