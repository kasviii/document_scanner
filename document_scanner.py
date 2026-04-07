"""

Demonstrates: Edge detection (Canny), Corner detection (Harris),
              Contour finding, Perspective warp (Homography)

Run: python document_scanner.py

Requirements:
    pip install opencv-python numpy Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0d14"
CARD    = "#14141f"
ACCENT  = "#7c6ff7"
ACCENT2 = "#4ecdc4"
TXT     = "#e8e8f0"
TXT_DIM = "#6b6b80"
BORDER  = "#252535"
GREEN   = "#56cf8a"
ORANGE  = "#f5a623"
RED     = "#f25f5c"

WIN_W, WIN_H     = 1280, 820
PANEL_W, PANEL_H = 380, 280

CAPTIONS = {
    "Original":
        "Your input photo — taken at any angle. The document does not need "
        "to be straight; the algorithm will correct it.",
    "Edges (Canny)":
        "Canny: Gaussian blur removes noise, then intensity gradients find "
        "boundaries. Two thresholds decide strong vs weak edges. Tune with sliders.",
    "Corners (Harris)":
        "Harris corner detector finds pixels where intensity changes sharply "
        "in ALL directions. Green dots = 4 document corners used for the warp.",
    "Scanned Output":
        "Perspective transform (homography) maps the 4 corners to a rectangle, "
        "giving a top-down flat scan — same math as CamScanner / Google PhotoScan.",
}


def make_demo_image():
    canvas = np.full((480, 640, 3), 55, dtype=np.uint8)
    pts = np.array([[80, 60], [560, 100], [520, 420], [60, 380]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (245, 245, 240))
    for i, y in enumerate(range(140, 370, 28)):
        x1 = 90 + i * 3;  x2 = 510 - i * 2;  yy = y + int(i * 1.5)
        cv2.line(canvas, (x1, yy), (x2, yy), (160, 160, 155), 2)
    cv2.rectangle(canvas, (90, 110), (440, 135), (190, 190, 185), -1)
    cv2.putText(canvas, "Load a real document photo via 'Load Photo'",
                (60, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 150), 1)
    return canvas


def to_tk(bgr, size=(PANEL_W, PANEL_H)):
    h, w   = bgr.shape[:2]
    scale  = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    small  = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s         = pts.sum(axis=1)
    rect[0]   = pts[np.argmin(s)]
    rect[2]   = pts[np.argmax(s)]
    diff      = np.diff(pts, axis=1)
    rect[1]   = pts[np.argmin(diff)]
    rect[3]   = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect            = order_points(pts)
    tl, tr, br, bl  = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst  = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]],
                    dtype=np.float32)
    M    = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def find_document_contour(edges):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:10]:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


class DocumentScanner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Document Scanner")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.original    = make_demo_image()
        self.warped      = None
        self.thresh_low  = tk.IntVar(value=50)
        self.thresh_high = tk.IntVar(value=150)
        self.blur_k      = tk.IntVar(value=5)
        self._refs       = []
        self._panels     = {}
        self._status     = tk.StringVar(value="Load a photo of a document to begin.")

        self._build_ui()
        self._process()

    def _build_ui(self):
        bar = tk.Frame(self, bg="#0a0a12", height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="Document Scanner",
                 font=("Georgia", 17, "bold"),
                 bg="#0a0a12", fg=TXT).pack(side="left", padx=20, pady=10)
        tk.Label(bar, text="Edge & Corner Detection",
                 font=("Courier", 10), bg="#0a0a12", fg=ACCENT).pack(side="left", padx=4)

        tk.Button(bar, text="Save Scan", command=self._save,
                  bg=GREEN, fg="#0d0d14", font=("Helvetica", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=8, pady=10)
        tk.Button(bar, text="Load Photo", command=self._load,
                  bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=4, pady=10)

        status_bar = tk.Frame(self, bg="#0a0a12", height=26)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self._status,
                 bg="#0a0a12", fg=TXT_DIM, font=("Courier", 9)).pack(
                 side="left", padx=16, pady=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        gf = tk.Frame(body, bg=BG)
        gf.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        steps = {"Original": "①", "Edges (Canny)": "②",
                 "Corners (Harris)": "③", "Scanned Output": "④"}
        for name, (r, c) in zip(steps.keys(),
                                 [(0,0),(0,1),(1,0),(1,1)]):
            self._panels[name] = self._card(gf, r, c, steps[name], name)

        self._build_sidebar(body)

    def _card(self, parent, row, col, num, name):
        f = tk.Frame(parent, bg=CARD,
                     highlightbackground=BORDER, highlightthickness=1)
        f.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(col, weight=1)

        tk.Label(f, text=f"{num}  {name}", bg=CARD, fg=ACCENT2,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=10, pady=(8,2))

        img_lbl = tk.Label(f, bg=CARD)
        img_lbl.pack(padx=6, pady=2)

        stat_lbl = tk.Label(f, text="", bg=CARD, fg=ORANGE, font=("Courier", 9))
        stat_lbl.pack(anchor="w", padx=10)

        tk.Label(f, text=CAPTIONS[name], bg=CARD, fg=TXT_DIM,
                 font=("Helvetica", 9), wraplength=PANEL_W-20,
                 justify="left").pack(anchor="w", padx=10, pady=(2,10))

        return img_lbl, stat_lbl

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=CARD, width=245,
                        highlightbackground=BORDER, highlightthickness=1)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="Canny Parameters", bg=CARD, fg=TXT,
                 font=("Georgia", 12, "bold")).pack(pady=(18,2), padx=16, anchor="w")
        tk.Label(side, text="These control which edges survive.\nToo low → noisy. Too high → missing edges.",
                 bg=CARD, fg=TXT_DIM, font=("Helvetica", 9),
                 justify="left").pack(padx=16, anchor="w")
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)

        self._val_lbls = {}
        sliders = [
            ("Low Threshold",  self.thresh_low,  0,   150, 1,
             "Weak edges below this are discarded"),
            ("High Threshold", self.thresh_high, 50,  300, 1,
             "Edges above this are always kept"),
            ("Blur Kernel",    self.blur_k,       1,   15, 2,
             "Odd number — larger = more noise removal"),
        ]
        for label, var, mn, mx, res, hint in sliders:
            grp = tk.Frame(side, bg=CARD)
            grp.pack(fill="x", padx=16, pady=(0,12))
            hdr = tk.Frame(grp, bg=CARD)
            hdr.pack(fill="x")
            tk.Label(hdr, text=label, bg=CARD, fg=TXT,
                     font=("Helvetica", 10, "bold")).pack(side="left")
            vl = tk.Label(hdr, text=str(var.get()), bg=CARD, fg=ACCENT,
                          font=("Courier", 10, "bold"))
            vl.pack(side="right")
            self._val_lbls[label] = (vl, var)
            tk.Label(grp, text=hint, bg=CARD, fg=TXT_DIM,
                     font=("Helvetica", 8)).pack(anchor="w")
            tk.Scale(grp, from_=mn, to=mx, resolution=res, variable=var,
                     orient="horizontal", bg=CARD, fg=TXT, troughcolor=BORDER,
                     activebackground=ACCENT, highlightthickness=0,
                     showvalue=False, relief="flat",
                     command=lambda v, lb=label, vr=var:
                         self._on_slide(lb, vr)).pack(fill="x")

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)
        tk.Label(side, text="How it works", bg=CARD, fg=TXT,
                 font=("Georgia", 12, "bold")).pack(padx=16, anchor="w")

        steps = [
            (ACCENT,  "① Gaussian blur reduces noise"),
            (ACCENT2, "② Canny finds all edges"),
            (GREEN,   "③ Largest 4-sided contour\n    = document boundary"),
            (ORANGE,  "④ Harris confirms corners"),
            (RED,     "⑤ Homography warp\n    flattens to rectangle"),
        ]
        for col, txt in steps:
            r = tk.Frame(side, bg=CARD)
            r.pack(fill="x", padx=16, pady=3)
            tk.Label(r, text="▶", bg=CARD, fg=col,
                     font=("Helvetica", 9)).pack(side="left", padx=(0,6), anchor="n")
            tk.Label(r, text=txt, bg=CARD, fg=TXT_DIM,
                     font=("Helvetica", 9), justify="left").pack(side="left", anchor="w")

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)
        tk.Label(side, text="Tip: if detection fails,\ntry lowering High Threshold\nor increasing Blur Kernel.",
                 bg=CARD, fg=TXT_DIM, font=("Helvetica", 9),
                 justify="left").pack(padx=16, anchor="w")

    def _on_slide(self, label, var):
        lbl, _ = self._val_lbls[label]
        lbl.config(text=str(var.get()))
        self._process()

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All", "*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open that file.")
            return
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            s = 1200 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        self.original = img
        self._process()

    def _save(self):
        if self.warped is None:
            messagebox.showwarning("Nothing to save",
                                   "No document detected yet. Load a photo first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.warped)
            messagebox.showinfo("Saved", f"Scan saved:\n{path}")

    def _process(self):
        self._refs.clear()
        src = self.original.copy()

        k = self.blur_k.get()
        if k % 2 == 0: k += 1
        blurred  = cv2.GaussianBlur(src, (k, k), 0)
        gray     = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        lo, hi   = self.thresh_low.get(), self.thresh_high.get()
        if hi <= lo: hi = lo + 1
        edges    = cv2.Canny(gray, lo, hi)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges_d  = cv2.dilate(edges, kernel, iterations=1)

        doc_pts  = find_document_contour(edges_d)

        gray_f   = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
        harris   = cv2.cornerHarris(gray_f, 4, 3, 0.04)
        harris   = cv2.dilate(harris, None)
        corner_vis = src.copy()
        corner_vis[harris > 0.01 * harris.max()] = [80, 220, 80]

        self.warped = None
        detected    = False
        scan_img    = None

        orig_vis = src.copy()

        if doc_pts is not None:
            detected = True
            cv2.polylines(corner_vis, [doc_pts.astype(np.int32)],
                          True, (124, 111, 247), 3)
            for pt in doc_pts:
                cv2.circle(corner_vis, tuple(pt.astype(int)), 10, (78,205,196), -1)
                cv2.circle(corner_vis, tuple(pt.astype(int)), 10, (255,255,255), 2)
            cv2.polylines(orig_vis, [doc_pts.astype(np.int32)],
                          True, (124, 111, 247), 2)
            self.warped = four_point_transform(src, doc_pts)
            scan_img    = self.warped

        ph1 = to_tk(orig_vis)
        self._set("Original", ph1,
                  f"Image: {src.shape[1]}×{src.shape[0]} px"
                  + ("  |  Document outline drawn" if detected else ""))

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        ph2 = to_tk(edges_bgr)
        self._set("Edges (Canny)", ph2,
                  f"Edge pixels: {int(np.sum(edges>0)):,}  |  "
                  f"Low={lo}  High={hi}  Blur={k}×{k}")

        ph3 = to_tk(corner_vis)
        self._set("Corners (Harris)", ph3,
                  "Document corners detected ✓" if detected
                  else "No 4-sided contour found — adjust sliders")

        if scan_img is not None:
            ph4 = to_tk(scan_img)
            self._set("Scanned Output", ph4,
                      f"Warped: {scan_img.shape[1]}×{scan_img.shape[0]} px  "
                      f"— click 'Save Scan' to export")
        else:
            blank = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
            cv2.putText(blank, "No document detected",
                        (55, PANEL_H//2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (90,90,110), 1, cv2.LINE_AA)
            cv2.putText(blank, "Adjust sliders or load a clearer photo",
                        (25, PANEL_H//2+28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (70,70,90), 1, cv2.LINE_AA)
            ph4 = to_tk(blank)
            self._set("Scanned Output", ph4, "Waiting for detection…")

        self._status.set(
            f"Pipeline: Blur {k}×{k}  →  Canny [{lo},{hi}]  →  "
            f"Contour {'FOUND ✓' if detected else 'not found'}  →  "
            f"Warp {'applied ✓' if detected else 'pending'}")

    def _set(self, name, ph, stat):
        img_lbl, stat_lbl = self._panels[name]
        img_lbl.config(image=ph)
        img_lbl.image = ph
        stat_lbl.config(text=stat)
        self._refs.append(ph)


if __name__ == "__main__":
    try:
        import cv2, numpy
        from PIL import Image
    except ImportError as e:
        print(f"\nMissing library: {e}")
        print("Install:  pip install opencv-python numpy Pillow\n")
        exit(1)

    DocumentScanner().mainloop()
