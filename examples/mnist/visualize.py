"""Interactive MNIST digit recognizer.

Draw a digit with the mouse on the left canvas; the trained network predicts
it in real time and displays the per-class probabilities as bars on the right.

Run:
    python examples/mnist/train.py          # train + save mnist_model.npz
    python examples/mnist/visualize.py      # launch the GUI

Requires only the standard library + numpy (Tkinter ships with Python).
"""

import os
import sys
import tkinter as tk
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import Load, Tensor


# ---------------------------------------------------------------------------
# Geometry — canvas is upscaled so drawing feels smooth, then downsampled
# to the 28x28 input the model expects.
# ---------------------------------------------------------------------------

GRID = 28
SCALE = 12                       # canvas pixel = SCALE x model pixel
CANVAS = GRID * SCALE            # 336 px square

# Wide brush with true Gaussian falloff — matches MNIST's anti-aliased strokes.
BRUSH_R = 34                     # painting radius for the Gaussian (buffer)
BRUSH_SIGMA = BRUSH_R * 0.45     # std-dev; smaller = tighter core, larger = softer
BRUSH_CORE_R = 17                # visible solid-white stroke half-width on canvas

# Training subtracted the dataset mean from every input — we must too.
MNIST_MEAN = 0.1307

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "mnist_model.npz")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, model) -> None:
        self.root = root
        self.model = model

        # Backing buffer at canvas resolution — 1.0 = ink, 0.0 = blank.
        self.buffer = np.zeros((CANVAS, CANVAS), dtype=np.float32)

        # Previous mouse position, used to interpolate fast drags.
        self.last_xy: tuple[int, int] | None = None

        self._BuildUI()
        self._UpdateBars(np.zeros(10))

    # -- UI ----------------------------------------------------------------

    def _BuildUI(self) -> None:
        self.root.title("MNIST Digit Recognizer")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        # Left column — canvas + clear button
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, padx=14, pady=14)

        self.canvas = tk.Canvas(
            left, width=CANVAS, height=CANVAS,
            bg="black", highlightthickness=1, highlightbackground="#444",
            cursor="pencil",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._OnPress)
        self.canvas.bind("<B1-Motion>", self._OnDrag)
        self.canvas.bind("<ButtonRelease-1>", self._OnRelease)

        tk.Button(
            left, text="Clear", command=self._Clear,
            bg="#3498db", fg="white", font=("Arial", 12, "bold"),
            relief="flat", width=12, pady=6,
        ).pack(pady=(10, 0))

        tk.Label(
            left, text="Draw a digit (0–9)", bg="#1e1e1e", fg="#aaa",
            font=("Arial", 10),
        ).pack(pady=(8, 0))

        # Right column — prediction + bars
        right = tk.Frame(self.root, bg="#1e1e1e")
        right.grid(row=0, column=1, padx=(0, 18), pady=14, sticky="n")

        tk.Label(
            right, text="Prediction", bg="#1e1e1e", fg="#aaa",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w")

        self.pred_label = tk.Label(
            right, text="–", bg="#1e1e1e", fg="#2ecc71",
            font=("Arial", 80, "bold"), width=2,
        )
        self.pred_label.pack(pady=(0, 6))

        self.conf_label = tk.Label(
            right, text="confidence: 0.0%", bg="#1e1e1e", fg="#aaa",
            font=("Arial", 10),
        )
        self.conf_label.pack(pady=(0, 12))

        # Per-class probability bars
        self.bars: list[tuple[tk.Canvas, tk.Label]] = []
        for digit in range(10):
            row = tk.Frame(right, bg="#1e1e1e")
            row.pack(fill="x", pady=1)

            tk.Label(
                row, text=str(digit), bg="#1e1e1e", fg="white",
                font=("Consolas", 11, "bold"), width=2,
            ).pack(side="left")

            bar = tk.Canvas(
                row, width=220, height=14, bg="#2a2a2a",
                highlightthickness=0,
            )
            bar.pack(side="left", padx=(4, 6))

            value = tk.Label(
                row, text="  0.0%", bg="#1e1e1e", fg="#888",
                font=("Consolas", 9), width=7, anchor="w",
            )
            value.pack(side="left")

            self.bars.append((bar, value))

    # -- Drawing -----------------------------------------------------------

    def _StampBuffer(self, x: int, y: int) -> None:
        """Apply a Gaussian brush stamp into the model-input buffer."""
        x0, x1 = max(0, x - BRUSH_R), min(CANVAS, x + BRUSH_R + 1)
        y0, y1 = max(0, y - BRUSH_R), min(CANVAS, y + BRUSH_R + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        intensity = np.exp(-dist_sq / (2.0 * BRUSH_SIGMA * BRUSH_SIGMA)).astype(np.float32)
        region = self.buffer[y0:y1, x0:x1]
        np.maximum(region, intensity, out=region)

    def _StrokeTo(self, x: int, y: int) -> None:
        """Paint a connected stroke from `last_xy` to (x, y)."""
        if self.last_xy is None:
            # First point of a stroke — a single circular cap.
            self.canvas.create_oval(
                x - BRUSH_CORE_R, y - BRUSH_CORE_R,
                x + BRUSH_CORE_R, y + BRUSH_CORE_R,
                fill="white", outline="",
            )
            self._StampBuffer(x, y)
        else:
            x0, y0 = self.last_xy
            # Visual: a single thick rounded-cap line — Tk renders this smoothly
            # with no inter-stamp "beads" no matter how fast the user drags.
            self.canvas.create_line(
                x0, y0, x, y,
                fill="white", width=BRUSH_CORE_R * 2,
                capstyle="round",
            )
            # Buffer: stamp Gaussian densely along the segment so falloffs blend.
            dx, dy = x - x0, y - y0
            dist = float(np.hypot(dx, dy))
            step_px = max(BRUSH_SIGMA / 2.0, 1.0)
            steps = max(int(dist / step_px), 1)
            for i in range(1, steps + 1):
                t = i / steps
                self._StampBuffer(int(x0 + t * dx), int(y0 + t * dy))
        self.last_xy = (x, y)

    def _OnPress(self, event: tk.Event) -> None:
        self.last_xy = None
        self._StrokeTo(event.x, event.y)
        self._Predict()

    def _OnDrag(self, event: tk.Event) -> None:
        self._StrokeTo(event.x, event.y)
        self._Predict()

    def _OnRelease(self, event: tk.Event) -> None:
        self.last_xy = None
        self._Predict()

    def _Clear(self) -> None:
        self.canvas.delete("all")
        self.buffer[:] = 0.0
        self.last_xy = None
        self.pred_label.config(text="–", fg="#666")
        self.conf_label.config(text="confidence: 0.0%")
        self._UpdateBars(np.zeros(10))

    # -- Inference ---------------------------------------------------------

    def _Predict(self) -> None:
        # Downsample CANVAS x CANVAS -> 28x28 by SCALE x SCALE block averaging.
        small = self.buffer.reshape(GRID, SCALE, GRID, SCALE).mean(axis=(1, 3))

        if small.sum() < 1e-3:               # blank — don't run the model
            self._UpdateBars(np.zeros(10))
            self.pred_label.config(text="-", fg="#666")
            self.conf_label.config(text="confidence: 0.0%")
            return

        # Mean-center to match training preprocessing.
        centered = (small - MNIST_MEAN).astype(np.float32)
        x = Tensor(centered.reshape(784, 1))
        probs = self.model(x).Data.flatten()

        best = int(np.argmax(probs))
        self.pred_label.config(text=str(best), fg="#2ecc71")
        self.conf_label.config(text=f"confidence: {probs[best] * 100:.1f}%")
        self._UpdateBars(probs)

    def _UpdateBars(self, probs: np.ndarray) -> None:
        best = int(np.argmax(probs)) if probs.max() > 0 else -1
        for i, (bar, value) in enumerate(self.bars):
            bar.delete("all")
            width = int(220 * probs[i])
            if width > 0:
                color = "#2ecc71" if i == best else "#3498db"
                bar.create_rectangle(0, 0, width, 14, fill=color, outline="")
            value.config(
                text=f"{probs[i] * 100:5.1f}%",
                fg="#eee" if i == best else "#888",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"No saved model at {MODEL_PATH}.\n"
            f"Train one first:  python {os.path.relpath(os.path.join(HERE, 'train.py'))}"
        )

    print(f"Loading model from {os.path.basename(MODEL_PATH)}...")
    model = Load(MODEL_PATH)
    print("Model loaded. Opening window — draw a digit with the mouse.")

    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
