from pathlib import Path
from PIL import Image
import html

base_dir = Path("/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters")

windows = [f"window_{i}" for i in range(1, 8)]

plot_names = [
    "brainplot_Alpha.png",
    "brainplot_DFA.png",
    "brainplot_Exponent.png",
    "brainplot_EI.png",
    "brainplot_SE.png",
]

cols = len(windows)      # 7
rows = len(plot_names)   # 8
pad = 12

# Optional space for labels
top_label_h = 40
left_label_w = 170

# Use first image to determine size
first_file = base_dir / windows[0] / plot_names[0]
with Image.open(first_file) as im:
    cell_w, cell_h = im.size

svg_w = left_label_w + cols * cell_w + (cols - 1) * pad
svg_h = top_label_h + rows * cell_h + (rows - 1) * pad

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
    f'viewBox="0 0 {svg_w} {svg_h}">',
    '<rect width="100%" height="100%" fill="white"/>'
]

# Column labels = windows
for col, window in enumerate(windows):
    x = left_label_w + col * (cell_w + pad) + cell_w / 2
    y = 25
    label_t = "T" + window.split("_")[-1]
    parts.append(
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="Arial" font-size="100" font-weight="bold">{html.escape(label_t)}</text>'
    )

# Row labels = plot types, and images
for row, plot_name in enumerate(plot_names):
    y = top_label_h + row * (cell_h + pad)

    # Row label
    label = plot_name.replace("brainplot_", "").replace(".png", "")
    if label == "Phase_Ex":
        label = "EI"
    elif label == "DFA_EI":
        label= "DFA"
        
    label_y = y + cell_h / 2
    parts.append(
        f'<text x="{left_label_w - 12}" y="{label_y}" text-anchor="end" '
        f'dominant-baseline="middle" font-family="Arial" font-size="100" font-weight="bold">'
        f'{html.escape(label)}</text>'
    )

    # Images across columns (windows)
    for col, window in enumerate(windows):
        file = base_dir / window / plot_name

        if not file.exists():
            print(f"Missing file: {file}")
            continue

        x = left_label_w + col * (cell_w + pad)
        href = html.escape(str(file))

        parts.append(
            f'<image href="{href}" x="{x}" y="{y}" '
            f'width="{cell_w}" height="{cell_h}" />'
        )

parts.append("</svg>")

Path(f"{base_dir}/brainplot_windows_clusters.svg").write_text("\n".join(parts), encoding="utf-8")
print("Saved to brainplot_windows_clusters.svg")