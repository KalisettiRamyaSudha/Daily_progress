"""
W3D2 - Save Charts to File
Saves generated charts as image files inside the project folder,
demonstrating different formats and output settings.
"""

import matplotlib.pyplot as plt
import os

# Keep chart output organized in its own subfolder within the project
output_dir = "charts"
os.makedirs(output_dir, exist_ok=True)

categories = ["A", "B", "C", "D"]
values = [23, 45, 12, 38]

plt.figure(figsize=(7, 5))
plt.bar(categories, values, color="#8172b2")
plt.title("Sample Chart for Saving")
plt.xlabel("Category")
plt.ylabel("Value")
plt.tight_layout()

# Save as PNG (most common, good for viewing/sharing)
png_path = os.path.join(output_dir, "sample_chart_W3D2.png")
plt.savefig(png_path, dpi=150)
print(f"Saved {png_path}")

# Save as PDF (good for reports/printing, vector format)
pdf_path = os.path.join(output_dir, "sample_chart_W3D2.pdf")
plt.savefig(pdf_path)
print(f"Saved {pdf_path}")

# Save as SVG (good for further editing, scalable vector format)
svg_path = os.path.join(output_dir, "sample_chart_W3D2.svg")
plt.savefig(svg_path)
print(f"Saved {svg_path}")

plt.close()
print(f"\nAll charts saved inside the '{output_dir}/' folder.")