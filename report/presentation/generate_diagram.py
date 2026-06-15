"""Genera PNG y PDF del diagrama SVG del pipeline."""
from pathlib import Path
import cairosvg

out_dir = Path(__file__).parent
svg_path = out_dir / "pipeline_arquitectura_oreilly_beamer.svg"

cairosvg.svg2png(
    url=str(svg_path),
    write_to=str(out_dir / "pipeline_arquitectura_oreilly_beamer.png"),
    output_width=1600,
    output_height=950,
)
cairosvg.svg2pdf(
    url=str(svg_path),
    write_to=str(out_dir / "pipeline_arquitectura_oreilly_beamer.pdf"),
)

print("Archivos generados:")
print(out_dir / "pipeline_arquitectura_oreilly_beamer.png")
print(out_dir / "pipeline_arquitectura_oreilly_beamer.pdf")
