"""Create stand-alone benchmark and annotated replay figures using bundled Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(face, size)
    except OSError:
        return ImageFont.load_default()


def performance_figure(path: Path, native, proposed):
    canvas = Image.new("RGB", (1080, 460), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 36), "32K input on a single RTX 4090", "#17233b", font(32, True))
    draw.text((45, 86), "Qwen2.5-VL-7B · BF16 · 20 cold-cache requests per backend",
              "#49566e", font(18))
    panels = [("Median first-token latency", native[0], proposed[0], "ms"),
              ("Peak occupied KV cache", native[-1], proposed[-1], "GiB")]
    for index, (title, baseline, candidate, unit) in enumerate(panels):
        x = 45 + index * 535
        draw.text((x, 142), title, "#17233b", font(23, True))
        maximum = max(baseline, candidate)
        for row, (label, value, color) in enumerate(
                (("Native", baseline, "#8390a3"), ("Working set", candidate, "#16766d"))):
            y = 200 + row * 80
            draw.text((x, y), label, "#17233b", font(17))
            draw.rounded_rectangle((x, y + 28, x + 330 * value / maximum, y + 49),
                                   radius=5, fill=color)
            display = f"{value:,.0f}" if unit == "ms" else f"{value:.3f}"
            draw.text((x + 350, y + 23), f"{display} {unit}", "#17233b", font(17))
        decline = (1 - candidate / baseline) * 100
        draw.text((x, 377), f"{decline:.1f}% reduction", "#16766d" if decline >= 0 else "#9b4723",
                  font(24, True))
    canvas.save(path)


def replay_figure(path, sample, point):
    with Image.open(sample["image"]) as source:
        screen = source.convert("RGB")
    original = screen.size
    screen.thumbnail((1100, 780), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max(screen.width + 48, 520), screen.height + 138), "#f7f8fb")
    canvas.paste(screen, (24, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 25), "Working-set replay", "#17233b", font(23, True))
    left, top, width, height = sample["bbox"]
    sx, sy = screen.width / original[0], screen.height / original[1]
    draw.rectangle((24 + left * sx, 76 + top * sy, 24 + (left + width) * sx,
                    76 + (top + height) * sy), outline="#25a770", width=4)
    if point is not None:
        x = 24 + point[0] * screen.width / 1000
        y = 76 + point[1] * screen.height / 1000
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#fa6256", outline="white", width=2)
    draw.text((24, screen.height + 94), "Green: labeled target · Red: predicted click",
              "#49566e", font(16))
    canvas.save(path)
