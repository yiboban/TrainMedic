"""Generate Xiaohongshu launch images for TrainMedic.

The script writes editable SVG files and matching PNG exports from the same
page definitions. It does not use network assets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1080
HEIGHT = 1440
SAFE = 86
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
EXPORT_DIR = ROOT / "export"
CONTACT_SHEET = ROOT / "contact-sheet.png"

FONT_FAMILY = '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif'
MONO_FAMILY = '"Consolas", "Cascadia Mono", "Menlo", monospace'

BG = "#07111F"
BG_2 = "#0B1628"
CARD = "#101B2F"
CARD_2 = "#13233B"
BORDER = "#2A3A57"
TEXT = "#F7FAFC"
MUTED = "#B7C3D7"
DIM = "#7D8CA5"
RED = "#FF5A6A"
YELLOW = "#F6C85F"
BLUE = "#5CC8FF"
GREEN = "#56D68A"
PURPLE = "#A78BFA"


@dataclass(frozen=True)
class Element:
    kind: Literal["rect", "text", "line", "polyline", "circle"]
    attrs: dict[str, object]


@dataclass(frozen=True)
class Page:
    filename: str
    elements: tuple[Element, ...]


def rect(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    fill: str = CARD,
    stroke: str = BORDER,
    rx: int = 30,
    width: int = 2,
) -> Element:
    return Element(
        "rect",
        {"x": x, "y": y, "w": w, "h": h, "fill": fill, "stroke": stroke, "rx": rx, "width": width},
    )


def text(
    x: int,
    y: int,
    value: str,
    *,
    size: int = 34,
    fill: str = TEXT,
    weight: str = "400",
    anchor: Literal["start", "middle", "end"] = "start",
    mono: bool = False,
) -> Element:
    return Element(
        "text",
        {
            "x": x,
            "y": y,
            "value": value,
            "size": size,
            "fill": fill,
            "weight": weight,
            "anchor": anchor,
            "mono": mono,
        },
    )


def line(x1: int, y1: int, x2: int, y2: int, *, stroke: str = BORDER, width: int = 3) -> Element:
    return Element(
        "line",
        {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": stroke, "width": width},
    )


def polyline(points: Iterable[tuple[int, int]], *, stroke: str, width: int = 4) -> Element:
    return Element("polyline", {"points": tuple(points), "stroke": stroke, "width": width})


def circle(x: int, y: int, r: int, *, fill: str, stroke: str = "none", width: int = 0) -> Element:
    return Element(
        "circle",
        {"x": x, "y": y, "r": r, "fill": fill, "stroke": stroke, "width": width},
    )


def add_text_lines(
    items: list[Element],
    x: int,
    y: int,
    lines: Iterable[str],
    *,
    size: int,
    fill: str = TEXT,
    weight: str = "400",
    gap: int | None = None,
    mono: bool = False,
    anchor: Literal["start", "middle", "end"] = "start",
) -> None:
    step = gap if gap is not None else int(size * 1.28)
    for index, value in enumerate(lines):
        items.append(
            text(
                x,
                y + index * step,
                value,
                size=size,
                fill=fill,
                weight=weight,
                mono=mono,
                anchor=anchor,
            )
        )


def add_chrome(items: list[Element], page: int, section: str) -> None:
    items.append(rect(0, 0, WIDTH, HEIGHT, fill=BG, stroke=BG, rx=0, width=0))
    for x in range(0, WIDTH + 1, 90):
        items.append(line(x, 0, x, HEIGHT, stroke="#0C1A2D", width=1))
    for y in range(0, HEIGHT + 1, 90):
        items.append(line(0, y, WIDTH, y, stroke="#0C1A2D", width=1))
    items.append(rect(SAFE, 48, 220, 56, fill="#0E2035", stroke="#24425F", rx=28, width=2))
    items.append(circle(SAFE + 34, 76, 14, fill=GREEN))
    items.append(line(SAFE + 26, 76, SAFE + 42, 76, stroke=BG, width=4))
    items.append(line(SAFE + 34, 68, SAFE + 34, 84, stroke=BG, width=4))
    items.append(text(SAFE + 60, 88, "TrainMedic", size=28, fill=TEXT, weight="700"))
    items.append(
        text(WIDTH - SAFE, 88, f"{page:02d}/08", size=28, fill=DIM, weight="700", anchor="end")
    )
    items.append(text(SAFE, HEIGHT - 54, section, size=24, fill=DIM))


def add_heartbeat(items: list[Element], x: int, y: int, w: int) -> None:
    points = []
    for i in range(0, w, 34):
        base = y + int(math.sin(i / 38) * 8)
        points.append((x + i, base))
    spike = x + w // 2
    points.extend(
        [
            (spike - 70, y),
            (spike - 38, y),
            (spike - 18, y - 38),
            (spike + 6, y + 34),
            (spike + 34, y - 18),
            (spike + 62, y),
            (x + w, y),
        ]
    )
    items.append(polyline(points, stroke=GREEN, width=5))


def add_terminal(
    items: list[Element],
    x: int,
    y: int,
    w: int,
    h: int,
    lines: Iterable[tuple[str, str]],
) -> None:
    items.append(rect(x, y, w, h, fill="#08111F", stroke="#23344D", rx=26, width=2))
    items.append(rect(x, y, w, 58, fill="#0F1E33", stroke="#23344D", rx=26, width=0))
    for offset, color in enumerate((RED, YELLOW, GREEN)):
        items.append(circle(x + 36 + offset * 32, y + 29, 9, fill=color))
    cursor_y = y + 104
    for value, color in lines:
        items.append(text(x + 36, cursor_y, value, size=28, fill=color, mono=True))
        cursor_y += 44


def card_with_title(
    items: list[Element],
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: Iterable[str],
    *,
    accent: str = BLUE,
) -> None:
    items.append(rect(x, y, w, h, fill=CARD, stroke=BORDER, rx=32))
    items.append(rect(x, y, 10, h, fill=accent, stroke=accent, rx=5, width=0))
    items.append(text(x + 36, y + 56, title, size=32, fill=TEXT, weight="700"))
    add_text_lines(items, x + 36, y + 104, body, size=25, fill=MUTED, gap=38)


def build_cover() -> Page:
    items: list[Element] = []
    add_chrome(items, 1, "开源 PyTorch 训练诊断工具")
    items.append(rect(790, 138, 150, 48, fill="#1F3A2D", stroke="#2C6B4F", rx=24))
    items.append(text(865, 171, "开源项目", size=26, fill=GREEN, weight="700", anchor="middle"))
    add_text_lines(
        items, SAFE, 250, ["PyTorch", "训不动？"], size=84, fill=TEXT, weight="800", gap=100
    )
    add_text_lines(
        items,
        SAFE,
        500,
        ["我做了个开源工具", "自动帮你找问题"],
        size=48,
        fill=MUTED,
        weight="700",
        gap=66,
    )
    items.append(rect(SAFE, 682, WIDTH - SAFE * 2, 88, fill="#221522", stroke="#62384A", rx=44))
    items.append(
        text(
            WIDTH // 2,
            738,
            "NaN｜grad=None｜参数没更新",
            size=34,
            fill=YELLOW,
            weight="700",
            anchor="middle",
        )
    )
    add_terminal(
        items,
        SAFE,
        842,
        WIDTH - SAFE * 2,
        354,
        [
            ("TM3001 ERROR  Forward output contains NaN", RED),
            ("Object: invalid_log", TEXT),
            ("Evidence: tensor_path=output", BLUE),
            ("Suggestion: check input range", GREEN),
        ],
    )
    add_heartbeat(items, SAFE + 70, 1272, WIDTH - SAFE * 2 - 140)
    return Page("01-cover", tuple(items))


def build_pain_points() -> Page:
    items: list[Element] = []
    add_chrome(items, 2, "真实训练痛点")
    items.append(text(SAFE, 190, "这些问题你遇到过吗？", size=58, fill=TEXT, weight="800"))
    pains = [
        ("loss 突然变成 NaN", RED),
        ("有些参数一直 grad=None", YELLOW),
        ("训练正常运行，但参数根本没更新", BLUE),
        ("optimizer 创建时漏掉了某一层", PURPLE),
        ("验证时忘了 model.eval()", GREEN),
    ]
    y = 310
    for index, (value, color) in enumerate(pains, start=1):
        items.append(rect(SAFE, y, WIDTH - SAFE * 2, 132, fill=CARD, stroke=BORDER, rx=30))
        items.append(circle(SAFE + 58, y + 66, 30, fill=color))
        items.append(
            text(SAFE + 58, y + 77, str(index), size=28, fill=BG, weight="800", anchor="middle")
        )
        items.append(text(SAFE + 112, y + 82, value, size=35, fill=TEXT, weight="700"))
        y += 158
    items.append(rect(SAFE, 1164, WIDTH - SAFE * 2, 116, fill="#17243A", stroke="#314766", rx=32))
    items.append(
        text(
            WIDTH // 2,
            1235,
            "代码不一定报错，但模型就是学不会。",
            size=34,
            fill=YELLOW,
            weight="700",
            anchor="middle",
        )
    )
    return Page("02-pain-points", tuple(items))


def build_why() -> Page:
    items: list[Element] = []
    add_chrome(items, 3, "从混乱到清晰")
    items.append(text(SAFE, 190, "以前只能这样排查", size=56, fill=TEXT, weight="800"))
    left_x = SAFE
    right_x = WIDTH // 2 + 24
    items.append(rect(left_x, 290, 430, 646, fill="#1B1625", stroke="#3C2C4F", rx=36))
    add_text_lines(
        items,
        left_x + 42,
        368,
        ["一层层 print", "手动检查梯度", "反复查看 optimizer", "猜是哪一步出了问题"],
        size=32,
        fill=MUTED,
        gap=86,
    )
    items.append(text(right_x, 340, "所以我做了", size=34, fill=MUTED, weight="700"))
    items.append(text(right_x, 410, "TrainMedic", size=58, fill=GREEN, weight="800"))
    items.append(rect(right_x, 500, 430, 436, fill=CARD, stroke="#2E4E6E", rx=36))
    add_text_lines(
        items,
        right_x + 40,
        582,
        ["诊断编号", "证据字段", "可能原因", "下一步建议"],
        size=34,
        fill=TEXT,
        weight="700",
        gap=76,
    )
    items.append(rect(SAFE, 1048, WIDTH - SAFE * 2, 156, fill="#102635", stroke="#2C5E73", rx=36))
    add_text_lines(
        items,
        WIDTH // 2,
        1110,
        ["让训练问题变成", "有证据的诊断结果。"],
        size=38,
        fill=TEXT,
        weight="800",
        anchor="middle",
        gap=52,
    )
    return Page("03-why-trainmedic", tuple(items))


def build_nan_example() -> Page:
    items: list[Element] = []
    add_chrome(items, 4, "Forward NaN 示例")
    items.append(text(SAFE, 190, "不用再一层层 print", size=56, fill=TEXT, weight="800"))
    add_terminal(
        items,
        SAFE,
        300,
        WIDTH - SAFE * 2,
        252,
        [
            ("with watch_forward(model) as monitor:", TEXT),
            ("    output = model(inputs)", BLUE),
            ("print(format_diagnostics(monitor.diagnostics))", DIM),
        ],
    )
    items.append(rect(SAFE, 626, WIDTH - SAFE * 2, 430, fill=CARD, stroke="#5A2631", rx=34))
    items.append(rect(SAFE + 32, 666, 154, 48, fill="#3A1821", stroke="#6F2B37", rx=24))
    items.append(text(SAFE + 109, 700, "TM3001", size=28, fill=RED, weight="800", anchor="middle"))
    add_text_lines(
        items,
        SAFE + 46,
        772,
        [
            "ERROR",
            "Forward output contains NaN",
            "Object: invalid_log",
            "tensor_path: output",
            "nan_count: 1",
        ],
        size=34,
        fill=TEXT,
        weight="700",
        gap=58,
    )
    items.append(rect(SAFE, 1132, WIDTH - SAFE * 2, 144, fill="#112539", stroke="#2C536D", rx=34))
    add_text_lines(
        items,
        WIDTH // 2,
        1188,
        ["它报告的是首次观察到异常的位置，", "不会武断地说这一定是根因。"],
        size=30,
        fill=MUTED,
        weight="700",
        anchor="middle",
        gap=44,
    )
    return Page("04-nan-example", tuple(items))


def build_features() -> Page:
    items: list[Element] = []
    add_chrome(items, 5, "当前可用能力")
    items.append(text(SAFE, 190, "它现在能检查什么？", size=56, fill=TEXT, weight="800"))
    features = [
        ("01", "optimizer 有没有漏参数", BLUE),
        ("02", "Forward 哪一层首次出现 NaN / Inf", RED),
        ("03", "哪些参数 grad=None 或梯度异常", YELLOW),
        ("04", "optimizer.step 后参数有没有变化", GREEN),
        ("05", "train / eval、Dropout、BatchNorm 是否正确", PURPLE),
    ]
    y = 318
    for number, value, color in features:
        items.append(rect(SAFE, y, WIDTH - SAFE * 2, 126, fill=CARD, stroke=BORDER, rx=30))
        items.append(rect(SAFE + 32, y + 31, 94, 64, fill=color, stroke=color, rx=26, width=0))
        items.append(
            text(SAFE + 79, y + 76, number, size=29, fill=BG, weight="800", anchor="middle")
        )
        items.append(text(SAFE + 156, y + 78, value, size=30, fill=TEXT, weight="700"))
        y += 150
    items.append(
        text(
            WIDTH // 2,
            1218,
            "不是自动修复，而是把问题讲清楚。",
            size=32,
            fill=MUTED,
            weight="700",
            anchor="middle",
        )
    )
    return Page("05-features", tuple(items))


def build_structure() -> Page:
    items: list[Element] = []
    add_chrome(items, 6, "诊断结果结构")
    items.append(text(SAFE, 190, "不只是告诉你“出错了”", size=52, fill=TEXT, weight="800"))
    items.append(rect(SAFE, 300, WIDTH - SAFE * 2, 680, fill=CARD, stroke="#2D4869", rx=38))
    sections = [
        ("问题对象", "Object: decoder.weight", BLUE),
        ("观测证据", "requires_grad: true", GREEN),
        ("可能原因", "optimizer 创建时漏掉模块", YELLOW),
        ("排查建议", "检查 optimizer 参数来源", PURPLE),
    ]
    y = 372
    for title, body, color in sections:
        items.append(
            rect(SAFE + 44, y, WIDTH - SAFE * 2 - 88, 118, fill=CARD_2, stroke=BORDER, rx=28)
        )
        items.append(circle(SAFE + 98, y + 59, 22, fill=color))
        items.append(text(SAFE + 140, y + 50, title, size=28, fill=TEXT, weight="800"))
        items.append(text(SAFE + 140, y + 92, body, size=25, fill=MUTED, mono=":" in body))
        y += 144
    items.append(rect(SAFE, 1066, WIDTH - SAFE * 2, 174, fill="#102635", stroke="#2C536D", rx=36))
    add_text_lines(
        items,
        WIDTH // 2,
        1132,
        ["每个结论都尽量附带可验证证据，", "而不是只给出模糊猜测。"],
        size=32,
        fill=TEXT,
        weight="700",
        anchor="middle",
        gap=48,
    )
    return Page("06-diagnostic-structure", tuple(items))


def build_status() -> Page:
    items: list[Element] = []
    add_chrome(items, 7, "真实项目状态")
    items.append(text(SAFE, 190, "目前是 Alpha 测试版", size=56, fill=TEXT, weight="800"))
    card_with_title(
        items,
        SAFE,
        302,
        WIDTH - SAFE * 2,
        390,
        "已具备",
        [
            "核心诊断功能已经可以运行",
            "Python 3.10 / 3.11 / 3.12 CI",
            "CPU 可用",
            "基础 CUDA Tensor 支持",
            "完全开源",
        ],
        accent=GREEN,
    )
    card_with_title(
        items,
        SAFE,
        738,
        WIDTH - SAFE * 2,
        326,
        "当前限制",
        [
            "还没有发布到 PyPI",
            "API 后续可能调整",
            "尚未正式支持 DDP / FSDP / DeepSpeed",
            "尚未正式支持 Lightning / Transformers Trainer",
        ],
        accent=YELLOW,
    )
    items.append(rect(SAFE, 1132, WIDTH - SAFE * 2, 136, fill="#281E16", stroke="#5F4930", rx=34))
    add_text_lines(
        items,
        WIDTH // 2,
        1186,
        ["适合现在试用和反馈，", "暂时不建议直接作为生产依赖。"],
        size=30,
        fill=YELLOW,
        weight="700",
        anchor="middle",
        gap=44,
    )
    return Page("07-project-status", tuple(items))


def build_cta() -> Page:
    items: list[Element] = []
    add_chrome(items, 8, "邀请真实反馈")
    add_text_lines(
        items,
        SAFE,
        190,
        ["比起 Star", "我更需要真实 Bug"],
        size=62,
        fill=TEXT,
        weight="800",
        gap=78,
    )
    card_with_title(
        items,
        SAFE,
        398,
        WIDTH - SAFE * 2,
        332,
        "想收集这些反馈",
        [
            "你遇到过哪些难排查的 PyTorch 问题？",
            "它有没有误报？",
            "哪些模型或 optimizer 还不能支持？",
        ],
        accent=RED,
    )
    items.append(rect(SAFE, 802, WIDTH - SAFE * 2, 260, fill="#0E2035", stroke="#2B506E", rx=38))
    items.append(
        text(WIDTH // 2, 870, "GitHub 搜索：", size=34, fill=MUTED, weight="700", anchor="middle")
    )
    items.append(
        text(
            WIDTH // 2,
            952,
            "yiboban/TrainMedic",
            size=48,
            fill=GREEN,
            weight="800",
            anchor="middle",
            mono=True,
        )
    )
    items.append(circle(WIDTH // 2 - 220, 1008, 14, fill=BLUE))
    items.append(circle(WIDTH // 2 + 220, 1008, 14, fill=BLUE))
    items.append(line(WIDTH // 2 - 200, 1008, WIDTH // 2 + 200, 1008, stroke=BLUE, width=4))
    items.append(rect(SAFE, 1144, WIDTH - SAFE * 2, 110, fill="#17243A", stroke="#314766", rx=34))
    items.append(
        text(
            WIDTH // 2,
            1212,
            "欢迎试用、提 Issue、提交最小复现",
            size=32,
            fill=TEXT,
            weight="700",
            anchor="middle",
        )
    )
    return Page("08-call-to-action", tuple(items))


def pages() -> tuple[Page, ...]:
    return (
        build_cover(),
        build_pain_points(),
        build_why(),
        build_nan_example(),
        build_features(),
        build_structure(),
        build_status(),
        build_cta(),
    )


def render_svg(page: Page) -> str:
    body = "\n".join(svg_element(element) for element in page.elements)
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    return f"""{svg_open}
  <defs>
    <style>
      text {{
        font-family: {FONT_FAMILY};
        dominant-baseline: alphabetic;
      }}
      .mono {{
        font-family: {MONO_FAMILY};
      }}
    </style>
  </defs>
{body}
</svg>
"""


def svg_element(element: Element) -> str:
    attrs = element.attrs
    if element.kind == "rect":
        return (
            f'  <rect x="{attrs["x"]}" y="{attrs["y"]}" width="{attrs["w"]}" '
            f'height="{attrs["h"]}" rx="{attrs["rx"]}" fill="{attrs["fill"]}" '
            f'stroke="{attrs["stroke"]}" stroke-width="{attrs["width"]}"/>'
        )
    if element.kind == "text":
        css_class = ' class="mono"' if attrs["mono"] else ""
        return (
            f'  <text{css_class} x="{attrs["x"]}" y="{attrs["y"]}" '
            f'font-size="{attrs["size"]}" fill="{attrs["fill"]}" '
            f'font-weight="{attrs["weight"]}" text-anchor="{attrs["anchor"]}">'
            f'{escape(str(attrs["value"]))}</text>'
        )
    if element.kind == "line":
        return (
            f'  <line x1="{attrs["x1"]}" y1="{attrs["y1"]}" '
            f'x2="{attrs["x2"]}" y2="{attrs["y2"]}" stroke="{attrs["stroke"]}" '
            f'stroke-width="{attrs["width"]}" stroke-linecap="round"/>'
        )
    if element.kind == "polyline":
        points = " ".join(f"{x},{y}" for x, y in attrs["points"])
        return (
            f'  <polyline points="{points}" fill="none" stroke="{attrs["stroke"]}" '
            f'stroke-width="{attrs["width"]}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if element.kind == "circle":
        return (
            f'  <circle cx="{attrs["x"]}" cy="{attrs["y"]}" r="{attrs["r"]}" '
            f'fill="{attrs["fill"]}" stroke="{attrs["stroke"]}" stroke-width="{attrs["width"]}"/>'
        )
    raise ValueError(f"Unsupported element kind: {element.kind}")


def render_png(page: Page, output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for element in page.elements:
        draw_element(draw, element)
    image.save(output)


def draw_element(draw: ImageDraw.ImageDraw, element: Element) -> None:
    attrs = element.attrs
    if element.kind == "rect":
        xy = (
            int(attrs["x"]),
            int(attrs["y"]),
            int(attrs["x"]) + int(attrs["w"]),
            int(attrs["y"]) + int(attrs["h"]),
        )
        draw.rounded_rectangle(
            xy,
            radius=int(attrs["rx"]),
            fill=str(attrs["fill"]),
            outline=str(attrs["stroke"]),
            width=int(attrs["width"]),
        )
        return
    if element.kind == "text":
        font = get_font(
            int(attrs["size"]),
            bold=str(attrs["weight"]) in {"700", "800"},
            mono=bool(attrs["mono"]),
        )
        value = str(attrs["value"])
        x = int(attrs["x"])
        y = int(attrs["y"])
        anchor = str(attrs["anchor"])
        if anchor == "middle":
            text_width = draw.textbbox((0, 0), value, font=font)[2]
            x -= text_width // 2
        elif anchor == "end":
            text_width = draw.textbbox((0, 0), value, font=font)[2]
            x -= text_width
        draw.text((x, y - int(attrs["size"])), value, font=font, fill=str(attrs["fill"]))
        return
    if element.kind == "line":
        draw.line(
            (
                int(attrs["x1"]),
                int(attrs["y1"]),
                int(attrs["x2"]),
                int(attrs["y2"]),
            ),
            fill=str(attrs["stroke"]),
            width=int(attrs["width"]),
        )
        return
    if element.kind == "polyline":
        points = [(int(x), int(y)) for x, y in attrs["points"]]
        draw.line(points, fill=str(attrs["stroke"]), width=int(attrs["width"]), joint="curve")
        return
    if element.kind == "circle":
        x = int(attrs["x"])
        y = int(attrs["y"])
        r = int(attrs["r"])
        stroke = str(attrs["stroke"])
        outline = None if stroke == "none" or int(attrs["width"]) == 0 else stroke
        draw.ellipse((x - r, y - r, x + r, y + r), fill=str(attrs["fill"]), outline=outline)
        return
    raise ValueError(f"Unsupported element kind: {element.kind}")


def get_font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates: list[str]
    if mono:
        candidates = [
            "consola.ttf",
            "CascadiaMono.ttf",
            "DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            "msyhbd.ttc",
            "simhei.ttf",
            "PingFang.ttc",
            "NotoSansCJK-Bold.ttc",
            "DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "msyh.ttc",
            "simhei.ttf",
            "PingFang.ttc",
            "NotoSansCJK-Regular.ttc",
            "DejaVuSans.ttf",
        ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_contact_sheet(image_paths: list[Path]) -> None:
    thumb_w = 270
    thumb_h = 360
    gap = 24
    sheet_w = thumb_w * 4 + gap * 5
    sheet_h = thumb_h * 2 + gap * 3
    sheet = Image.new("RGB", (sheet_w, sheet_h), "#0A1220")
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = gap + (index % 4) * (thumb_w + gap)
        y = gap + (index // 4) * (thumb_h + gap)
        sheet.paste(image, (x, y))
    sheet.save(CONTACT_SHEET)


def main() -> int:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for page in pages():
        svg_path = SOURCE_DIR / f"{page.filename}.svg"
        png_path = EXPORT_DIR / f"{page.filename}.png"
        svg_path.write_text(render_svg(page), encoding="utf-8")
        render_png(page, png_path)
        with Image.open(png_path) as image:
            if image.size != (WIDTH, HEIGHT):
                raise RuntimeError(f"{png_path} has wrong size: {image.size}")
        image_paths.append(png_path)
    make_contact_sheet(image_paths)
    print(f"Generated {len(image_paths)} SVG files in {SOURCE_DIR}")
    print(f"Generated {len(image_paths)} PNG files in {EXPORT_DIR}")
    print(f"Generated contact sheet: {CONTACT_SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
