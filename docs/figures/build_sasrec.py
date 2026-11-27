"""Sinh docs/figures/sasrec.drawio (kiến trúc SASRec gốc — ablation 1, cấu hình đã tune trong dự án).

    python docs/figures/build_sasrec.py
    "%LOCALAPPDATA%/Programs/draw.io/draw.io.exe" -x -f png -s 4 -b 24 -o docs/figures/sasrec.png docs/figures/sasrec.drawio
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).with_name("sasrec.drawio")
FONT = "fontFamily=Helvetica;fontSize=18;fontColor=#1a1a1a;html=1;whiteSpace=wrap;"
GREEN = "rounded=1;arcSize=14;fillColor=#d5e8d4;strokeColor=#82b366;" + FONT
RED = "rounded=1;arcSize=14;fillColor=#f8cecc;strokeColor=#b85450;" + FONT
ORANGE = "rounded=0;fillColor=#ffe6cc;strokeColor=#d79b00;" + FONT
WHITE_RED = "rounded=0;fillColor=#ffffff;strokeColor=#b85450;" + FONT
STACK = "rounded=0;fillColor=#f5f5f5;strokeColor=#4d4d4d;" + FONT
STACK_W = "rounded=0;fillColor=#ffffff;strokeColor=#1a1a1a;" + FONT
TEXT = "text;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;" + FONT
ZONE = ("rounded=1;arcSize=4;dashed=1;dashPattern=8 6;fillColor=#fdf3f3;strokeColor=#c0504d;strokeWidth=2;"
        "html=1;whiteSpace=wrap;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;"
        "endSize=8;fontFamily=Helvetica;fontSize=16;fontColor=#1a1a1a;labelBackgroundColor=#ffffff;")
E_GREEN = EDGE + "strokeColor=#82b366;strokeWidth=3;"
E_GREEN_D = EDGE + "strokeColor=#82b366;strokeWidth=3;dashed=1;"
E_RED = EDGE + "strokeColor=#ea6b66;strokeWidth=3;"
E_BLACK = EDGE + "strokeColor=#333333;strokeWidth=2;"

cells: list[str] = []
_id = [10]


def nid() -> str:
    _id[0] += 1
    return f"c{_id[0]}"


def q(s: str) -> str:
    return escape(s, {'"': "&quot;"})


def box(label, x, y, w, h, style, cid=None):
    cid = cid or nid()
    cells.append(f'<mxCell id="{cid}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def stack(label, x, y, w, h, style=STACK):
    box("", x + 10, y + 10, w, h, style)
    box("", x + 5, y + 5, w, h, style)
    return box(label, x, y, w, h, style)


def edge(src, dst, style, label="", points=None, extra=""):
    pts = ""
    if points:
        pts = '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
    cells.append(f'<mxCell id="{nid()}" value="{q(label)}" style="{style}{extra}" edge="1" parent="1" '
                 f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')


SMALL = '<font style="font-size:15px">{}</font>'

# vùng huấn luyện (vẽ trước)
box("", 300, 300, 1290, 250, ZONE)
box("<i>Chỉ khi huấn luyện</i>", 1410, 514, 170, 30,
    TEXT.replace("fontColor=#1a1a1a", "fontColor=#c0504d").replace("fontSize=18", "fontSize=16") + "align=right;")

# hàng 1: luồng chính
seq = stack("Chuỗi phim đã thích<br>\\(i_1, \\dots, i_t\\)<br>" + SMALL.format("\\(t \\le 100\\)"), 40, 66, 180, 108)
emb = box("Embedding ID<br>" + SMALL.format("\\(e_i \\in \\mathbb{R}^{64}\\)"), 300, 85, 190, 70, GREEN)
pos = box("+ Vị trí<br>" + SMALL.format("\\(p_1, \\dots, p_{100}\\)"), 570, 85, 170, 70, GREEN)
trf = box("Transformer ×2<br>" + SMALL.format("self-attention nhân quả · FFN"), 820, 85, 230, 70, GREEN)
out = stack("\\(s_j = h_t^{\\top} e_j\\)<br>" + SMALL.format("với mọi phim \\(j\\)"), 1130, 66, 190, 108, STACK_W)
topk = box("Top-k phim<br>" + SMALL.format("bỏ phim đã xem"), 1345, 0, 190, 50, TEXT)

edge(seq, emb, E_GREEN)
edge(emb, pos, E_GREEN, "\\(e_i\\)")
edge(pos, trf, E_GREEN)
edge(trf, out, E_GREEN, "\\(h_t\\)")
edge(out, topk, E_BLACK, points=[(1440, 120)], extra="exitX=1;exitY=0.5;entryX=0.5;entryY=1;")
# embedding đầu ra dùng chung với đầu vào
edge(emb, out, E_GREEN_D, "\\(e_j\\) dùng chung (đầu vào = đầu ra)", points=[(395, 245), (1160, 245)],
     extra="exitX=0.5;exitY=1;entryX=0.16;entryY=1;")

# hàng 3: huấn luyện
tgt = stack(r"Phim kế tiếp \(i_{t+1}\)<br>" + SMALL.format("mục tiêu dương"), 40, 384, 180, 100)
neg = box("Lấy mẫu âm<br>" + SMALL.format("1 phim / vị trí · đều"), 560, 322, 250, 70, RED)
loss_f = box(r"\(-\log \sigma(s^{+}) - \log \sigma(-s^{-})\)<br>"
             + SMALL.format(r"\(s^{\pm} = h_t^{\top} e_{i^{\pm}}\)"),
             1050, 384, 350, 84, WHITE_RED)
loss = box(r"\(\mathcal{L}_{\text{BCE}}\)", 1440, 391, 130, 70, ORANGE)
box("chỉ 1 mẫu âm: tín hiệu yếu,<br>dễ quá tự tin với phim phổ biến", 1035, 482, 380, 50,
    TEXT.replace("fontSize=18", "fontSize=15").replace("fontColor=#1a1a1a", "fontColor=#555555"))

edge(tgt, loss_f, E_RED, r"\(s^{+}\)", extra="entryX=0;entryY=0.5;")
edge(neg, loss_f, E_RED, r"\(s^{-}\)", extra="entryX=0;entryY=0.2;")
edge(out, loss_f, E_BLACK, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
edge(loss_f, loss, E_BLACK)

xml = ('<mxfile host="drawio" version="24.0.0"><diagram id="sasrec" name="SASRec">'
       '<mxGraphModel dx="1600" dy="560" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
       'fold="1" page="0" pageScale="1" pageWidth="1650" pageHeight="600" math="1" shadow="0" background="#ffffff">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
OUT.write_text(xml, encoding="utf-8")
print(OUT)
