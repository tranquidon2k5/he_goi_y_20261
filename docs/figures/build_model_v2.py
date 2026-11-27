"""Sinh docs/figures/model_v2.drawio (kiến trúc mô hình đề xuất v2) rồi xuất PNG / SVG bằng draw.io desktop.

    python docs/figures/build_model_v2.py            # chỉ sinh .drawio
    "%LOCALAPPDATA%/Programs/draw.io/draw.io.exe" -x -f png -s 4 -b 24 -o docs/figures/model_v2.png docs/figures/model_v2.drawio
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).with_name("model_v2.drawio")
FONT = "fontFamily=Helvetica;fontSize=18;fontColor=#1a1a1a;html=1;whiteSpace=wrap;"
GREEN = "rounded=1;arcSize=14;fillColor=#d5e8d4;strokeColor=#82b366;" + FONT
BLUE = "rounded=1;arcSize=14;fillColor=#dae8fc;strokeColor=#6c8ebf;" + FONT
RED = "rounded=1;arcSize=14;fillColor=#f8cecc;strokeColor=#b85450;" + FONT
ORANGE = "rounded=0;fillColor=#ffe6cc;strokeColor=#d79b00;" + FONT
WHITE_RED = "rounded=0;fillColor=#ffffff;strokeColor=#b85450;" + FONT
STACK = "rounded=0;fillColor=#f5f5f5;strokeColor=#4d4d4d;" + FONT
STACK_W = "rounded=0;fillColor=#ffffff;strokeColor=#1a1a1a;" + FONT
NOTE = "rounded=1;arcSize=14;fillColor=#f5f5f5;strokeColor=#808080;dashed=1;" + FONT.replace("fontSize=18", "fontSize=16")
TEXT = "text;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;" + FONT
ZONE = ("rounded=1;arcSize=4;dashed=1;dashPattern=8 6;fillColor=#fdf3f3;strokeColor=#c0504d;strokeWidth=2;"
        "html=1;whiteSpace=wrap;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;"
        "endSize=8;fontFamily=Helvetica;fontSize=16;fontColor=#1a1a1a;labelBackgroundColor=#ffffff;")
E_GREEN = EDGE + "strokeColor=#82b366;strokeWidth=3;"
E_BLUE = EDGE + "strokeColor=#6c8ebf;strokeWidth=3;"
E_RED = EDGE + "strokeColor=#ea6b66;strokeWidth=3;"
E_BLACK = EDGE + "strokeColor=#333333;strokeWidth=2;"
E_GREY = EDGE + "strokeColor=#808080;strokeWidth=2;dashed=1;"

cells: list[str] = []
_id = [10]


def nid() -> str:
    _id[0] += 1
    return f"c{_id[0]}"


def box(label: str, x: float, y: float, w: float, h: float, style: str, cid: str | None = None) -> str:
    cid = cid or nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                  f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def stack(label: str, x: float, y: float, w: float, h: float, style: str = STACK) -> str:
    """Ba tờ xếp chồng như hình mẫu; trả id tờ trên cùng."""
    box("", x + 10, y + 10, w, h, style)
    box("", x + 5, y + 5, w, h, style)
    return box(label, x, y, w, h, style)


def edge(src: str, dst: str | None, style: str, label: str = "", points: list[tuple[float, float]] | None = None,
         extra: str = "", target_xy: tuple[float, float] | None = None, label_x: float | None = None) -> None:
    """dst=None: mũi tên trỏ vào điểm target_xy. label_x ∈ [-1, 1]: vị trí nhãn dọc theo mũi tên."""
    pts = ""
    if points:
        pts = "<Array as=\"points\">" + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
    if target_xy:
        pts += f'<mxPoint x="{target_xy[0]}" y="{target_xy[1]}" as="targetPoint"/>'
    tgt = f' target="{dst}"' if dst else ""
    lx = f' x="{label_x}"' if label_x is not None else ""
    cells.append(f'<mxCell id="{nid()}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}{extra}" edge="1" '
                 f'parent="1" source="{src}"{tgt}><mxGeometry{lx} relative="1" as="geometry">{pts}</mxGeometry></mxCell>')


# ------------------------------------------------------------------ vùng "chỉ khi huấn luyện" (vẽ trước để nằm dưới)
box("", 330, 362, 1350, 268, ZONE)
box("<i>Chỉ khi huấn luyện</i>", 1500, 594, 170, 30, TEXT.replace("fontColor=#1a1a1a", "fontColor=#c0504d")
    .replace("fontSize=18", "fontSize=16") + "align=right;")

# ------------------------------------------------------------------ hàng 1: luồng chính
seq = stack("Chuỗi phim đã thích<br>\\(i_1, \\dots, i_t\\)<br><font style=\"font-size:15px\">\\(t \\le 100\\)</font>",
            40, 86, 200, 108)
emb = box("Embedding ID", 300, 105, 180, 70, GREEN)
gate = box("Cổng hợp nhất<br>\\(r_{i_k} = g \\odot e_{i_k} + (1-g) \\odot c_{i_k}\\)", 560, 98, 300, 84, BLUE)
trf = box("Vị trí + Transformer ×2<br><font style=\"font-size:15px\">nhân quả · pre-norm</font>", 910, 105, 200, 70, GREEN)
out = stack("\\(s_j = h_t^{\\top} r_j\\)<br><font style=\"font-size:15px\">\\(r_j\\) từ \\(e_j\\) và \\(f_j\\) · mọi phim \\(j\\)</font>",
            1190, 86, 190, 108, STACK_W)
topk = box("Top-k phim<br><font style=\"font-size:15px\">bỏ phim đã xem</font>", 1405, 0, 190, 50, TEXT)

edge(seq, emb, E_GREEN)
edge(emb, gate, E_GREEN, "\\(e_{i_k}\\)")
edge(gate, trf, E_GREEN, "\\(r_{i_k}\\)")
edge(trf, out, E_GREEN, "\\(h_t\\)")
edge(out, topk, E_BLACK, points=[(1500, 140)], extra="exitX=1;exitY=0.5;entryX=0.5;entryY=1;")

# ------------------------------------------------------------------ khởi tạo ấm
warm = box("gSASRec đã huấn luyện<br><font style=\"font-size:14px\">khởi tạo ấm · cổng \\(g \\approx 1\\)</font>",
           590, 0, 220, 56, NOTE)
edge(warm, emb, E_GREY, points=[(390, 28)], extra="exitX=0;exitY=0.5;entryX=0.5;entryY=0;")
edge(warm, trf, E_GREY, points=[(1010, 28)], extra="exitX=1;exitY=0.5;entryX=0.5;entryY=0;")

# ------------------------------------------------------------------ hàng 2: nhánh nội dung
feat = stack("Đặc trưng nội dung<br>\\(f_{i_1}, \\dots, f_{i_t}\\)<br>"
             "<font style=\"font-size:14px\">mỗi \\(f \\in \\mathbb{R}^{326}\\)<br>thể loại · thập niên · tag</font>",
             40, 216, 200, 118)
lin = box("Linear<br>\\(326 \\to 64\\)", 300, 241, 180, 70, BLUE)
edge(feat, lin, E_BLUE)
edge(lin, gate, E_BLUE, "\\(c_{i_k}\\)", points=[(650, 276)], extra="exitX=1;exitY=0.5;entryX=0.3;entryY=1;")
# biểu diễn có nội dung cũng dùng để chấm phim đích và phim âm khi huấn luyện
# (vẽ trước mũi tên "dùng chung" để nhãn của mũi tên kia nằm đè lên đoạn trùng nhau)
edge(gate, None, E_BLUE + "dashed=1;",
     "\\(r_{i_{t+1}},\\ r_{i^-}\\)<br><font style=\"font-size:14px\">cùng Linear + cổng</font>",
     points=[(760, 326), (1180, 326)], extra="exitX=0.667;exitY=1;", target_xy=(1180, 440), label_x=0.86)
edge(gate, out, E_BLUE, "\\(r_j\\) dùng chung", points=[(760, 326), (1220, 326)],
     extra="exitX=0.667;exitY=1;entryX=0.16;entryY=1;")

# ------------------------------------------------------------------ hàng 3: huấn luyện
tgt = stack("Phim kế tiếp \\(i_{t+1}\\)<br><font style=\"font-size:15px\">mục tiêu dương<br>"
            "ID \\(e_{i_{t+1}}\\) + nội dung \\(f_{i_{t+1}}\\)</font>", 40, 440, 200, 108)
neg = box("Lấy mẫu âm<br><font style=\"font-size:15px\">26 phim \\(i^-\\) / vị trí<br>"
          "ID \\(e_{i^-}\\) + nội dung \\(f_{i^-}\\)</font>", 640, 378, 220, 90, RED)
loss_f = box("\\(-\\beta \\log \\sigma(s^{+}) - \\sum_{k=1}^{26} \\log \\sigma(-s^{-}_{k})\\)<br>"
             "<font style=\"font-size:15px\">\\(s^{+} = h_t^{\\top} r_{i_{t+1}},\\ s^{-} = h_t^{\\top} r_{i^-}\\)"
             "<br>\\(\\beta = \\alpha\\,(t(1-\\alpha^{-1}) + \\alpha^{-1}),\\ t = 0.18\\)</font>",
             1110, 440, 350, 110, WHITE_RED)
loss = box("\\(\\mathcal{L}_{\\text{gBCE}}\\)", 1510, 460, 150, 70, ORANGE)
box("\\(\\lambda_{\\text{CL}} = 0\\)<br><font style=\"font-size:14px\">tune tự tắt học tương phản</font>",
    1485, 538, 200, 50, TEXT.replace("fontSize=18", "fontSize=16"))

edge(tgt, loss_f, E_RED, "\\(i_{t+1}\\)", extra="entryX=0;entryY=0.49;")
edge(neg, loss_f, E_RED, "\\(i^-\\)", extra="entryX=0;entryY=0.2;")
edge(out, loss_f, E_BLACK, "\\(h_t\\)", extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
edge(loss_f, loss, E_BLACK)

xml = (
    '<mxfile host="drawio" version="24.0.0"><diagram id="v2" name="Mô hình v2">'
    '<mxGraphModel dx="1640" dy="640" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
    'fold="1" page="0" pageScale="1" pageWidth="1700" pageHeight="700" math="1" shadow="0" background="#ffffff">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
)
OUT.write_text(xml, encoding="utf-8")
print(OUT)
