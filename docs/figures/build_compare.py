"""Ghép gsasrec.drawio (trên) và model_v2.drawio (dưới) thành compare_gsasrec_v2.drawio,
khoanh tím các phần v2 thêm vào so với gSASRec.

    python docs/figures/build_gsasrec.py && python docs/figures/build_model_v2.py && python docs/figures/build_compare.py
    "%LOCALAPPDATA%/Programs/draw.io/draw.io.exe" -x -f png -s 3 -b 24 -o docs/figures/compare_gsasrec_v2.png docs/figures/compare_gsasrec_v2.drawio
"""

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "compare_gsasrec_v2.drawio"
TOP_Y = 60        # chừa chỗ cho tiêu đề (a)
BOTTOM_Y = 760    # gốc toạ độ của hình (b)
PURPLE = "#9673a6"
FONT = "fontFamily=Helvetica;html=1;whiteSpace=wrap;"
TITLE = "text;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=24;fontStyle=1;fontColor=#1a1a1a;" + FONT
NOTE = "text;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=17;fontColor=" + PURPLE + ";" + FONT
MARK = f"rounded=1;arcSize=10;fillColor=none;strokeColor={PURPLE};strokeWidth=3;dashed=1;dashPattern=10 6;" + FONT


def cells_of(path: Path, prefix: str, dy: float) -> list[ET.Element]:
    root = ET.parse(path).getroot().find(".//root")
    out = []
    for c in root.findall("mxCell"):
        if c.get("id") in ("0", "1"):
            continue
        c.set("id", prefix + c.get("id"))
        for k in ("source", "target"):
            if c.get(k):
                c.set(k, prefix + c.get(k))
        g = c.find("mxGeometry")
        if g is not None:
            if g.get("y") is not None:
                g.set("y", str(float(g.get("y")) + dy))
            for p in g.iter("mxPoint"):
                p.set("y", str(float(p.get("y")) + dy))
        out.append(c)
    return out


extra: list[ET.Element] = []


def vertex(cid: str, label: str, x: float, y: float, w: float, h: float, style: str) -> None:
    c = ET.Element("mxCell", id=cid, value=label, style=style, vertex="1", parent="1")
    ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    extra.append(c)


vertex("t_a", "(a) gSASRec", 30, 0, 600, 40, TITLE)
vertex("t_b", "(b) Đề xuất v2 = gSASRec + nội dung", 30, BOTTOM_Y - 80, 700, 40, TITLE)
vertex("lg_box", "", 760, BOTTOM_Y - 70, 46, 22, MARK)
vertex("lg_txt", "phần thêm so với (a)", 816, BOTTOM_Y - 80, 260, 40, NOTE)
# vùng khác biệt trên hình (b) — toạ độ theo build_model_v2.py
vertex("m_content", "", 26, BOTTOM_Y + 204, 470, 150, MARK)    # đặc trưng nội dung + Linear
vertex("m_gate", "", 548, BOTTOM_Y + 88, 324, 104, MARK)       # cổng hợp nhất
vertex("m_warm", "", 578, BOTTOM_Y - 10, 244, 76, MARK)        # khởi tạo ấm từ gSASRec

body = cells_of(HERE / "gsasrec.drawio", "a_", TOP_Y) + cells_of(HERE / "model_v2.drawio", "b_", BOTTOM_Y)

mxfile = ET.Element("mxfile", host="drawio", version="24.0.0")
diagram = ET.SubElement(mxfile, "diagram", id="compare", name="gSASRec vs v2")
model = ET.SubElement(diagram, "mxGraphModel", dx="1700", dy="1400", grid="1", gridSize="10", guides="1", tooltips="1",
                      connect="1", arrows="1", fold="1", page="0", pageScale="1", pageWidth="1750", pageHeight="1450",
                      math="1", shadow="0", background="#ffffff")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")
for c in body + extra:   # vùng khoanh vẽ sau cùng để nằm trên
    root.append(c)
ET.ElementTree(mxfile).write(OUT, encoding="utf-8", xml_declaration=False)
print(OUT)
