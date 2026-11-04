from pathlib import Path
from datetime import date
import textwrap

from docx import Document
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent / 'output' / '开题报告成果'
OUT.mkdir(parents=True, exist_ok=True)
FONT_DIR = Path('C:/Windows/Fonts')
FONT_PATH = FONT_DIR / 'msyh.ttc'
if not FONT_PATH.exists():
    FONT_PATH = FONT_DIR / 'simsun.ttc'


def font(size, index=0):
    return ImageFont.truetype(str(FONT_PATH), size=size, index=index)


def rounded(draw, box, fill, outline='#536273', radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_cn(text, max_chars):
    return textwrap.wrap(text, width=max_chars, break_long_words=True, break_on_hyphens=False)


def draw_box(draw, box, title, body='', fill='#F4F7FA', title_color='#183B56', size=26):
    rounded(draw, box, fill)
    x1, y1, x2, y2 = box
    lines = wrap_cn(body, max(8, int((x2-x1-32)/(size+2))))
    draw.text((x1+16, y1+14), title, fill=title_color, font=font(size+2))
    y = y1+54
    for line in lines[:5]:
        draw.text((x1+16, y), line, fill='#263746', font=font(size-3))
        y += size+3


def arrow(draw, start, end, color='#536273', width=4):
    draw.line([start, end], fill=color, width=width)
    import math
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    length = 15
    p1 = (end[0]-length*math.cos(angle-0.5), end[1]-length*math.sin(angle-0.5))
    p2 = (end[0]-length*math.cos(angle+0.5), end[1]-length*math.sin(angle+0.5))
    draw.polygon([end, p1, p2], fill=color)


def architecture_figure(route):
    W, H = (2200 if route == 'A' else 2700), 1540
    img = Image.new('RGB', (W, H), '#FFFFFF')
    d = ImageDraw.Draw(img)
    title = ('路线一：EMCAD + 内容感知抗混叠上采样 + 面积自适应Tversky损失'
             if route == 'A' else
             '路线二：EMCAD + 内容感知抗混叠上采样 + 像素级可靠性多尺度融合')
    d.text((75, 42), title, fill='#142B3B', font=font(38))
    d.text((78, 100), '共享主干：PVTv2-B2 编码器 + EMCAD 渐进式多尺度解码器（示意，非按比例）',
           fill='#536273', font=font(25))

    # Encoder pyramid.
    d.text((90, 190), '分层编码特征', fill='#244B62', font=font(27))
    enc_boxes = [(90, 255, 450, 385), (90, 440, 450, 570),
                 (90, 625, 450, 755), (90, 810, 450, 940)]
    labels = [('x1  浅层', '高分辨率纹理/边界'), ('x2  中浅层', '局部结构表征'),
              ('x3  中深层', '中尺度语义表征'), ('x4  深层', '低分辨率全局语义')]
    for b, (t, body) in zip(enc_boxes, labels):
        draw_box(d, b, t, body, fill='#E8F2F7', title_color='#20566E', size=24)
    draw_box(d, (90, 1010, 450, 1140), '图像输入', 'CT / MRI / 内镜 / 皮肤 / 超声 / 显微', fill='#EFF2F4', size=24)
    d.line([(90, 1070), (55, 1070), (55, 325), (90, 325)], fill='#536273', width=4)
    arrow(d, (90, 325), (102, 325), color='#536273', width=4)
    for y, b in zip([330, 515, 700, 885], enc_boxes):
        if y < 880:
            arrow(d, (270, y), (270, y+110), color='#7C96A5', width=3)

    # Decoder blocks and gated skips.
    d.text((600, 190), 'EMCAD 解码路径', fill='#244B62', font=font(27))
    dec_y = [300, 500, 700]
    for i, y in enumerate(dec_y, 1):
        draw_box(d, (590, y, 1060, y+125), f'EUCB{4-i} 上采样', '内容感知抗混叠块 + 深度卷积 + 点卷积',
                 fill='#FFF2D9', title_color='#82510C', size=22)
        draw_box(d, (1190, y, 1570, y+125), f'LGAG{4-i} 跳连门控', '编码器同尺度特征筛选后相加',
                 fill='#E8F4EC', title_color='#2F6545', size=22)
        if i == 1:
            arrow(d, (450, 880), (590, y+60))
        else:
            arrow(d, (825, y-75), (825, y))
        arrow(d, (1060, y+62), (1190, y+62))
        skip_index = 3-i
        arrow(d, (450, enc_boxes[skip_index][1]+65), (1190, y+85), color='#70977B', width=3)
        if i < 3:
            arrow(d, (1380, y+126), (825, y+195), color='#526B7D', width=3)

    d.text((1640, 190), '多尺度预测与训练 / 推理分流', fill='#244B62', font=font(27))
    pred_boxes = [(1640, 260, 2070, 345), (1640, 385, 2070, 470),
                  (1640, 510, 2070, 595), (1640, 635, 2070, 720)]
    for b, name in zip(pred_boxes, ['p4 深层预测', 'p3 中深层预测', 'p2 中浅层预测', 'p1 高分辨率预测']):
        draw_box(d, b, name, '', fill='#EEF0FB', title_color='#434A84', size=22)
    for i, (y, b) in enumerate(zip([320, 445, 570, 695], pred_boxes)):
        source = (450, 875) if i == 0 else (1570, dec_y[i-1]+62)
        arrow(d, source, (1640, y+40), color='#73789A', width=3)

    if route == 'A':
        draw_box(d, (1640, 810, 2070, 975), '训练阶段', '原多尺度监督\n+  p1 面积加权 Tversky\n仅作用于损失，不增加推理网络层',
                 fill='#FCE8E7', title_color='#8A3D34', size=22)
        arrow(d, (1855, 720), (1855, 810), color='#A85C52')
        draw_box(d, (1640, 1045, 2070, 1170), '推理阶段', '使用 p1 输出预测掩膜',
                 fill='#E8F4EC', title_color='#2F6545', size=24)
        arrow(d, (1855, 975), (1855, 1045), color='#526B7D')
        draw_box(d, (590, 1240, 2070, 1375), '面积自适应权重',
                 '对图像内存在的前景类别，按真值面积平方根倒数构造相对权重并设上限；不改变类别标签，也不参与推理。',
                 fill='#F5F6F7', title_color='#34495A', size=23)
        arrow(d, (1640, 890), (1470, 1240), color='#A85C52', width=3)
    else:
        d.text((2050, 190), '像素级可靠性分支', fill='#244B62', font=font(27))
        reliability_boxes = [(2100, 260, 2510, 345), (2100, 385, 2510, 470),
                             (2100, 510, 2510, 595), (2100, 635, 2510, 720)]
        for i, (y, b, rb) in enumerate(zip([320, 445, 570, 695], pred_boxes, reliability_boxes), 1):
            scale = 5-i
            draw_box(d, rb, f'可靠性头 r{scale}', f'1×1卷积：p{scale} → logit',
                     fill='#F1EAF6', title_color='#614372', size=20)
            arrow(d, (2070, y+40), (2050, y+40), color='#816B91', width=3)
        draw_box(d, (2100, 800, 2510, 940), '像素级权重', '四分数沿尺度维softmax\n每像素权重和为1',
                 fill='#F1EAF6', title_color='#614372', size=22)
        for rb in reliability_boxes:
            arrow(d, ((rb[0]+rb[2])//2, rb[3]), (2305, 800), color='#816B91', width=2)
        draw_box(d, (1640, 840, 1980, 1000), '融合logit', 'Σi wi(x,y) · pi(x,y)\n按任务阈值/argmax解码',
                 fill='#FFF2D9', title_color='#82510C', size=21)
        arrow(d, (2100, 870), (1980, 900), color='#816B91')
        for y, b in zip([300, 425, 550, 675], pred_boxes):
            arrow(d, (1855, y+40), (1855, 840), color='#A3A3B3', width=2)
        draw_box(d, (1640, 1050, 2070, 1235), '僅训练使用真值监督',
                 '融合分割损失 + 逐像素正确性BCE\n正确性目标由各级预测与标签构造\n推理时不读取标签',
                 fill='#FCE8E7', title_color='#8A3D34', size=20)
        arrow(d, (1810, 1000), (1810, 1050), color='#A85C52')
        draw_box(d, (590, 1280, 2460, 1410), '报告边界',
                 '四个p4–p1预测头属于EMCAD监督结构；可靠性融合为项目新增实现。须消融融合模块与新增BCE目标各自贡献。',
                 fill='#F5F6F7', title_color='#34495A', size=22)
    d.text((90, 1460), '注：图示抽象了活动代码的前向与损失关系；具体通道数、分辨率、标签/阈值约定以分支配置和实验配置为准。',
           fill='#62707A', font=font(19))
    path = OUT / f'模型结构图_路线{route}.png'
    img.save(path, dpi=(220, 220))
    return path


FIG_A = architecture_figure('A')
FIG_B = architecture_figure('B')


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def set_cell_margins(cell, top=100, start=110, bottom=100, end=110):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, v in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        node = tcMar.find(qn('w:'+m))
        if node is None:
            node = OxmlElement('w:'+m); tcMar.append(node)
        node.set(qn('w:w'), str(v)); node.set(qn('w:type'), 'dxa')


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)


def set_table_borders(table, color='B9C3CC', size='6'):
    tblPr = table._tbl.tblPr
    borders = tblPr.first_child_found_in('w:tblBorders')
    if borders is None:
        borders = OxmlElement('w:tblBorders'); tblPr.append(borders)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag='w:'+edge
        el=borders.find(qn(tag))
        if el is None: el=OxmlElement(tag); borders.append(el)
        el.set(qn('w:val'),'single'); el.set(qn('w:sz'),size); el.set(qn('w:color'),color)


def run_style(run, name='宋体', size=10.5, bold=False, color='222222', italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    return run


def configure(doc, title):
    sec=doc.sections[0]
    sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(2.15); sec.bottom_margin=Cm(2.1)
    sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.25)
    sec.header_distance=Cm(1.1); sec.footer_distance=Cm(1.0)
    styles=doc.styles
    normal=styles['Normal']; normal.font.name='宋体'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.35; normal.paragraph_format.space_after=Pt(4)
    for sty,sz in [('Title',20),('Heading 1',15),('Heading 2',12),('Heading 3',11)]:
        s=styles[sty]; s.font.name='黑体'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); s.font.size=Pt(sz); s.font.bold=True; s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.keep_with_next=True; s.paragraph_format.space_before=Pt(11 if sty!='Title' else 0); s.paragraph_format.space_after=Pt(5)
    p=sec.footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(title+'  |  '); run_style(r,size=8,color='6A7782')
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); p._p.append(fld)
    return doc


def para(doc,text='',style=None,boldlead=None,align=None,keep=False):
    p=doc.add_paragraph(style=style)
    if align is not None: p.alignment=align
    if boldlead and text.startswith(boldlead):
        run_style(p.add_run(boldlead),bold=True)
        run_style(p.add_run(text[len(boldlead):]))
    else:
        run_style(p.add_run(text))
    if keep: p.paragraph_format.keep_with_next=True
    return p


def heading(doc,text,level=1):
    return doc.add_heading(text,level=level)


def table(doc,headers,rows,widths=None,font_size=8.5):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    if widths:
        for col,w in zip(t.columns,widths): col.width=Cm(w)
    set_table_borders(t)
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=''; set_cell_shading(c,'DCE7EF'); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; set_cell_margins(c)
        p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; run_style(p.add_run(h),'黑体',font_size,True,'193B52')
    set_repeat_table_header(t.rows[0])
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,val in enumerate(row):
            c=cells[i]; c.text=''; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; set_cell_margins(c)
            if widths: c.width=Cm(widths[i])
            if ri%2: set_cell_shading(c,'F5F7F9')
            p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or (isinstance(val,str) and len(val)<16) else WD_ALIGN_PARAGRAPH.LEFT
            run_style(p.add_run(str(val)),size=font_size)
    doc.add_paragraph()
    return t


def figure(doc,path,caption,width=6.25):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path),width=Inches(width))
    c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before=Pt(2); c.paragraph_format.space_after=Pt(7)
    run_style(c.add_run(caption),'宋体',9,color='45545F')


def equation(doc,text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(4); p.paragraph_format.space_after=Pt(7)
    run_style(p.add_run(text),'Cambria Math',11,color='173B52')


def section_label(doc,n,title):
    heading(doc,f'{n}、{title}',1)


def add_page(doc):
    doc.add_page_break()


def build_topics():
    doc=configure(Document(),'硕士学位论文题目备选与排序')
    p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run_style(p.add_run('硕士学位论文题目备选与排序'),'黑体',20,True,'000000')
    para(doc,'排序日期：2026年11月3日。排序优先考虑研究主线统一、与两条已实现路线对应、能形成可检验实验，并避免题目过宽或把尚未证明的性能增益写成结论。')
    titles=[
      ('推荐','面向多尺度医学图像分割的特征重建与预测融合方法研究','用同一主线统摄解码上采样和多尺度预测决策，同时可以纳入训练目标优化；题目不过度绑定某个网络或宣称效果。'),
      ('2','基于高效解码优化的医学图像分割方法研究','覆盖面较适中，但需在正文把优化聚焦在上采样与多尺度监督/融合。'),
      ('3','面向医学图像分割的内容感知特征上采样与多尺度预测优化研究','与两个方案高度对应，略偏技术名词，研究主问题要在开题中解释清楚。'),
      ('4','医学图像分割中多尺度特征重建与像素级预测可靠性建模研究','突出路线二；面积加权损失作为平行训练优化内容，需要更好解释统一性。'),
      ('5','基于内容感知抗混叠重建的医学图像分割方法研究','集中、易落地，但不能自然容纳第二条可靠性融合主线，适合只做一条主线时。'),
      ('6','面向医学图像分割的多尺度解码与类别不平衡优化研究','能纳入面积加权损失和CAA；可靠性融合需作为拓展实验而非并列核心。'),
      ('7','基于多尺度输出可靠性融合的医学图像分割方法研究','聚焦融合，CAA作为配套解码细节；题目相对窄。'),
      ('8','面向复杂形态医学目标的特征重建与多尺度决策研究','强调目标形态，但“复杂”需定义数据与假设。'),
      ('9','基于高效多尺度解码器的医学图像分割算法研究','架构中心明确，但创新点容易被误读为重做整个EMCAD解码器。'),
      ('10','医学图像分割中的内容感知上采样和面积自适应损失研究','精准对应路线一，第二条线无法充分覆盖。'),
      ('11','面向跨模态医学图像的多尺度分割方法研究','覆盖CT/MRI/内镜/皮肤/超声/显微，范围偏大且需真正验证跨模态迁移。'),
      ('12','基于预测可信度建模的医学图像分割研究','“可信度”容易被理解为校准概率/临床可信，当前可靠性图尚未校准验证，现阶段不建议。'),
      ('13','基于不确定性感知的医学图像分割方法研究','与现有像素正确性预测并不等价，容易超出已实现代码，不建议当前使用。'),
      ('14','面向临床辅助诊断的医学图像智能分割研究','“临床辅助诊断”超出本项目验证范围，过宽，不建议。'),
      ('15','基于深度学习的医学图像分割算法研究','过于宽泛，无法体现具体问题，不建议。'),
    ]
    table(doc,['顺序','备选题目','判断依据'],titles,[1.2,6.5,9.0],8.4)
    heading(doc,'建议优先定题',1)
    para(doc,'首选“面向多尺度医学图像分割的特征重建与预测融合方法研究”。该题目把两条路线共同落在解码阶段的多尺度信息恢复与决策整合上；按当前代码，路线一的面积自适应Tversky属于训练目标，不宜误称网络结构。若导师希望题目更聚焦，可采用第3项；若要求以模型方法为主、弱化损失，可采用第2项。最终题目须与导师研究方向和培养领域协商确定。')
    heading(doc,'定题前核实',1)
    for x in ['确认专业学位类别/领域、导师课题来源及学院题名格式。','确认两套分支的实验日志、配置、数据划分和 checkpoint 可一一对应；未核实的效果不写成“已取得提升”。','两路线作为两个独立模型变体分别完成配对消融，不能将两个分支的单次最好值拼成一个“联合模型”。','若未来更改开题题目且研究方向与内容明显变化，按学校管理办法办理重新开题；开题通过后原则上不随意改题。']:
        para(doc,'• '+x)
    path=OUT/'硕士论文题目备选与排序.docx'; doc.save(path); return path


def build_proposal():
    doc=configure(Document(),'专业学位硕士研究生学位论文开题报告')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(78)
    run_style(p.add_run('新 疆 大 学'),'黑体',22,True,'000000')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(26)
    run_style(p.add_run('专业学位研究生学位论文开题报告'),'黑体',19,True,'000000')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(40); p.paragraph_format.space_after=Pt(38)
    run_style(p.add_run('面向多尺度医学图像分割的特征重建与预测融合方法研究'),'黑体',17,True,'173B52')
    info=[('攻读学位层次','☑ 硕士    □ 博士'),('培养方式','□ 全日制    □ 非全日制（请核对）'),('研究生姓名','【待填写】'),('学号','【待填写】'),('所在学院','【待填写】'),('专业学位类别或领域','【待填写，以培养方案为准】'),('学位类别','【待填写】'),('校内指导教师','【待填写】'),('校外指导教师','【无则填写“无”】'),('填表日期','2026年11月【日待填】')]
    t=doc.add_table(rows=0,cols=2); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    for row in info:
        cells=t.add_row().cells; cells[0].width=Cm(5.2); cells[1].width=Cm(9.6)
        for c in cells: set_cell_margins(c,top=130,bottom=130); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cells[0].text=''; run_style(cells[0].paragraphs[0].add_run(row[0]),'黑体',11,True,'34495A')
        cells[1].text=''; run_style(cells[1].paragraphs[0].add_run(row[1]),size=11)
    set_table_borders(t,color='D2D9DE',size='4')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(60)
    run_style(p.add_run('新疆大学研究生院制'),'宋体',11,color='333333')
    add_page(doc)

    heading(doc,'论文题目',1); para(doc,'面向多尺度医学图像分割的特征重建与预测融合方法研究')
    heading(doc,'课题基本信息',1)
    table(doc,['项目','拟填内容'],[
      ('课题类型','□基础研究  ☑应用研究  □综合研究（建议；请按导师意见核实）'),
      ('课题来源','□国家课题 □部省（市）课题 □国际合作 □学校资助 □横向研究 ☑其他/研究生自选（需核实）'),
      ('选题方式','□导师推荐  □研究生自选（按实际勾选；目前不代选）'),
    ],[4.2,11.1],9)
    para(doc,'填写说明：课题来源、选题方式、学位类别等个人/项目事实由研究生和导师核实后勾选，本稿不猜填。以下正文依附件3的栏目顺序组织。')

    section_label(doc,1,'学位论文的选题依据和研究意义，国内外研究现状和发展态势，主要参考文献，以及已有的工作积累和研究成果')
    heading(doc,'1.1 选题依据',2)
    para(doc,'医学图像分割旨在为影像中的器官、组织或病灶赋予像素/体素级语义标签，是定量影像分析、病灶范围测量和后续决策支持的重要基础。深度学习已显著提升自动分割能力，但数据模态、目标尺度和形态差异，以及有限标注和类别不均衡，仍会使边界定位、细小目标召回和不同数据集上的稳定性受到影响[1-4]。本研究讨论的是公开数据上的算法方法，不直接作临床诊断或治疗结论。')
    para(doc,'编码器-解码器及跳跃连接构成医学分割的重要范式。U-Net及其多种扩展通过融合浅层空间细节与深层语义，建立了稳健基线；Transformer及分层视觉骨干则增强长程上下文表征[3,5-10]。EMCAD由Rahman等提出，是一种高效多尺度卷积注意力解码器，包含多尺度深度卷积和通道、空间及分组门控注意力；本课题以其为基础模型，不将其原创模块重新表述为本研究贡献[11-16]。')
    para(doc,'对本项目而言，关注点不是泛称“医学图像分割精度不足”，而是两个相互关联且可检验的技术问题：第一，逐级解码上采样会影响细节与空间结构的恢复；第二，来自不同解码尺度的预测质量可能因位置和尺度而不同，固定地只用一个输出或直接融合并不一定能利用其互补性。另一方面，训练时大面积与小面积前景类别贡献可能不平衡。上述判断是研究动机与待检验假设，不预先宣称某个模块已带来提升。')
    para(doc,'本研究已有两条可运行代码路线：两者都在EMCAD逐级解码上采样路径引入内容感知抗混叠模块；路线A再对最终高分辨率预测使用按真值前景面积构造相对权重的Tversky损失，路线B则为四级分割输出预测像素级可靠性并进行加权logit融合。两条路线分别研究“训练目标对面积不均衡的响应”和“多尺度预测的空间自适应融合”，不把它们预设成同一个联合网络或互相独立的原创定理。')
    heading(doc,'1.2 研究意义',2)
    para(doc,'理论与方法意义在于：把解码特征恢复、前景面积加权的训练目标和多尺度输出融合放在一个可控的分割框架中，采用基线及单模块/双模块对照，分别测量其作用位置、增量参数/计算量和不同数据任务上的表现。若对照实验支持相应假设，可为轻量解码设计和像素级多尺度决策提供可复核的经验；若效果不稳定或无增益，也将明确适用条件，避免仅报告有利数据。')
    para(doc,'应用意义在于探索方法在腹部CT、心脏MRI、内镜、皮肤镜、乳腺超声及细胞显微等公开基准中的适用性。这里的“应用”指算法基准验证，不等同于已达到临床部署、跨医院验证、患者受益或医疗器械要求。研究将遵守数据集许可、隐私和来源要求，并避免把像素级分割指标外推为诊疗效能。')
    heading(doc,'1.3 国内外研究现状和发展趋势',2)
    heading(doc,'1.3.1 编码器-解码器与多尺度医学图像分割',3)
    para(doc,'U-Net通过收缩路径和扩张路径配合跳跃连接，兼顾语义抽象与空间定位[3]；FCN奠定端到端稠密预测基础[4]。UNet++重新设计嵌套跳跃连接，UNet 3+聚合全尺度信息，Attention U-Net在跳连处进行关注区域筛选，nnU-Net则系统化处理预处理、训练配置和推理流程[5-8,17]。这些工作说明模型性能不仅由主干决定，解码和实验协议也会影响结果。')
    para(doc,'TransUNet、Swin-UNet和TransFuse等方法分别探索CNN与Transformer混合、分层窗口注意力及双路特征融合；PVT/PVTv2为密集预测提供分层Transformer表征[9-13]。CASCADE、混合Transformer U-Net及EMCAD在解码和注意力融合方向继续发展[14-16]。当前趋势是在保持多尺度上下文的同时控制复杂度、改善局部边界并提高协议可复现性，而非简单增加模型层数。Transformer在医学影像中的系统性综述亦总结了相关趋势[40]。')
    heading(doc,'1.3.2 上采样、抗混叠和面积不均衡损失',3)
    para(doc,'解码器上采样需要将低分辨率语义逐级映射回高分辨率网格。双线性/最近邻、转置卷积及内容感知特征重组各有平滑、学习能力和开销方面的取舍。关于卷积网络平移稳定性及上采样伪影的工作表明，采样与重建算子需要在目标分辨率下进行分析[22-24]。本项目实现以固定二项式低通卷积形成抗混叠基底，再叠加内容门控的深度卷积残差；它与CARAFE的动态重组核不同，也不能仅凭命名推断其一定提高抗混叠性能。')
    para(doc,'类别不均衡常通过Dice、广义Dice、Focal或Tversky类目标调节漏检与误检的权衡[17-20,25]。本项目路线A在Tversky前景项之外，按每个样本中存在类别的真值面积平方根倒数构造相对权重、归一化并限制最大权重。它是本项目特定的损失实现，需要与标准Dice/交叉熵、固定Tversky以及只加权不加权版本对照；其效应是训练期目标变化，不是推理网络层，也不保证小目标必然改善。')
    heading(doc,'1.3.3 多尺度输出融合与可靠性建模',3)
    para(doc,'深监督使不同解码层获得直接训练信号[21]；分割网络也常用跳连、注意力或多尺度融合协调信息[5-8,14-16]。不确定性估计和校准研究提示，置信分数若要解释为概率可信度，需检验可靠性与真实正确率之间的对应关系[26-27,39]。当前项目可靠性头是对每级logit产生单通道像素分数，经四尺度softmax后加权logit；它以预测正确/错误目标进行训练监督，不是标准多头自注意力，也尚未证明分数是校准的不确定性。')
    para(doc,'由此形成研究空缺假设：在统一EMCAD与训练协议下，检验（i）内容门控残差抗混叠上采样相对原上采样的效果和效率，（ii）按前景面积重加权的Tversky目标对小/大目标类别的影响，（iii）像素级多尺度可靠性融合相对末级输出及不带可靠性监督融合的增益，以及（iv）这些变化能否跨任务稳定复现。检索范围截至2026年11月；正式送审前还应补查2021—2026年该主题最新研究及候选题目的新颖性。')
    heading(doc,'1.4 主要参考文献',2)
    para(doc,'参考文献共40篇，按正文首次引用顺序编号；出版信息需在定稿前逐条以出版社、会议论文集或作者正式页面再次核验，尤其预印本应按最终发表版本更新。')

    heading(doc,'1.5 已有的工作积累和研究成果',2)
    para(doc,'（1）已获得并检查本地EMCAD作者论文PDF及其补充材料，理解原模型的多尺度深度卷积、注意力解码和12数据集/6任务评测设置。论文中的原作者结果仅用于了解基线文献，不作为本研究实验结果。')
    para(doc,'（2）已基于本地两个代码分支完成两种路线的代码实现：分支contentAware_mjzsy_1029包含CAA与面积自适应Tversky训练损失；分支contentAware_xxjkkxdtrh_261101_createFrom_contentAware_mjzsy_1029包含CAA与像素级可靠性融合。其具体实验配置、checkpoint与每次运行日志需在定稿前建立对应清单。')
    para(doc,'（3）已有训练和测试工作积累由研究生后续以可核验日志、配置、随机种子、逐病例预测和指标表补充。当前草稿不填写未经审计的均值、标准差、最好单次值或“优于SOTA”等结论，也不把代码可运行等同于性能有效。')

    section_label(doc,2,'论文研究的主要内容')
    para(doc,'研究目标：在固定EMCAD基线和一致数据协议下，对解码上采样、训练阶段面积加权目标以及多尺度像素级预测融合开展可解释、可消融的实现与评测，确定其适用任务、性能变化和计算代价。')
    heading(doc,'2.1 统一基线与数据协议',2)
    para(doc,'选定与两分支兼容的EMCAD骨干及预训练权重；固定输入通道、分辨率、增强、优化器、训练轮次、学习率、batch、随机种子、损失基项、验证频率和推理后处理。为每个任务保存代码提交号、配置文件、数据清单哈希、预训练权重来源和checkpoint选择规则。采用训练集训练、独立验证集选模、测试集仅作最终评估；Synapse须先建立独立验证病例清单，禁止默认以最终test_vol逐epoch选best。')
    heading(doc,'2.2 路线A：CAA与面积自适应Tversky训练目标',2)
    para(doc,'在三个逐级EUCB上采样环节，以CAA模块替换原上采样基底；在原有深监督/组合监督损失之外，仅训练阶段对p1最高分辨率预测增加面积自适应前景Tversky损失。该损失按存在前景类的标签面积计算相对权重，并设上限。测试阶段仍使用p1完成预测，不额外计算损失。')
    figure(doc,FIG_A,'图1 路线A结构与训练/推理路径（依据 contentAware_mjzsy_1029 分支绘制）',6.2)
    equation(doc,'TPc = Σx pc(x) yc(x)，FPc = Σx pc(x)[1−yc(x)]，FNc = Σx [1−pc(x)]yc(x)')
    equation(doc,'Tc = (TPc + ε) / (TPc + α FPc + β FNc + ε)，Larea = Σc wc(1−Tc) / Σc wc')
    equation(doc,'wc ∝ 1 / √Ac；w̃c = min(wc / meanpresent(w), wmax)')
    para(doc,'其中p为sigmoid/softmax前景概率，y为真值指示图，Ac为相应前景面积；仅存在的前景类参加归一化。分支代码默认α=0.4、β=0.6、最大相对权重3.0、ε=10⁻⁶，具体实验还须从运行配置核实。图中公式表达实际代码的面积加权归约，不把α/β写成由面积动态改变的参数。')
    heading(doc,'2.3 路线B：像素级可靠性多尺度输出融合',2)
    para(doc,'保留p4至p1四个EMCAD输出，并为每一级logit连接1×1卷积可靠性头。将四个可靠性logit在尺度维softmax得到每像素权重，再对四级logit逐像素加权求和。训练包含已有监督、融合结果分割损失以及预测正确性BCE；推理阶段不访问标签。应另比较只有融合分割损失与再加正确性BCE的配置，以分离结构与训练目标贡献。')
    figure(doc,FIG_B,'图2 路线B结构与训练/推理路径（依据当前分支绘制）',6.2)
    equation(doc,'wi(x,y) = softmaxi(ri(x,y))，Σi wi(x,y)=1')
    equation(doc,'zfuse(x,y) = Σi wi(x,y) pi(x,y)')
    para(doc,'多分类任务对最终融合logit按argmax解码；二分类按预先确定且固定的阈值协议生成掩膜。可靠性监督的“正确”定义必须在论文中按代码和评测策略准确报告，并测试其对阈值/后处理变化的敏感性。')
    heading(doc,'2.4 统一实验矩阵与12个数据集',2)
    para(doc,'两条路线分别做独立的2×2因子实验，不把两个分支拼接为一个5因素联合方案。每条路线中：00=原EMCAD；10=仅CAA；01=仅该路线第二因素（路线A为面积损失，路线B为可靠性融合）；11=CAA加第二因素。每个配置使用预先固定的多个随机种子，完全匹配数据、训练预算和checkpoint选择规则。')
    table(doc,['任务/数据集','类别与图像','主要报告指标（按任务协议）'],[
      ('腹部多器官：Synapse','CT，9类含背景','Dice、HD95；按器官及病例报告'),
      ('心脏：ACDC','MRI，4类含背景','Dice、HD95/ASSD；按RV/Myo/LV报告'),
      ('息肉：BKAI、ClinicDB、ColonDB、ETIS、Kvasir','内镜，二值','Dice、IoU、Sensitivity、Precision；表面距离需说明单位'),
      ('皮肤病灶：ISIC2017、ISIC2018','皮肤镜，二值','Dice、IoU及敏感度/特异度；遵循对应官方/公开划分'),
      ('乳腺超声：BUSI','超声病灶，当前实现需核实是否不含normal','Dice、IoU及病例级统计；说明划分不是患者级则不得宣称患者级'),
      ('细胞：DSB18、EM','显微，二值语义分割','Dice、IoU；DSB18实例掩膜合并后不得称实例分割AP'),
    ],[3.6,5.3,6.5],8.1)
    para(doc,'公开基准来源包括Kvasir-SEG[28]、ClinicDB/ColonDB/ETIS相关内镜基准[29,36]、ISIC2017与ISIC2018挑战[30-31]、BUSI[32]、Data Science Bowl细胞数据[33]、ACDC[34]和Synapse腹部多器官挑战[35]；BKAI及息肉比较方法需按当前代码引用的数据版本再补核对[37-39]。12个数据集的结果不能相互替代。五个息肉数据分别训练测试不能宣称跨数据集泛化；若主张泛化，需固定源域训练、目标域仅测试的跨域协议。BUSI、ISIC、Synapse等需明确标签语义和划分来源。')
    heading(doc,'2.5 消融、对比与统计',2)
    para(doc,'至少执行两套分别匹配的00/10/01/11消融。路线A还应比较标准交叉熵+Dice、固定Tversky和面积加权Tversky；路线B应比较末级p1、固定四头等权融合、可学习融合、可靠性BCE监督融合。对照包括原始EMCAD复现与适用的公开强基线；以验证集选模，测试集不得反复调参。建议报告每seed结果、均值±标准差、逐病例/逐器官数据、95%置信区间或合适的配对检验，并校正多重比较。')

    section_label(doc,3,'研究中所要突破的难题或解决的问题')
    for h,b in [
      ('3.1 上采样重建能否改善细节而不过度增算','CAA包含固定低通基底和门控残差。要验证边界/小结构指标是否提高，同时比较额外参数、MACs、延迟和显存，防止把模块名称直接当作效果结论。'),
      ('3.2 面积权重是否改善稀有/小目标且不损害大结构','逆平方根面积权重存在长尾和极小区域高权重风险，需记录权重分布、权重上限敏感性、类别缺失处理，并分别评估小/大器官或病灶。'),
      ('3.3 多尺度可靠性是否真正包含可用的位置条件信息','需测融合权重与各头逐像素正确率的关系，消融BCE监督，评估校准误差及不同阈值下稳定性；不能仅凭BCE下降称“可靠性已校准”。'),
      ('3.4 评测协议和旧结果如何保证独立、可复核','修正验证/测试角色、空掩膜指标、物理间距和二值阈值定义；归档日志与权重。若既有模型选择曾用测试标签，需对外如实说明并依新方案重选/重训。'),
      ('3.5 两条路线的统一性和边界','以共同基线与共同的“多尺度解码—预测”问题串联两项独立方法，禁止把损失函数说成模型层、把两个独立最佳结果合并报告，亦禁止未经文献检索宣称首创。'),
    ]: heading(doc,h,2); para(doc,b)

    section_label(doc,4,'拟采取的主要理论、研究方法、技术路线和实施方案，有何特色与创新之处')
    heading(doc,'4.1 理论基础与研究方法',2)
    para(doc,'以稠密预测的编码器—解码器、多尺度表征、采样与重建、重叠度损失和概率/可靠性监督为基础。采用文献梳理、分支差异审计、控制变量实验、逐病例统计和复杂度测量。算法评估遵循预先设定的训练/验证/测试角色；所有主要结论需在重复seed及适当任务上复核。')
    heading(doc,'4.2 技术路线',2)
    for x in [
      '确认代码基线、两个分支提交、预训练权重许可和配置；锁定数据清单及病人/病例隔离；修复评测与记录问题后保存新版本。',
      '复现原EMCAD对照，分别验证Synapse、ACDC及二值任务入口、类别数、指标和检查点选择，确保基线训练与测试闭环。',
      '实现路线A因子消融：基线、仅CAA、仅面积损失、两者并用；固定训练预算，检查每类别/目标面积分层指标。',
      '实现路线B因子消融：基线、仅CAA、仅可靠性融合、两者并用；进一步分离融合分割项和correctness BCE项。',
      '在12数据集按任务分别报告结果；至少选择多器官CT、心脏MRI、内镜二值任务开展完整消融，其余数据集进行统一主结果与必要消融。跨域结论需单独设计协议。',
      '使用一致环境和统计口径测参数量、完整推理图MACs/FLOPs、显存与延迟；分析失败病例、空掩膜和置信校准。',
      '归档代码版本、逐seed指标、配置、日志、图表和引用来源，撰写论文并根据导师/评审意见修订。',
    ]: para(doc,'• '+x)
    heading(doc,'4.3 特色与拟检验创新点',2)
    para(doc,'（1）在共享EMCAD解码路径研究内容门控与抗混叠基底结合的上采样算子，并将局部残差的作用位置、计算代价与不同尺度边界表现一起评估。该实现属于有待通过对照验证的研究设计，不宣称从未有人使用类似机制。')
    para(doc,'（2）研究一种按图像/前景类别面积构造相对权重的Tversky训练目标，以分析前景面积与错误代价之间的关系；该目标只用于训练，是否改善数据中真实小目标仍需按面积分组实验验证。')
    para(doc,'（3）研究由四级预测自身生成像素级尺度权重的融合机制，结合正确性监督学习不同位置的尺度贡献；通过不带正确性BCE、等权融合和校准分析评估其真实性能边界。')
    para(doc,'本报告把“拟创新点”限定为算法组合与实验证据范围内的候选贡献。最终创新性、专利新颖性/创造性及自由实施空间，需完成系统查新并由导师、学校知识产权部门和代理机构审定。')

    section_label(doc,5,'研究计划可行性，研究条件落实情况，可能存在的问题及解决办法')
    heading(doc,'5.1 可行性',2)
    para(doc,'两条技术路线已有对应的本地代码分支和训练/测试入口，且可按模块开关建立对照；EMCAD作者原文提供公开的多任务评估范式。所需方法基于PyTorch、常见卷积/损失和标准分割评估，工程实现难度可控。硕士周期内可按“先解决协议、再完成两条最小消融、再扩至全部数据集”分阶段推进。已有结果数字目前应以日志、配置和重新核算的指标表补证。')
    heading(doc,'5.2 研究条件',2)
    para(doc,'软件条件：PyTorch及项目现有训练/测试管线；硬件条件：实验室/云服务器GPU、CPU和存储（型号、显存、实际可用时段待填写）；数据条件：公开数据集及其许可/协议文件；人员条件：导师指导和本人承担模型核查、实验、分析与撰写。服务器资源、队列冲突和数据版本须在正式计划中明确。')
    heading(doc,'5.3 风险与对策',2)
    table(doc,['风险','影响','控制与补救'],[
      ('验证集与最终测试集重叠','测试结果参与模型选择，不能作为独立泛化证据','建立训练/验证/测试病例清单；以独立验证集选模；受影响实验重新选择/训练'),
      ('空掩膜指标约定错误','假阳性可能被记成Dice/IoU满分；距离为0','使用预先声明的空类策略；报告失败率/有效数；重算旧逐病例指标'),
      ('多模块效果不稳定或负增益','无法支撑预期创新','保持基线和因子消融；报告负结果；缩小论文核心范围，不以挑seed替代证据'),
      ('12数据集协议差异大','预处理和统计不可直接合并','按任务独立协议、病例级划分与表格；不做未经依据的跨域总体均值'),
      ('计算资源/训练时间不足','实验未完成或重复数不足','优先完成三代表任务全消融和多seed，再扩展；用资源预算表跟踪'),
      ('专利/会议公开影响新颖性','提前公开可能影响可专利性或权利安排','在论文、会议、预印本、答辩公开前咨询学校知识产权部门；核对EMCAD许可与衍生作品条款'),
    ],[3.6,5.0,6.8],8.1)

    section_label(doc,6,'主要方法或设备、仪器及材料等')
    para(doc,'研究方法：深度学习语义分割、损失函数分析、配对消融、分层统计、校准评估和完整推理复杂度测试。')
    para(doc,'主要设备：带CUDA GPU的训练服务器/工作站（GPU型号、显存、CPU、内存、系统盘与数据盘容量待填写）；开发设备为Windows工作站。')
    para(doc,'主要软件：Python、PyTorch、CUDA/cuDNN、NumPy、医学图像IO与评估工具、Git及表格绘图工具。需在实验记录中锁定实际版本，避免“本机能跑”代替目标服务器复现。')
    para(doc,'主要材料：公开数据集图像、标签及许可证/数据使用说明；EMCAD作者代码与许可证文本；训练配置、模型权重、日志、预测掩膜和逐病例结果。不得未经许可上传、再分发数据或移除原作者声明。')

    section_label(doc,7,'实验地点或调研单位或课题（工程）项目名称、来源、总经费、合作单位、完成时间和本人在该项目中承担的任务、地点简介')
    para(doc,'实验地点：【待本人填写实际实验室/课题组及服务器地点】。')
    para(doc,'课题/项目名称与来源：【待导师确认；若为研究生自选且无项目经费，按学院要求填写“无/研究生自选”】。总经费：【待核实，不得推测】。合作单位：【无则填“无”】。计划完成时间：2027年11月（须与培养计划及导师确认）。本人任务：负责文献研究、代码实现核查、数据协议整理、训练/验证/测试、误差分析、结果复核与论文撰写；导师负责研究方向把关和学术指导。')

    section_label(doc,8,'预期的成果')
    for x in [
      '完成经严格协议核验的EMCAD+CAA+面积自适应Tversky路线与EMCAD+CAA+像素级可靠性融合路线对照研究，结论以实际复现实验为准。',
      '形成12个数据集的主结果汇总和按代表任务设计的模块/损失消融表；所有待计算字段先保留空白，不代填数字。',
      '提交硕士学位论文及代码/实验记录归档；可探索投稿ICME或相关会议/期刊，能否按当前截稿时间完成及录用情况不作承诺。',
      '根据查新、导师和学校知识产权审查结果，分别评估两项技术路线是否适合申请专利；是否提交/授权取决于新颖性检索、权属和许可审查。',
    ]: para(doc,'• '+x)
    heading(doc,'表1 代表性数据集实验结果记录框架（数字待补）',2)
    table(doc,['任务/数据集','配置','Seed 1','Seed 2','Seed 3','均值±SD','备注'],[
      (ds,var,'—','—','—','—','病例级统计/配置哈希待登记')
      for ds,var in [('Synapse','00/10/01/11 Dice，HD95'),('ACDC','00/10/01/11 Dice，HD95'),('Kvasir','00/10/01/11 Dice，IoU'),('ISIC2018','00/10/01/11 Dice，IoU'),('BUSI','00/10/01/11 Dice，IoU'),('DSB18','00/10/01/11 Dice，IoU')]
    ],[2.1,3.4,1.4,1.4,1.4,2.3,3.4],7.6)
    heading(doc,'表2 各数据集最终测试集结果框架（逐项复核后填写）',2)
    table(doc,['数据集','N病例/图像','Dice','IoU','HD95/ASSD及单位','Sensitivity','Precision','模型/种子'],[
      (ds,'待补','待补','待补','按任务定义','待补','待补','待补')
      for ds in ['Synapse','ACDC','BKAI','ClinicDB','ColonDB','ETIS','Kvasir','ISIC2017','ISIC2018','BUSI','DSB18','EM']
    ],[2.1,1.7,1.4,1.4,3.1,1.7,1.7,2.5],7.4)
    para(doc,'统计原则：病例/图像均值、跨seed标准差、置信区间、HD95的有效样本数分别报告；不得把像素数当独立样本，不得忽略空预测病例，不得只挑最好seed。上表是框架，不代表已有性能数据。')

    section_label(doc,9,'论文工作阶段计划进度')
    table(doc,['工作项目','内容','起讫时间'],[
      ('科研调查与协议冻结','补充近五年文献；与导师定题；查新；校验12数据集来源、标签与独立划分；确定统计计划','2026.11—2026.12'),
      ('方法审计与基线复核','修正评测/日志问题；固定环境、配置、checkpoint选择；复现EMCAD基线并留档','2026.12—2027.02'),
      ('路线A实验','完成CAA×面积加权损失消融、面积分层及敏感性分析','2027.02—2027.04'),
      ('路线B实验','完成CAA×可靠性融合消融；分离融合分割损失与正确性BCE贡献','2027.04—2027.06'),
      ('数据材料整理与补充实验','扩至12数据集；多seed、跨域（如有）、校准和复杂度复核；错误病例分析','2027.06—2027.08'),
      ('理论分析与论文撰写','整理结果、图表、失败分析；完成学位论文初稿和方法/实验章节','2027.08—2027.10'),
      ('论文评阅与修改','按导师及评阅意见修订；检查引用、数据/代码归档、格式与学术规范','2027.10—2027.11'),
      ('论文答辩','按培养单位正式安排完成预答辩/答辩及材料归档','2027.11起，依学院安排'),
    ],[3.3,8.6,4.0],8.0)
    para(doc,'本进度按硕士论文研究时间不少于一年的原则暂拟，须以开题实际通过日期、培养方案和学院日程调整；若开题延迟，答辩节点相应顺延。')

    section_label(doc,10,'论文提纲')
    outline=[
      '前言：研究背景、意义、研究问题、技术路线和论文结构。',
      '第一章 绪论：医学图像分割、EMCAD基线、多尺度解码/上采样、类别不均衡损失、可靠性融合研究现状。',
      '第二章 相关理论与实验协议：任务定义、数据集及划分、网络基础、损失和评价指标、复杂度统计与统计方案。',
      '第三章 面向面积不均衡的内容感知解码训练方法：CAA实现、面积加权Tversky公式、消融与跨任务评估。',
      '第四章 面向尺度差异的像素级可靠性预测融合方法：四级logit融合、训练监督、校准与消融评估。',
      '第五章 多数据集综合实验与讨论：12数据集主结果、资源开销、失败案例、局限性和复现说明。',
      '结论与展望：有证据支持的结论、适用范围、未来改进。',
      '参考文献、附录：逐seed结果、配置、代码版本和补充预测图。',
    ]
    for x in outline: para(doc,x)

    section_label(doc,11,'导师对学位论文选题和论文计划可行性意见，是否同意开题')
    para(doc,'【由导师填写意见并签字；本稿不代拟、不代签。】')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.RIGHT; run_style(p.add_run('导师（组）签字：________________    年    月    日'),'宋体',10.5)
    heading(doc,'参加研究生论文开题报告的专家',2)
    table(doc,['姓名','职称','工作单位'],[('','','') for _ in range(4)],[4.0,4.0,7.7],9)
    heading(doc,'对开题报告的意见',2)
    para(doc,'开题通过，同意进入论文撰写阶段（      ）')
    para(doc,'开题未通过（      ）')
    para(doc,'【评审组填写具体意见并签字】')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.RIGHT; run_style(p.add_run('组长签名：________________    年    月    日'),'宋体',10.5)
    heading(doc,'学院主管领导意见',2)
    para(doc,'【由学院填写】')
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.RIGHT; run_style(p.add_run('签名：________________    年    月    日'),'宋体',10.5)

    refs=[
    'Litjens G, Kooi T, Bejnordi B E, et al. A survey on deep learning in medical image analysis. Medical Image Analysis, 2017, 42: 60-88. doi:10.1016/j.media.2017.07.005.',
    'Tajbakhsh N, Jeyaseelan L, Li Q, et al. Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation. Medical Image Analysis, 2020, 63: 101693. doi:10.1016/j.media.2020.101693.',
    'Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. MICCAI, 2015: 234-241. doi:10.1007/978-3-319-24574-4_28.',
    'Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation. CVPR, 2015: 3431-3440. doi:10.1109/CVPR.2015.7298965.',
    'Zhou Z, Siddiquee M M R, Tajbakhsh N, Liang J. UNet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE TMI, 2019, 39(6): 1856-1867. doi:10.1109/TMI.2019.2959609.',
    'Huang H, Lin L, Tong R, et al. UNet 3+: A full-scale connected UNet for medical image segmentation. ICASSP, 2020: 1055-1059. doi:10.1109/ICASSP40776.2020.9053405.',
    'Oktay O, Schlemper J, Folgoc L L, et al. Attention U-Net: Learning where to look for the pancreas. arXiv:1804.03999, 2018.',
    'Isensee F, Jaeger P F, Kohl S A A, Petersen J, Maier-Hein K H. nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 2021, 18: 203-211. doi:10.1038/s41592-020-01008-z.',
    'Chen J, Lu Y, Yu Q, et al. TransUNet: Transformers make strong encoders for medical image segmentation. arXiv:2102.04306, 2021.',
    'Cao H, Wang Y, Chen J, et al. Swin-Unet: Unet-like pure transformer for medical image segmentation. ECCV Workshops, 2022: 205-218. doi:10.1007/978-3-031-25066-8_9.',
    'Zhang Y, Liu H, Hu Q. TransFuse: Fusing transformers and CNNs for medical image segmentation. MICCAI, 2021: 14-24. doi:10.1007/978-3-030-87193-2_2.',
    'Wang W, Xie E, Li X, et al. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. ICCV, 2021: 568-578.',
    'Wang W, Xie E, Li X, et al. PVT v2: Improved baselines with pyramid vision transformer. Computational Visual Media, 2022, 8(3): 415-424.',
    'Rahman M M, Marculescu R. EMCAD: Efficient multi-scale convolutional attention decoding for medical image segmentation. arXiv:2405.06880, 2024.',
    'Rahman M M, Marculescu R. Medical image segmentation via cascaded attention decoding. WACV, 2023: 6222-6231.',
    'Wang H, Xie S, Lin L, et al. Mixed transformer U-Net for medical image segmentation. ICASSP, 2022: 2390-2394.',
    'Milletari F, Navab N, Ahmadi S A. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. 3DV, 2016: 565-571. doi:10.1109/3DV.2016.79.',
    'Salehi S S M, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. MLMI, 2017: 379-387.',
    'Abraham N, Khan N M. A novel focal Tversky loss function with improved attention U-Net for lesion segmentation. ISBI, 2019: 683-687. doi:10.1109/ISBI.2019.8759329.',
    'Sudre C H, Li W, Vercauteren T, Ourselin S, Jorge Cardoso M. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. DLMIA, 2017: 240-248.',
    'Lee C Y, Xie S, Gallagher P, Zhang Z, Tu Z. Deeply-supervised nets. AISTATS, 2015: 562-570.',
    'Zhang R. Making convolutional networks shift-invariant again. ICML, 2019: 1254-1263.',
    'Wang J, Chen K, Xu R, Liu Z, Loy C C, Lin D. CARAFE: Content-aware reassembly of features. ICCV, 2019: 3007-3016.',
    'Odena A, Dumoulin V, Olah C. Deconvolution and checkerboard artifacts. Distill, 2016. doi:10.23915/distill.00003.',
    'Kervadec H, Bouget D, Dicecca R, et al. Boundary loss for highly unbalanced segmentation. Medical Image Analysis, 2021, 67: 101851. doi:10.1016/j.media.2020.101851.',
    'Kendall A, Gal Y. What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS, 2017, 30.',
    'Guo C, Pleiss G, Sun Y, Weinberger K Q. On calibration of modern neural networks. ICML, 2017: 1321-1330.',
    'Jha D, Smedsrud P H, Riegler M A, et al. Kvasir-SEG: A segmented polyp dataset. MMM, 2020: 451-462.',
    'Vázquez D, Bernal J, Sánchez F J, et al. A benchmark for endoluminal scene segmentation of colonoscopy images. Journal of Healthcare Engineering, 2017: 1-9.',
    'Codella N, Rotemberg V, Tschandl P, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration. arXiv:1902.03368, 2019.',
    'Codella N C F, Gutman D, Celebi M E, et al. Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging. ISBI, 2018: 168-172.',
    'Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief, 2020, 28: 104863. doi:10.1016/j.dib.2019.104863.',
    'Caicedo J C, Goodman A, Karhohs K W, et al. Nucleus segmentation across imaging experiments: The 2018 Data Science Bowl. Nature Methods, 2019, 16(12): 1247-1253. doi:10.1038/s41592-019-0612-7.',
    'Bernard O, Lalande A, Zotti C, et al. Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved? IEEE TMI, 2018, 37(11): 2514-2525. doi:10.1109/TMI.2018.2837502.',
    'Landman B, Xu Z, Igelsias J E, et al. Multi-atlas labeling beyond the cranial vault: Workshop and challenge. MICCAI Multi-Atlas Labeling Beyond Cranial Vault Workshop, 2015.',
    'Bernal J, Sánchez F J, Fernández-Esparrach G, et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized Medical Imaging and Graphics, 2015, 43: 99-111.',
    'Dong B, Wang W, Fan D P, et al. Polyp-PVT: Polyp segmentation with pyramid vision transformers. arXiv:2108.06932, 2021.',
    'Fan D P, Ji G P, Zhou T, et al. PraNet: Parallel reverse attention network for polyp segmentation. MICCAI, 2020: 263-273.',
    'Kim T, Lee H, Kim D. UACANet: Uncertainty augmented context attention for polyp segmentation. ACM International Conference on Multimedia, 2021: 2167-2175.',
    'Shamshad F, Khan S, Zamir S W, et al. Transformers in medical imaging: A survey. Medical Image Analysis, 2023, 88: 102802.',
    ]
    assert len(refs)==40, len(refs)
    for i,r in enumerate(refs,1):
        p=doc.add_paragraph(); p.paragraph_format.left_indent=Cm(.7); p.paragraph_format.first_line_indent=Cm(-.7); p.paragraph_format.space_after=Pt(2); p.paragraph_format.line_spacing=1.12
        run_style(p.add_run(f'[{i}] {r}'),size=8.7)
    path=OUT/'新疆大学专业学位硕士研究生学位论文开题报告_工作稿.docx'
    doc.save(path); return path


def build_notes():
    doc=configure(Document(),'开题报告提交前的重要核实事项')
    p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run_style(p.add_run('开题报告提交前的重要核实事项'),'黑体',19,True,'000000')
    heading(doc,'最重要的选题判断',1)
    para(doc,'建议先与导师讨论首选题目“面向多尺度医学图像分割的特征重建与预测融合方法研究”。它能统摄两条当前路线，也不把一个特定代码名写死。第二条线中的面积自适应是训练损失，不是网络结构模块；CAA上采样才属于模型结构。两分支是独立方案，不要拼成已经完成的三模块联合系统。')
    heading(doc,'已有运行结果应怎样写',1)
    for x in [
      '以日志、config、提交号、数据清单和best.pth一一对应后，才填“初步结果”；本稿所有数据表留空。此前代码审核发现Synapse默认训练用test_vol逐epoch选best、Synapse/ACDC单边空掩膜指标存在错误；先核对旧实验是否触发，避免把受影响数字写进开题。',
      '现有脚本默认模块开关并不在12个数据集上一致，需逐次检查config中的开关值。用户口头的“两套已跑通”不等于12个数据集都有匹配多seed结果。',
      '文中写12数据集研究计划时要说明按6类任务分别报告；“跨数据集泛化”需要明确源域训练和目标域测试，不能由12个独立数据集的结果直接推出。',
      '二分类后处理有逐图min-max归一化，和固定sigmoid阈值不同；测试指标应按真实代码口径准确披露。HD95/ASSD单位、空掩膜失败率、有效样本数、case/seed标准差也需单独交代。',
    ]: para(doc,'• '+x)
    heading(doc,'专利与ICME投稿的时间先后',1)
    para(doc,'这是本项目必须先向学校知识产权办公室/导师核实的事项。项目本地LICENSE标注UT Austin Research License；文本将该许可扩展到改进/衍生作品，并包含向许可方提供衍生作品、衍生作品继续受相同限制等条款，同时授予许可方特定的非排他许可。两套方法均基于该EMCAD代码修改，因此不能仅凭“代码是我改的”判断专利权属或可实施性。先让学校知识产权部门审阅具体LICENSE版本、受保护代码/方法范围、学生与课题组权属，再决定能否提交申请及需要何种授权。不要在专利顾问审查前向外发送衍生源代码。')
    para(doc,'专利新颖性/创造性通常受申请日前公开内容影响；公开学位论文、会议摘要/全文、预印本、演示材料或可访问代码都可能构成需要评估的公开。若目标是分别申请两件专利，应尽早让专利代理师分别检索并设计互不冲突的权利要求；“两份专利申请”不是保证授权，也不等于可以把同一宽泛构思拆写两遍。建议在任何公开提交前完成权属及专利策略评估。')
    para(doc,'“ICME 12月底截稿”目前仅作为用户提供的计划信息，本稿没有核验其对应年份、官方Full Paper截止日期、时区、页数/模板及投稿轨道。会议投稿、专利申请和论文开题安排可能相互影响；以ICME官方网站当年Call for Papers为准。不要为赶截止日期牺牲独立测试集、复现实验、多seed或审稿所需诚实披露。')
    heading(doc,'个人信息与学院程序',1)
    for x in [
      '封面中的姓名、学号、学院、专业学位类别/领域、校内外导师、培养方式、课题来源和选题方式均须本人/导师补齐。课题经费、实验室和合作单位没有证据的地方不要填写推测。',
      '通知提及原则上第三学期末完成开题、硕士论文工作时间不少于一年；开题通过后原则上不再更改题目。最终计划需按学院系统、导师和秘书老师通知确认。',
      '管理办法要求涉及人群或动物研究同步提交伦理审查。本项目目前描述为公开、去标识化图像算法研究，但仍应核实数据使用协议、伦理/豁免要求和数据集许可；若取得或使用可识别私人临床数据，应及时向机构伦理委员会咨询。',
      '开题通知要求专家提前收到报告、公开线下答辩等流程，请预留时间。模板中的导师意见、专家名单、考核结果、学院意见和签字栏只能由相应人员填写。',
    ]: para(doc,'• '+x)
    heading(doc,'参考文献和图表核验',1)
    para(doc,'报告列出40条文献并在1.1、1.2、1.3采用顺序编码。初稿中的少数会议版本、页码、DOI、挑战集来源在最终提交前还应逐一对出版社原文；若找到正式期刊/会议扩展版本，应引用实际读过的版本。未核实条目应删除或标明预印本，不要仅凭师兄参考文献或模型生成的格式照抄。')
    para(doc,'两张模型图按两个分支代码绘制：图1展示三处CAA及只在训练时使用的面积损失；图2展示四级logits可靠性头、softmax权重及训练标签监督。请根据最终固定提交版本复查层级、箭头、损失路径和术语，正式论文中应保留模型图源文件并标注自绘/代码实现。')
    para(doc,'单独提醒：最终定题、开题申请和专利/会议公开安排应由你和导师共同确认；本文件是完整工作稿，不是导师审核或学院盖章后的正式件。')
    path=OUT/'开题报告提交前的重要核实事项.docx'; doc.save(path); return path


if __name__=='__main__':
    outputs=[build_proposal(),build_topics(),build_notes(),FIG_A,FIG_B]
    for p in outputs: print(p, p.stat().st_size)
