#!/usr/bin/env python3
"""Build report.pdf from report.md with embedded Unicode fonts (Cyrillic-safe)."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, KeepTogether, Table, TableStyle
import glob

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'report.md'
PDF = ROOT / 'report.pdf'


def find_font(filename: str, fallback: str | None = None) -> Path:
    candidates = [
        Path('/usr/share/fonts/truetype/dejavu') / filename,
        Path('/usr/local/share/fonts') / filename,
        Path('/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf') / filename,
        Path('/usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf') / filename,
        Path('/usr/local/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf') / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = glob.glob(f'/usr/**/{filename}', recursive=True) + glob.glob(f'/usr/local/**/{filename}', recursive=True)
    for match in matches:
        if Path(match).is_file():
            return Path(match)
    if fallback:
        return find_font(fallback)
    # Matplotlib is present in most Colab images, but import it only as a final fallback.
    try:
        from matplotlib import font_manager
        match = Path(font_manager.findfont('DejaVu Sans'))
        if match.exists():
            return match
    except Exception:
        pass
    raise FileNotFoundError(f'Unicode font {filename} not found; install fonts-dejavu-core before building PDF.')


regular_font = find_font('DejaVuSans.ttf')
bold_font = find_font('DejaVuSans-Bold.ttf', 'DejaVuSans.ttf')
oblique_font = find_font('DejaVuSans-Oblique.ttf', 'DejaVuSans.ttf')
mono_font = find_font('DejaVuSansMono.ttf', 'DejaVuSans.ttf')
pdfmetrics.registerFont(TTFont('DV', str(regular_font)))
pdfmetrics.registerFont(TTFont('DV-Bold', str(bold_font)))
pdfmetrics.registerFont(TTFont('DV-Oblique', str(oblique_font)))
pdfmetrics.registerFont(TTFont('DVM', str(mono_font)))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-Bold', italic='DV-Oblique', boldItalic='DV-Bold')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyRU', fontName='DV', fontSize=8.4, leading=11.2, spaceAfter=3.5, textColor=colors.HexColor('#263244'), alignment=TA_LEFT))
styles.add(ParagraphStyle(name='TitleRU', fontName='DV-Bold', fontSize=18, leading=21.5, spaceAfter=9, textColor=colors.HexColor('#17324D'), alignment=TA_LEFT, keepWithNext=True))
styles.add(ParagraphStyle(name='H2RU', fontName='DV-Bold', fontSize=11.6, leading=13.7, spaceBefore=6, spaceAfter=3.2, textColor=colors.HexColor('#176B70'), keepWithNext=True))
styles.add(ParagraphStyle(name='H3RU', fontName='DV-Bold', fontSize=9.5, leading=11.2, spaceBefore=5, spaceAfter=2.5, textColor=colors.HexColor('#176B70'), keepWithNext=True))
styles.add(ParagraphStyle(name='BulletRU', parent=styles['BodyRU'], leftIndent=13, firstLineIndent=-8, spaceAfter=2.5))


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="DVM" size="8">\1</font>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor('#D8E1E8'))
    canvas.setLineWidth(0.5)
    canvas.line(20*mm, 15*mm, w-20*mm, 15*mm)
    canvas.setFont('DV', 7.5)
    canvas.setFillColor(colors.HexColor('#65758B'))
    canvas.drawString(20*mm, 10*mm, 'Лабораторная работа № 6 · AI-детектив')
    canvas.drawRightString(w-20*mm, 10*mm, f'{doc.page}')
    canvas.restoreState()

story = []
lines = MD.read_text(encoding='utf-8').splitlines()
paragraph = []
bullets = []
table_rows = []

def flush_paragraph():
    global paragraph
    if paragraph:
        txt = ' '.join(x.strip() for x in paragraph)
        story.append(Paragraph(inline(txt), styles['BodyRU']))
        paragraph = []

def flush_table():
    global table_rows
    if table_rows:
        cell = ParagraphStyle(name='Cell', parent=styles['BodyRU'], fontSize=7.6, leading=9.6, spaceAfter=0)
        head = ParagraphStyle(name='CellH', parent=cell, fontName='DV-Bold')
        data = [[Paragraph(inline(c), head if i == 0 else cell) for c in row] for i, row in enumerate(table_rows)]
        t = Table(data, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#C5D1DC')),
                               ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F1F4')),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
        story.append(t); story.append(Spacer(1, 5))
        table_rows = []

def flush_bullets():
    global bullets
    if bullets:
        items = [ListItem(Paragraph(inline(x), styles['BodyRU']), leftIndent=10) for x in bullets]
        story.append(ListFlowable(items, bulletType='bullet', start='circle', leftIndent=15, bulletFontName='DV', bulletFontSize=7, spaceAfter=5))
        bullets = []

for raw in lines:
    line = raw.strip()
    if line.startswith('|'):
        flush_paragraph(); flush_bullets()
        cells = [c.strip() for c in line.strip('|').split('|')]
        if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            table_rows.append(cells)
        continue
    flush_table()
    if not line:
        flush_paragraph(); flush_bullets(); continue
    if line.startswith('# '):
        flush_paragraph(); flush_bullets()
        story.append(Paragraph(inline(line[2:]), styles['TitleRU']))
    elif line.startswith('## '):
        flush_paragraph(); flush_bullets()
        story.append(Paragraph(inline(line[3:]), styles['H2RU']))
    elif line.startswith('### '):
        flush_paragraph(); flush_bullets()
        story.append(Paragraph(inline(line[4:]), styles['H3RU']))
    elif line.startswith('- '):
        flush_paragraph(); bullets.append(line[2:])
    elif re.match(r'^\d+\.\s+', line):
        flush_bullets()
        # Keep numbered paragraphs readable without losing the source numbering.
        flush_paragraph()
        m = re.match(r'^(\d+)\.\s+(.*)$', line)
        story.append(Paragraph(f'<b>{m.group(1)}.</b> {inline(m.group(2))}', styles['BodyRU']))
    elif line.startswith('```'):
        flush_paragraph(); flush_bullets()
    else:
        paragraph.append(line)
flush_paragraph(); flush_bullets(); flush_table()

doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=18*mm, bottomMargin=21*mm, title='Отчёт по лабораторной работе № 6 «AI-детектив»', author='AI-детектив project')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f'Wrote {PDF} ({PDF.stat().st_size} bytes)')
