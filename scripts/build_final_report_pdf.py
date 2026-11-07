"""Render FINAL_RESEARCH_REPORT.md and its figures as a formatted PDF."""
from pathlib import Path
import csv, html, re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    PageBreak, Table, TableStyle, Preformatted, Image, KeepTogether, NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/FINAL_RESEARCH_REPORT.pdf'
def pdf_text(s):
    return (s.replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')
        .replace('\u2018',"'").replace('\u2019',"'").replace('\u201c','"').replace('\u201d','"')
        .replace('\u00a0',' ').replace('\u2264','<=').replace('\u2265','>=').replace('\u0394','Delta')
        .replace('\u03c3','sigma').replace('\u03c4','tau').replace('\u00d7','x').replace('\u2026','...'))
def inline(s):
    s=html.escape(pdf_text(s),quote=False)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#1769aa">\1</link>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<font color="#1769aa">\1</font>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
    return s
class ReportDoc(BaseDocTemplate):
    def __init__(self,path,**kw):
        super().__init__(path,**kw)
        frame=Frame(19*mm,17*mm,172*mm,263*mm,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='main',frames=frame,onPage=self.page_marks)])
    def page_marks(self,canvas,doc):
        canvas.saveState();w,h=A4
        canvas.setStrokeColor(colors.HexColor('#d8dee8'));canvas.line(19*mm,14*mm,w-19*mm,14*mm)
        canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#566273'))
        canvas.drawString(19*mm,9*mm,'Final research report | Research Experiment')
        canvas.drawRightString(w-19*mm,9*mm,str(doc.page));canvas.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and getattr(flowable,'_outlineLevel',None) is not None:
            text=flowable.getPlainText();level=flowable._outlineLevel;key=f'h{self.seq.nextf("heading")}'
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level,False)
            self.notify('TOCEntry',(level,text,self.page,key))

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    font='/System/Library/Fonts/Supplemental/Arial.ttf'
    bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    italic='/System/Library/Fonts/Supplemental/Arial Italic.ttf'
    if Path(font).exists():
        pdfmetrics.registerFont(TTFont('Report',font));pdfmetrics.registerFont(TTFont('Report-Bold',bold));pdfmetrics.registerFont(TTFont('Report-Italic',italic))
        pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report-Italic',boldItalic='Report-Bold')
    else:
        pdfmetrics.registerFontFamily('Helvetica',normal='Helvetica',bold='Helvetica-Bold',italic='Helvetica-Oblique',boldItalic='Helvetica-BoldOblique')
    base='Report' if Path(font).exists() else 'Helvetica';code='Courier'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle',parent=styles['Title'],fontName=base+'-Bold' if base=='Report' else 'Helvetica-Bold',fontSize=25,leading=32,textColor=colors.HexColor('#17365d'),alignment=TA_CENTER,spaceAfter=14))
    styles.add(ParagraphStyle(name='CoverSub',parent=styles['Normal'],fontName=base,fontSize=12,leading=18,alignment=TA_CENTER,textColor=colors.HexColor('#536579')))
    styles.add(ParagraphStyle(name='H1x',parent=styles['Heading1'],fontName=base+'-Bold' if base=='Report' else 'Helvetica-Bold',fontSize=15,leading=19,textColor=colors.HexColor('#17365d'),spaceBefore=12,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontName=base+'-Bold' if base=='Report' else 'Helvetica-Bold',fontSize=11,leading=14,textColor=colors.HexColor('#245b89'),spaceBefore=9,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontName=base,fontSize=9,leading=13,spaceAfter=6,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Bulletx',parent=styles['BodyText'],fontName=base,fontSize=8.8,leading=12.4,leftIndent=12,firstLineIndent=-8,spaceAfter=3))
    styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontName=base,fontSize=7.7,leading=10,textColor=colors.HexColor('#586575'),spaceAfter=3))
    styles.add(ParagraphStyle(name='CodeBlockx',fontName=code,fontSize=7.2,leading=9,backColor=colors.HexColor('#f2f5f8'),borderColor=colors.HexColor('#d6dee8'),borderWidth=.4,borderPadding=6,spaceBefore=4,spaceAfter=8))
    story=[]
    story += [Spacer(1,35*mm),Paragraph('FINAL RESEARCH REPORT',styles['CoverTitle']),Spacer(1,7*mm),Paragraph('How Much Accuracy Is Worth the Additional Computing Cost?',styles['CoverSub']),Paragraph('Investigating the Trade-off Between AI Performance and Computational Resources',styles['CoverSub']),Spacer(1,25*mm),Paragraph('Final project audit and results report',styles['CoverSub']),Paragraph('Group 119 | Section C',styles['CoverSub']),Spacer(1,9*mm),Paragraph('Evidence cut-off: 7 November 2026',styles['CoverSub']),PageBreak()]
    story += [Paragraph('Contents',styles['H1x'])]
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle(name='TOC1x',fontName=base,fontSize=8.5,leading=12,leftIndent=0,firstLineIndent=0,spaceBefore=2),ParagraphStyle(name='TOC2x',fontName=base,fontSize=7.5,leading=10,leftIndent=12,firstLineIndent=0)]
    story += [toc,PageBreak()]
    lines=(ROOT/'FINAL_RESEARCH_REPORT.md').read_text(encoding='utf-8').splitlines()
    i=0;in_code=False;code_lines=[];paragraph=[]
    def flush_para():
        nonlocal paragraph
        if paragraph:
            story.append(Paragraph(inline(' '.join(x.strip() for x in paragraph)),styles['Bodyx']));paragraph=[]
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            flush_para()
            if not in_code:in_code=True;code_lines=[]
            else:story.append(Preformatted(pdf_text('\n'.join(code_lines)),styles['CodeBlockx']));in_code=False
            i+=1;continue
        if in_code:code_lines.append(line);i+=1;continue
        if line.startswith('|'):
            flush_para();block=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?',c.replace(' ','')) for c in row):block.append(row)
                i+=1
            if block:
                width=max(len(r) for r in block);data=[]
                for ri,row in enumerate(block):
                    row=row+['']*(width-len(row));data.append([Paragraph(inline(c),ParagraphStyle(name=f'T{ri}_{j}',fontName=(base+'-Bold' if base=='Report' else 'Helvetica-Bold') if ri==0 else base,fontSize=6.7,leading=8.3,textColor=colors.white if ri==0 else colors.HexColor('#202a35'))) for j,c in enumerate(row)])
                avail=172*mm;colw=avail/width
                t=Table(data,colWidths=[colw]*width,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#244c70')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')]),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#c9d2dd')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story += [Spacer(1,3),t,Spacer(1,7)]
            continue
        if not line.strip():flush_para();i+=1;continue
        if line.startswith('# '):flush_para();i+=1;continue
        if line.startswith('## '):
            flush_para();p=Paragraph(inline(line[3:]),styles['H1x']);p._outlineLevel=0;story.append(p);i+=1;continue
        if line.startswith('### '):
            flush_para();p=Paragraph(inline(line[4:]),styles['H2x']);p._outlineLevel=1;story.append(p);i+=1;continue
        if line.startswith('- '):
            flush_para();story.append(Paragraph('&bull; '+inline(line[2:]),styles['Bulletx']));i+=1;continue
        if re.match(r'^\d+\. ',line):
            flush_para();m=re.match(r'^(\d+)\. (.*)',line);story.append(Paragraph(f"{m.group(1)}. {inline(m.group(2))}",styles['Bulletx']));i+=1;continue
        paragraph.append(line);i+=1
    flush_para()
    cat=ROOT/'results/processed/figure_catalogue.csv'
    if cat.exists():
        figrows=list(csv.DictReader(cat.open(encoding='utf-8-sig')))
        story += [PageBreak(),Paragraph('Figure appendix',styles['H1x'])]
        for item in figrows:
            story += [Paragraph(f"Figure {item['number']}. {html.escape(item['title'])}",styles['H2x'])]
            p=ROOT/item['file_png']
            if p.exists():
                im=Image(str(p));im._restrictSize(170*mm,112*mm);story.append(im)
            story += [Paragraph('<b>Data:</b> '+html.escape(item['data_source']),styles['Smallx']),Paragraph('<b>Interpretation:</b> '+html.escape(item['interpretation']),styles['Smallx']),Paragraph('<b>Limitations:</b> '+html.escape(item['limitations']),styles['Smallx']),Spacer(1,6)]
            if int(item['number'])%1==0:story.append(PageBreak())
    doc=ReportDoc(str(OUT),pagesize=A4,title='Final Research Report',author='Research Project Group 119',leftMargin=19*mm,rightMargin=19*mm,topMargin=17*mm,bottomMargin=19*mm)
    doc.multiBuild(story)
    print(f'Wrote {OUT} ({OUT.stat().st_size} bytes)')
if __name__=='__main__':main()
