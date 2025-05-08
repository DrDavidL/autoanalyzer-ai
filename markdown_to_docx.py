#!/usr/bin/env python3
import os
import tempfile
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import markdown
from bs4 import BeautifulSoup
from PIL import Image
from docx.oxml.shared import OxmlElement, qn
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
import re
import html2docx


def write_out_html(file_name, text_html, encoding="utf8"):
    """Write HTML content to a file"""
    try:
        with open(file_name, "w", encoding=encoding) as output_fd:
            output_fd.write(text_html)
    except Exception as e:
        print(f"Could not write HTML file {file_name}: {e}")


def do_table_of_contents(document):
    """Add a table of contents to the document"""
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = r'TOC \o "1-3" \h \z \u'
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "separate")
    fld_char3 = OxmlElement("w:t")
    fld_char3.text = "Right-click to update field."
    fld_char2.append(fld_char3)
    fld_char4 = OxmlElement("w:fldChar")
    fld_char4.set(qn("w:fldCharType"), "end")
    r_element = run._r
    r_element.append(fld_char)
    r_element.append(instr_text)
    r_element.append(fld_char2)
    r_element.append(fld_char4)


def add_hyperlink(paragraph, url, text):
    """Add a hyperlink to a paragraph"""
    part = paragraph.part
    r_id = part.relate_to(
        url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True
    )
    hyperlink = docx.oxml.shared.OxmlElement("w:hyperlink")
    hyperlink.set(docx.oxml.shared.qn("r:id"), r_id)
    new_run = docx.oxml.shared.OxmlElement("w:r")
    rPr = docx.oxml.shared.OxmlElement("w:rPr")
    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink


def create_code_block_style(doc):
    """Create a style for code blocks"""
    styles = doc.styles
    style_name = "Code Block"
    if style_name not in styles:
        style = styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        font = style.font
        font.name = "Courier New"
        font.size = Pt(9)
        paragraph_format = style.paragraph_format
        paragraph_format.space_before = Pt(6)
        paragraph_format.space_after = Pt(6)
        paragraph_format.left_indent = Inches(0.5)
        paragraph_format.right_indent = Inches(0.5)
    return style_name


def enhance_html_for_docx(html_content):
    """Enhance HTML content for better DOCX conversion"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Improve code blocks
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        if code:
            # Add a div with special class for code blocks
            code_div = soup.new_tag('div')
            code_div['class'] = 'code-block'
            code_div['style'] = 'background-color: #f5f5f5; padding: 10px; font-family: Courier New; font-size: 9pt; margin: 10px 0;'
            code.wrap(code_div)
    
    # Improve tables
    for table in soup.find_all('table'):
        table['border'] = '1'
        table['style'] = 'border-collapse: collapse; width: 100%;'
        
        # Add thead if not present
        if not table.find('thead') and table.find('tr'):
            first_row = table.find('tr')
            thead = soup.new_tag('thead')
            table.insert(0, thead)
            thead.append(first_row)
            
            # Convert td to th in the header row
            for td in first_row.find_all('td'):
                th = soup.new_tag('th')
                th.string = td.string
                th['style'] = 'background-color: #4472C4; color: white; font-weight: bold; text-align: center; padding: 5px;'
                td.replace_with(th)
        
        # Style all cells
        for td in table.find_all('td'):
            td['style'] = 'padding: 5px; border: 1px solid #DDDDDD;'
            
            # Right-align numeric cells
            if td.string and td.string.strip().replace('.', '', 1).replace('-', '', 1).isdigit():
                td['style'] += ' text-align: right;'
    
    # Improve images
    for img in soup.find_all('img'):
        if 'width' not in img.attrs:
            img['width'] = '80%'
        img['style'] = 'display: block; margin: 10px auto;'
    
    return str(soup)


def markdown_to_docx(project_name, markdown_content):
    """Convert markdown to DOCX using html2docx for better fidelity"""
    try:
        # Create a temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert markdown to HTML with extensions
            # The markdown module has a function called markdown, not a method on the module
            import markdown as md
            html_content = md.markdown(
                markdown_content,
                extensions=[
                    'markdown.extensions.tables',
                    'markdown.extensions.fenced_code',
                    'markdown.extensions.codehilite',
                    'markdown.extensions.toc',
                    'markdown.extensions.nl2br',
                ]
            )
            
            # Enhance HTML for better DOCX conversion
            enhanced_html = enhance_html_for_docx(html_content)
            
            # Write HTML to a temporary file
            html_path = os.path.join(temp_dir, f"{project_name}.html")
            write_out_html(html_path, enhanced_html)
            
            # Convert HTML to DOCX using html2docx
            docx_path = f"{project_name}.docx"
            
            # Create a new document
            doc = docx.Document()
            
            # Create code block style
            code_style = create_code_block_style(doc)
            
            # Convert HTML to DOCX using html2docx
            # html2docx does not have HTML2DOCX class, use the convert() function
            html2docx.convert(enhanced_html, doc)
            
            # Post-process the document to improve formatting
            for paragraph in doc.paragraphs:
                # Fix code blocks
                if paragraph.text.strip().startswith('```') or paragraph.text.strip().endswith('```'):
                    paragraph.style = code_style
                    # Remove the backticks
                    for run in paragraph.runs:
                        run.text = run.text.replace('```', '')
                
                # Fix heading styles
                if paragraph.style.name.startswith('Heading'):
                    # Ensure consistent heading formatting
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor(0, 0, 139)  # Dark blue
            
            # Save the document
            doc.save(docx_path)
            return docx_path
    except Exception as e:
        print(f"Could not convert markdown to docx: {e}")
        try:
            # Fallback to the original implementation if html2docx fails
            return fallback_markdown_to_docx(project_name, markdown_content)
        except Exception as fallback_error:
            print(f"Fallback conversion also failed: {fallback_error}")
            # Return a default path that will be checked for existence
            return f"{project_name}.docx"


def fallback_markdown_to_docx(project_name, markdown_content):
    """Fallback method using the original implementation"""
    try:
        # Create a new document
        doc = docx.Document()
        
        # Convert markdown to HTML
        import markdown as md
        html_content = md.markdown(
            markdown_content,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
            ]
        )
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Create code block style
        code_style = create_code_block_style(doc)
        
        # Process each element
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'table', 'ul', 'ol', 'blockquote', 'hr']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                doc.add_heading(element.get_text(), level)
            elif element.name == 'p':
                # Check if paragraph contains an image
                img = element.find('img')
                if img and 'src' in img.attrs:
                    try:
                        doc.add_picture(img['src'], width=Inches(5))
                    except Exception:
                        doc.add_paragraph(f"[Image could not be loaded: {img['src']}]")
                else:
                    p = doc.add_paragraph()
                    for child in element.children:
                        if child.name == 'strong':
                            p.add_run(child.get_text()).bold = True
                        elif child.name == 'em':
                            p.add_run(child.get_text()).italic = True
                        elif child.name == 'code':
                            run = p.add_run(child.get_text())
                            run.font.name = "Courier New"
                        elif child.name == 'a' and 'href' in child.attrs:
                            add_hyperlink(p, child['href'], child.get_text())
                        else:
                            p.add_run(child.get_text() if hasattr(child, 'get_text') else str(child))
            elif element.name == 'pre':
                code = element.find('code')
                if code:
                    p = doc.add_paragraph(style=code_style)
                    p.add_run(code.get_text())
            elif element.name == 'table':
                rows = element.find_all('tr')
                if rows:
                    table = doc.add_table(rows=len(rows), cols=len(rows[0].find_all(['td', 'th'])))
                    table.style = 'Table Grid'
                    
                    for i, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        for j, cell in enumerate(cells):
                            table.cell(i, j).text = cell.get_text()
                            # Make header row bold
                            if i == 0 or cell.name == 'th':
                                for paragraph in table.cell(i, j).paragraphs:
                                    for run in paragraph.runs:
                                        run.bold = True
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    style = 'List Bullet' if element.name == 'ul' else 'List Number'
                    doc.add_paragraph(li.get_text(), style=style)
            elif element.name == 'blockquote':
                doc.add_paragraph(element.get_text(), style='Intense Quote')
            elif element.name == 'hr':
                doc.add_paragraph('─' * 50)
        
        # Save the document
        docx_path = f"{project_name}.docx"
        doc.save(docx_path)
        return docx_path
    except Exception as e:
        print(f"Fallback conversion failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    project_name = "example_project"
    markdown_content = """
    # Hello, Markdown!

    This is a sample markdown content.

    ## Features
    - Easy to use
    - Converts to Word document

    1. Numbered list
    2. With multiple items

    > This is a blockquote

    ```python
    def hello_world():
        print("Hello, World!")
    ```

    [Link to Google](https://www.google.com)

    ---

    | Column 1 | Column 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    """
    output_file = markdown_to_docx(project_name, markdown_content)
    print(f"Docx file created: {output_file}")
