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


def generate_gpt_analysis_docx(
    file_name,
    question,
    research_summary,
    code,
    output,
    image_paths=None,
    categorical_mappings=None,
):
    """
    Generate a DOCX file for GPT analysis, with sections for question, summary, code, output, images, and categorical mappings.
    """
    doc = docx.Document()

    # Title
    doc.add_heading("GPT Analysis Report", 0)

    # Question
    doc.add_heading("Original Question", level=1)
    doc.add_paragraph(question)

    # Helper: add HTML content to a docx paragraph (supports bold, italic, lists, etc)
    def add_html_to_doc(doc, html):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        def walk(element, parent=None):
            if element.name == "ul":
                for li in element.find_all("li", recursive=False):
                    p = doc.add_paragraph(style="List Bullet")
                    walk(li, p)
            elif element.name == "ol":
                for li in element.find_all("li", recursive=False):
                    p = doc.add_paragraph(style="List Number")
                    walk(li, p)
            elif element.name in ["p", "li"]:
                p = parent if parent else doc.add_paragraph()
                for child in element.children:
                    walk(child, p)
            elif element.name in ["strong", "b"]:
                run = parent.add_run(element.get_text())
                run.bold = True
            elif element.name in ["em", "i"]:
                run = parent.add_run(element.get_text())
                run.italic = True
            elif element.name == "code":
                run = parent.add_run(element.get_text())
                run.font.name = "Courier New"
            elif element.name == "br":
                parent.add_run("\n")
            elif element.name is None:
                # Plain text node
                parent.add_run(str(element))
            # Add more tags as needed (a, blockquote, etc)
        for elem in soup.contents:
            walk(elem)

    # Research Summary
    if research_summary:
        doc.add_heading("Research Summary", level=1)
        # Convert markdown to HTML, then add to docx
        html = markdown.markdown(research_summary)
        add_html_to_doc(doc, html)

    # Code
    if code:
        doc.add_heading("Code used for analysis", level=1)
        code_block_style = create_code_block_style(doc)
        for line in code.strip().split("\n"):
            p = doc.add_paragraph(line, style=code_block_style)

    # Helper: parse markdown-style bold/italic in a string and add to a paragraph
    def add_markdown_text(paragraph, text):
        """Adds markdown-formatted text to a paragraph, handling bold, italics, and other elements."""
        try:
            html = markdown.markdown(text, extensions=['nl2br', 'fenced_code', 'tables'])
            soup = BeautifulSoup(html, 'html.parser')

            def walk(element, parent):
                for child in element.contents:
                    if isinstance(child, str):
                        parent.add_run(child)
                    elif child.name == 'strong' or child.name == 'b':
                        run = parent.add_run(child.get_text())
                        run.bold = True
                    elif child.name == 'em' or child.name == 'i':
                        run = parent.add_run(child.get_text())
                        run.italic = True
                    elif child.name == 'a':
                        add_hyperlink(parent, child['href'], child.get_text())
                    elif child.name == 'code':
                        run = parent.add_run(child.get_text())
                        run.font.name = "Courier New"
                        run.font.size = Pt(10)
                    elif child.name == 'ul':
                        for li in child.find_all('li'):
                            p = paragraph.insert_paragraph_before("• " + li.get_text(), style='List Bullet')
                    elif child.name == 'ol':
                        for i, li in enumerate(child.find_all('li')):
                            p = paragraph.insert_paragraph_before(f"{i+1}. " + li.get_text(), style='List Number')
                    elif child.name == 'table':
                        # Handle tables by creating a docx table
                        table = docx.Document()
                        docx_table = table.add_table(rows=0, cols=len(child.find_all('th')))
                        docx_table.style = 'Table Grid'
                        for row in child.find_all('tr'):
                            cells = row.find_all(['td', 'th'])
                            row_cells = docx_table.add_row().cells
                            for i, cell in enumerate(cells):
                                row_cells[i].text = cell.get_text()
                        # Add the table to the document
                        paragraph._p.addnext(docx_table._element)
                    elif child.name == 'br':
                        parent.add_run("\n")
                    else:
                        walk(child, parent)

            walk(soup.body, paragraph)

        except Exception as e:
            paragraph.add_run(f"Error adding markdown text: {e}")

    # Output (with table parsing and markdown-style formatting)
    if output:
        doc.add_heading("Analysis Output", level=1)
        # Try to parse tables from output, otherwise add as preformatted text
        table_blocks = []
        other_blocks = []
        import re

        # Split output into blocks by double newlines
        blocks = re.split(r"\n\s*\n", output)
        for block in blocks:
            # Heuristic: if block looks like a table (multiple lines, columns separated by spaces)
            if re.search(r"\n\s*\w+\s+\w+\s+\w+", block):
                table_blocks.append(block)
            else:
                other_blocks.append(block)

        # Add non-table blocks as paragraphs, parsing markdown with HTML
        for block in other_blocks:
            if block.strip():
                html = markdown.markdown(block.strip())
                add_html_to_doc(doc, html)

        # Add tables (unchanged)
        for table_block in table_blocks:
            lines = [l for l in table_block.strip().split("\n") if l.strip()]
            if len(lines) < 2:
                doc.add_paragraph(table_block)
                continue
            # Try to parse header and rows
            header = re.split(r"\s{2,}", lines[0].strip())
            rows = [re.split(r"\s{2,}", l.strip()) for l in lines[1:]]
            # Remove any rows that don't match header length
            rows = [r for r in rows if len(r) == len(header)]
            if not rows:
                doc.add_paragraph(table_block)
                continue
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Table Grid"
            hdr_cells = table.rows[0].cells
            for i, h in enumerate(header):
                hdr_cells[i].text = h
            for row in rows:
                row_cells = table.add_row().cells
                for i, cell in enumerate(row):
                    row_cells[i].text = cell

    # Categorical mappings
    if categorical_mappings:
        doc.add_heading("Categorical Variable Encodings", level=1)
        for col, mapping in categorical_mappings.items():
            doc.add_heading(f"Column '{col}'", level=2)
            table = doc.add_table(rows=1, cols=2)
            table.style = "Table Grid"
            table.rows[0].cells[0].text = "Original Value"
            table.rows[0].cells[1].text = "Encoded Value"
            for orig, enc in mapping.items():
                row_cells = table.add_row().cells
                row_cells[0].text = str(orig)
                row_cells[1].text = str(enc)

    # Images
    if image_paths:
        doc.add_heading("Generated Plots", level=1)
        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    doc.add_picture(img_path, width=Inches(5.5))
                except Exception:
                    doc.add_paragraph(f"[Image could not be loaded: {img_path}]")

    # Save the document
    docx_file_path = file_name + ".docx"
    doc.save(docx_file_path)
    return docx_file_path


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
    output_file = generate_gpt_analysis_docx(
        project_name,
        "Example question?",
        "This is a research summary.",
        "def hello_world():\n    print('Hello, World!')",
        "Sample output text.",
        image_paths=None,
        categorical_mappings=None,
    )
    print(f"Docx file created: {output_file}")
