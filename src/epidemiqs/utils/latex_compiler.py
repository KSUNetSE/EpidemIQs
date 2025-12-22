import subprocess
import os
import re
from pathlib import Path
from termcolor import colored
from os.path import join as ospj

def clean_document_env(file_path):
    lines = Path(file_path).read_text().splitlines()

    begin_found = False
    end_indices = []

    cleaned_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == r'\begin{document}':
            if not begin_found:
                begin_found = True
                cleaned_lines.append(line)
        elif stripped == r'\end{document}':
            end_indices.append(idx)
        else:
            cleaned_lines.append(line)

    if end_indices:
        cleaned_lines.append(r'\end{document}')

    Path(file_path).write_text('\n'.join(cleaned_lines))
    
def latex_linter(file_path):
    clean_document_env(file_path)
    tex = Path(file_path).read_text()
    cleaned_tex = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?)\{.*[/\\]([^/\\}]+)\}', r'\1{\2}', tex)
    cleaned_tex = re.sub(r'\\begin{figure}(\[[^\]]*\])?', r'\\begin{figure}[http]', cleaned_tex, flags=re.IGNORECASE)


    Path(file_path).write_text(cleaned_tex)
    

"""
compiles a .tex file to PDF using pdflatex.
may need to be run multiple times for complex documents
(e.g., for table of contents, citations).
"""
def compile_tex(tex_filepath):
    if not os.path.exists(tex_filepath):
        print(f"Error: File not found at {tex_filepath}")
        return False
    latex_linter(tex_filepath)
    tex_dir = os.path.dirname(os.path.abspath(tex_filepath))
    tex_filename = os.path.basename(tex_filepath)
 
    num_runs = 10
    for i in range(num_runs):
        print(f"--- pdflatex: Run {i + 1} of {num_runs} on {tex_filename} ---")
        
        try:
            if i < 5:
                subprocess.run(
                    [
                        'pdflatex',
                        # Prevents stopping on minor errors
                        '-interaction=nonstopmode', 
                        # Save output to the .tex file's directory
                        '-output-directory=' + tex_dir, 
                        tex_filename
                    ],
                    cwd=tex_dir,  
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # Raises CalledProcessError if pdflatex returns a non-zero exit code
                    check=True  
                )
                print(f"pdflatex run {i + 1} successful.")
                # Optional: print stdout for more details
                # print("Stdout:", process.stdout.decode(errors='ignore'))
            else:
                subprocess.run(
                [
                    'latexmk',
                    '-pdf',
                    '-interaction=nonstopmode',
                    '-output-directory=' + tex_dir,
                    tex_filename
                ],
                cwd=tex_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

        except subprocess.CalledProcessError as e:
            print(f"Error during pdflatex compilation (run {i + 1}):")
            #print("Stdout:", e.stdout.decode(errors='ignore'))
           # print("Stderr:", e.stderr.decode(errors='ignore'))
            print(f"Check the .log file in {tex_dir} for more detailed LaTeX errors.")
            #return None
            pdf_basename = os.path.splitext(tex_filename)[0] + '.pdf'
            pdf_filepath = os.path.join(tex_dir, pdf_basename)

            if os.path.exists(os.path.join(pdf_filepath)):
                print(colored(f"PDF generated despite Errors . Check latexmk.log in {tex_dir} for details.","green"))
            else:
                print(f"PDF generation failed. Output PDF not found at {pdf_filepath}")
            log_filename = os.path.splitext(tex_filename)[0] + '.log'
            print(f"Check the LaTeX log file for details: {os.path.join(tex_dir, log_filename)}")
            pass
        except FileNotFoundError:
            print("Error: 'pdflatex' command not found.")
            print("Please ensure a LaTeX distribution (like TeX Live or MiKTeX) is installed ")
            print("and the 'pdflatex' executable is in your system's PATH.")
            pdf_basename = os.path.splitext(tex_filename)[0] + '.pdf'
            pdf_filepath = os.path.join(tex_dir, pdf_basename)
            if os.path.exists(os.path.join(pdf_filepath)):
                print(colored(f"PDF generated despite Errors . Check latexmk.log in {tex_dir} for details.","green"))
            else:
                print(f"PDF generation failed. Output PDF not found at {pdf_filepath}")
            log_filename = os.path.splitext(tex_filename)[0] + '.log'
            print(f"Check the LaTeX log file for details: {os.path.join(tex_dir, log_filename)}")
            pass



import re
from pathlib import Path

def convert_tabular_to_tabularx(file_path):
    text = Path(file_path).read_text()
    def replace_tabular(match):
        col_format = match.group(1)
        content = match.group(2)
        new_format = col_format[0] + ''.join(['X' if c in 'lcr' else c for c in col_format[1:]])

        return f"\\begin{{tabularx}}{{\\textwidth}}{{{new_format}}}\n{content}\n\\end{{tabularx}}"

    text = re.sub(
        r'\\begin{tabular}\{([lcr| ]+)\}(.+?)\\end{tabular}',
        replace_tabular,
        text,
        flags=re.DOTALL
    )
    if '\\usepackage{tabularx}' not in text:
        text = re.sub(r'(\\usepackage[^\n]*\n)', r'\1\\usepackage{tabularx}\n', text, count=1)
    Path(file_path).write_text(text)
    print(f"Converted tabular to tabularx in: {file_path}")



import os
import re
from textwrap import indent

def safe_caption(text: str) -> str:
    """
    Removes characters that could break LaTeX captions.
    Underscores are replaced with spaces; other dangerous characters removed.
    """
    text = text.replace("_", " ")
    text = re.sub(r"[^a-zA-Z0-9 .-]", "", text)
    return text.strip()


def safe_label(text: str) -> str:
    """
    Create a LaTeX-safe label string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def generate_appendix_latex(images_dir,
                            section_title="Appendix: All Pipeline Figures",
                            image_width="0.35\\textwidth") -> str:
    supported_ext = {".png", ".jpg", ".jpeg"}
    
    images = [f for f in os.listdir(images_dir)
              if os.path.splitext(f)[1].lower() in supported_ext]
    images.sort()

    lines = []
    lines.append("\\section*{%s}" % section_title)
    lines.append("\\addcontentsline{toc}{section}{%s}" % section_title)
    lines.append("")

    for i in range(0, len(images), 2):
        img1 = images[i]
        img2 = images[i+1] if i+1 < len(images) else None

        cap1 = safe_caption(img1)
        lab1 = safe_label(img1)

        lines.append("\\begin{figure}[H]")
        lines.append("    \\centering")

        lines.append("    \\begin{subfigure}[b]{%s}" % image_width)
        lines.append("        \\centering")
        lines.append(f"        \\includegraphics[width=\\textwidth]{{{images_dir}/{img1}}}")
        lines.append(f"        \\caption*{{{cap1}}}")
        lines.append("    \\end{subfigure}")

        if img2:
            cap2 = safe_caption(img2)
            lab2 = safe_label(img2)

            lines.append("    \\begin{subfigure}[b]{%s}" % image_width)
            lines.append("        \\centering")
            lines.append(f"        \\includegraphics[width=\\textwidth]{{{images_dir}/{img2}}}")
            lines.append(f"        \\caption*{{{cap2}}}")
            lines.append("    \\end{subfigure}")

        combined_caption = f"Figures: {cap1}" + (f" and {cap2}" if img2 else "")
        lines.append(f"    \\caption{{{combined_caption}}}")
        lines.append(f"    \\label{{fig:{lab1}}}")
        lines.append("\\end{figure}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    dummy_tex_path = os.path.join(os.getcwd(),'output' ,'initial_report.tex')
    convert_tabular_to_tabularx(dummy_tex_path)

    if compile_tex(dummy_tex_path):
        print("Compilation with pdflatex successful.")
    else:
        print("Compilation with pdflatex failed.")
    print("-" * 30)