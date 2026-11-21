"""Build the project report: python paper/report/build_report.py"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
OUT = HERE / "BPSO_Project_Report.docx"

FONT = "Times New Roman"
ACCENT = RGBColor(0x0B, 0x25, 0x45)

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = Cm(3.0)
sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)

# ---------------------------------------------------------------- styles
st = doc.styles
normal = st["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
for name, size, before, after in [("Heading 1", 16, 0, 18), ("Heading 2", 13, 12, 6),
                                  ("Heading 3", 12, 10, 4)]:
    h = st[name]
    h.font.name = FONT
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.italic = False
    h.font.color.rgb = ACCENT
    h.paragraph_format.space_before = Pt(before)
    h.paragraph_format.space_after = Pt(after)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.line_spacing = 1.15
cap = st["Caption"]
cap.font.name = FONT
cap.font.size = Pt(10.5)
cap.font.italic = False
cap.font.bold = True
cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


# ---------------------------------------------------------------- helpers
def add_field(paragraph, instr, placeholder=""):
    """Insert a Word field (TOC, PAGE, SEQ ...). Word fills it when fields are updated."""
    r = paragraph.add_run()
    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    r._r.append(b)
    r2 = paragraph.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r2._r.append(it)
    r3 = paragraph.add_run()
    s = OxmlElement("w:fldChar")
    s.set(qn("w:fldCharType"), "separate")
    r3._r.append(s)
    r4 = paragraph.add_run(placeholder)
    r5 = paragraph.add_run()
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    r5._r.append(e)
    return r4


def para(text="", bold=False, italic=False, size=None, align=None, space_after=None,
         indent_first=False, keep=False):
    p = doc.add_paragraph()
    if text:
        add_runs(p, text, bold=bold, italic=italic, size=size)
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def add_runs(p, text, bold=False, italic=False, size=None):
    """Supports **bold** and _italic_ inline markers, plus ^{sup} and _{sub}."""
    import re
    tokens = re.split(r"(\*\*[^*]+\*\*|\^\{[^}]+\}|_\{[^}]+\}|\b_[^_]+_\b)", text)
    for tok in tokens:
        if not tok:
            continue
        b, i, sup, sub = bold, italic, False, False
        if tok.startswith("**"):
            tok, b = tok[2:-2], True
        elif tok.startswith("^{"):
            tok, sup = tok[2:-1], True
        elif tok.startswith("_{"):
            tok, sub = tok[2:-1], True
        elif tok.startswith("_") and tok.endswith("_") and len(tok) > 2:
            tok, i = tok[1:-1], True
        r = p.add_run(tok)
        r.bold, r.italic = b, i
        r.font.superscript, r.font.subscript = sup, sub
        if size:
            r.font.size = Pt(size)
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        add_runs(p, it)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def numbered(items):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(3)
        add_runs(p, f"{i}.\t{it}")


def equation(text, number):
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = Cm(13.5), Cm(2.0)
    p = c0.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, text, italic=False)
    for r in p.runs:
        r.font.name = "Cambria Math"
    q = c1.paragraphs[0]
    q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    q.add_run(f"({number})")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def caption(kind, text):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{kind} ")
    add_field(p, f"SEQ {kind} \\* ARABIC", "#")
    p.add_run(f": {text}")
    if kind == "Table":
        p.paragraph_format.keep_with_next = True
    return p


def table(rows, widths, title, font_size=10.5, header=True):
    caption("Table", title)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.width = Cm(widths[c])
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(font_size)
            if header and r == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "0B2545")
            elif r % 2 == 0:
                shade(cell, "F1F5F9")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(name, width_cm, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    caption("Figure", title)


def chapter(title):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    return doc.add_heading(title, level=1)


def h2(t):
    doc.add_heading(t, level=2)


def h3(t):
    doc.add_heading(t, level=3)


def code(lines):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    cell.width = Cm(15.5)
    shade(cell, "F1F5F9")
    first = True
    for ln in lines.strip("\n").split("\n"):
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(ln)
        r.font.name = "Consolas"
        r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def centered(text, size=12, bold=False, after=6, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size, r.bold, r.italic = Pt(size), bold, italic
    p.paragraph_format.space_after = Pt(after)
    return p


def signature_block(left, right):
    t = doc.add_table(rows=1, cols=2)
    for cell, lines in zip(t.rows[0].cells, (left, right)):
        cell.width = Cm(7.75)
        first = True
        for ln in lines:
            p = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            p.paragraph_format.line_spacing = 1.15
            p.add_run(ln)


TEAM = ["Shashank Kushwaha", "Anshika Srivastava", "Muskan Mishra", "Roushan Kumar"]
TITLE = "Swarm Intelligence for Feature Selection in Disease Prediction"

# ================================================================ TITLE PAGE
centered("A Project Report on", 14, after=12)
centered(TITLE, 22, bold=True, after=18)
centered("Submitted in partial fulfilment of the requirements for the award of the degree of",
         12, italic=True, after=4)
centered("Bachelor of Technology", 14, bold=True, after=2)
centered("in", 12, after=2)
centered("Computer Science and Engineering (Artificial Intelligence and Machine Learning)", 13,
         bold=True, after=24)
centered("Submitted by", 12, italic=True, after=6)
for n in TEAM:
    centered(f"{n}    (Roll No. ____________)", 12, after=2)
centered("", after=14)
centered("Under the guidance of", 12, italic=True, after=4)
centered("Niharika Chaudhary", 13, bold=True, after=2)
centered("Assistant Professor", 12, after=40)
centered("Department of Computer Science and Engineering (AIML)", 13, bold=True, after=2)
centered("IILM University, Greater Noida, Uttar Pradesh, India", 12, after=2)
centered("Academic Year 2026–27", 12, after=2)

# ================================================================ CERTIFICATE
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("CERTIFICATE", 16, bold=True, after=18)
para("This is to certify that the project report entitled **“" + TITLE + "”** submitted by "
     + ", ".join(TEAM[:-1]) + " and " + TEAM[-1] + " in partial fulfilment of the requirements "
     "for the award of the degree of Bachelor of Technology in Computer Science and Engineering "
     "(AIML) at IILM University, Greater Noida, is a record of bona fide work carried out by them "
     "under my supervision and guidance.")
para("To the best of my knowledge, the matter embodied in this report has not been submitted to "
     "any other university or institute for the award of any degree or diploma.")
for _ in range(4):
    doc.add_paragraph()
signature_block(
    ["______________________", "Niharika Chaudhary", "Assistant Professor (Project Guide)",
     "Dept. of CSE (AIML), IILM University"],
    ["______________________", "Head of Department", "Dept. of CSE (AIML)", "IILM University"],
)
doc.add_paragraph()
para("Date: ____________          Place: Greater Noida", align=WD_ALIGN_PARAGRAPH.LEFT)

# ================================================================ DECLARATION
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("DECLARATION", 16, bold=True, after=18)
para("We hereby declare that the project report entitled **“" + TITLE + "”** is an authentic "
     "record of our own work carried out under the guidance of Niharika Chaudhary, Assistant "
     "Professor, Department of Computer Science and Engineering (AIML), IILM University, Greater "
     "Noida. The work has not been submitted elsewhere for the award of any degree or diploma. "
     "All sources of information and assistance used have been duly acknowledged.")
for _ in range(3):
    doc.add_paragraph()
t = doc.add_table(rows=len(TEAM) + 1, cols=3)
t.style = "Table Grid"
for c, hd in enumerate(["Name", "Roll No.", "Signature"]):
    t.cell(0, c).text = hd
    t.cell(0, c).paragraphs[0].runs[0].bold = True
for r, n in enumerate(TEAM, 1):
    t.cell(r, 0).text = n
for row in t.rows:
    row.height = Cm(1.1)
    for c, wd in enumerate([6.5, 4.0, 5.0]):
        row.cells[c].width = Cm(wd)
doc.add_paragraph()
para("Date: ____________", align=WD_ALIGN_PARAGRAPH.LEFT)

# ================================================================ ACKNOWLEDGEMENT
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("ACKNOWLEDGEMENT", 16, bold=True, after=18)
para("We express our sincere gratitude to our project guide, **Niharika Chaudhary**, Assistant "
     "Professor, Department of Computer Science and Engineering (AIML), IILM University, for her "
     "guidance, encouragement and valuable feedback throughout this project.")
para("We thank the Head of the Department and all faculty members of the Department of Computer "
     "Science and Engineering (AIML) for providing the facilities and support needed to complete "
     "this work. We also thank the maintainers of the UCI Machine Learning Repository and the "
     "open-source scientific Python community, whose datasets and tools made this study possible.")
para("Finally, we thank our families and friends for their constant support and motivation.")
for _ in range(3):
    doc.add_paragraph()
for n in TEAM:
    para(n, align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=0)

# ================================================================ ABSTRACT
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("ABSTRACT", 16, bold=True, after=18)
para("Feature selection is an important stage in machine-learning-based disease prediction, "
     "because irrelevant or redundant clinical variables increase computational cost, make "
     "models harder to interpret and can require unnecessary medical tests. This project applies "
     "Binary Particle Swarm Optimization (BPSO), a swarm-intelligence algorithm, to select compact "
     "feature subsets for binary disease classification. Each particle represents a candidate "
     "subset as a binary vector, and a fitness function balances inner cross-validated accuracy "
     "against the number of selected features.")
para("BPSO was evaluated on five public benchmark datasets — breast cancer, chronic kidney "
     "disease, heart disease, liver disease and diabetes — and compared with nine alternatives: "
     "all features, a binary genetic algorithm, mutual-information and ANOVA-F filters, recursive "
     "feature elimination, L1-regularised logistic regression and random subsets. Five "
     "classifiers were used, and every method was evaluated with 5 × repeated stratified 10-fold "
     "nested cross-validation (12,500 test evaluations), so that feature selection and tuning "
     "never used test data.")
para("BPSO removed 38–58 % of the features on every dataset while keeping the F1-score within "
     "about 0.015 of the all-feature model, and it selected clinically plausible features such as "
     "glucose and BMI for diabetes. However, it did not outperform simpler alternatives: its "
     "average Friedman rank was 7.20 of 10 (p < 10^{−12}), similar to the genetic algorithm and "
     "behind all features and cross-validated filters. On these low-dimensional datasets, BPSO is "
     "best suited to building compact, interpretable models rather than improving accuracy. The "
     "system was also deployed as a public prediction API and interactive web dashboard.")
para("**Keywords:** swarm intelligence, feature selection, Binary Particle Swarm Optimization, "
     "disease prediction, machine learning, nested cross-validation.",
     align=WD_ALIGN_PARAGRAPH.LEFT)

# ================================================================ TOC
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("TABLE OF CONTENTS", 16, bold=True, after=12)
add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u',
          "Right-click and choose Update Field to build the table of contents.")
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
centered("LIST OF FIGURES", 16, bold=True, after=12)
add_field(doc.add_paragraph(), 'TOC \\h \\z \\c "Figure"', "Right-click and choose Update Field.")
doc.add_paragraph()
centered("LIST OF TABLES", 16, bold=True, after=12)
add_field(doc.add_paragraph(), 'TOC \\h \\z \\c "Table"', "Right-click and choose Update Field.")

# ================================================================ CH 1
chapter("Chapter 1: Introduction")
h2("1.1 Background")
para("Machine learning is increasingly used to analyse medical data and support disease-risk "
     "prediction. A clinical dataset may contain many variables — laboratory results, vital "
     "signs, demographic information and imaging measurements — but not every variable "
     "contributes equally to prediction. Irrelevant or redundant features increase training "
     "time, make models harder to interpret and can reduce generalisation.")
para("In clinical settings the cost of collecting each variable is also a practical concern. "
     "Every additional measurement may require a test, time and resources from both the patient "
     "and the healthcare provider. A model that predicts well from fewer measurements is "
     "therefore cheaper to use and easier for clinicians to trust.")
h2("1.2 Motivation")
para("Feature selection identifies a useful subset of the available variables while retaining "
     "predictive information. Because the number of possible subsets grows as 2^{d} for d "
     "features, exhaustive search is impossible for all but the smallest datasets: 30 features "
     "already give more than one billion subsets. Swarm-intelligence algorithms are "
     "population-based optimisation methods inspired by collective behaviour in nature, such as "
     "bird flocks and fish schools, and are well suited to this kind of large search space. "
     "Binary Particle Swarm Optimization (BPSO) represents feature selection directly as a "
     "search over binary vectors, where each bit indicates whether a feature is used.")
para("Many published studies apply BPSO to a single dataset with a single train/test split and "
     "compare it only with the full feature set. Such evaluations can be optimistic, especially "
     "when feature selection is accidentally performed on data that is later used for testing. "
     "This project therefore evaluates BPSO under a strict, leakage-free protocol on several "
     "diseases and against a broad set of alternatives.")
h2("1.3 Problem Statement")
para("Given a labelled medical dataset with d input features X = {x_{1}, x_{2}, …, x_{d}}, the "
     "goal is to find a binary selection vector z = [z_{1}, …, z_{d}], z_{j} ∈ {0, 1}, where "
     "z_{j} = 1 means feature j is used, such that a classifier trained on the selected subset "
     "X_{z} = {x_{j} | z_{j} = 1} achieves good predictive performance with as few features as "
     "practical.")
h2("1.4 Objectives")
numbered([
    "To implement a BPSO wrapper feature-selection method with a size-penalised fitness function.",
    "To evaluate it on five public disease datasets using nested, leakage-free repeated "
    "cross-validation.",
    "To compare BPSO with eight alternative feature-selection strategies and the full feature "
    "set across five classifiers.",
    "To analyse the results statistically, including significance tests, ranking, selection "
    "stability, runtime and sensitivity to the trade-off parameter α.",
    "To deploy a working prediction service and dashboard that demonstrate the system.",
])
h2("1.5 Scope of the Project")
para("The project covers binary disease classification on tabular clinical data from public "
     "benchmarks. It does not cover image or signal data, multi-class diagnosis or clinical "
     "deployment. The models are for research and educational use only and are not clinically "
     "validated.")
h2("1.6 Organisation of the Report")
para("Chapter 2 reviews related work. Chapter 3 lists the system requirements. Chapter 4 "
     "describes the datasets and preprocessing. Chapter 5 presents the proposed methodology, "
     "including the BPSO algorithm, fitness function and evaluation protocol. Chapter 6 "
     "describes the implementation and deployment. Chapter 7 reports and discusses the results, "
     "and Chapter 8 concludes the report with limitations and future work.")

# ================================================================ CH 2
chapter("Chapter 2: Literature Review")
h2("2.1 Particle Swarm Optimization")
para("Particle Swarm Optimization (PSO) was introduced by Kennedy and Eberhart [1] as a "
     "population-based optimisation method inspired by the social behaviour of bird flocks. Each "
     "particle moves through the search space under the influence of its own best position and "
     "the best position found by the swarm. Shi and Eberhart [2] added the inertia weight, which "
     "controls the balance between exploration and exploitation. Kennedy and Eberhart later "
     "proposed a discrete binary version [3], in which velocities are mapped to probabilities by "
     "a sigmoid function and positions are binary — the formulation used in this project.")
h2("2.2 Feature Selection Approaches")
para("Feature-selection methods are commonly grouped into three families [4]. **Filter** methods "
     "rank features with statistical measures, such as mutual information or the ANOVA F-score, "
     "independently of any learning algorithm. **Wrapper** methods evaluate candidate subsets by "
     "the performance of a classifier trained on them [5]; they are often more accurate for a "
     "given classifier but more expensive. **Embedded** methods perform selection during model "
     "training, as in L1-regularised (lasso) regression [6] or recursive feature elimination with "
     "support vector machines [7]. Population-based search methods such as PSO and genetic "
     "algorithms reduce the number of subsets that must be evaluated in a wrapper framework [8].")
h2("2.3 Swarm Intelligence in Disease Prediction")
para("Chuang et al. [9] applied an improved binary PSO to gene-expression data, a biomedical "
     "setting with a very large number of features. Several studies have applied swarm methods "
     "to heart-related diagnosis. Elbedwehy et al. [10] used BPSO to select features from heart "
     "sound recordings before classification with a support vector machine. Pathan et al. [11] "
     "examined how feature selection affects the accuracy of heart disease prediction. Fajri et "
     "al. [12] combined bee swarm optimisation with Q-learning for coronary heart disease "
     "diagnosis. Namlı et al. [13] used BPSO to choose features for each base classifier of an "
     "ensemble for coronary artery disease. Zomorodi-Moghadam et al. [14] used a hybrid "
     "binary-real PSO to discover classification rules, reporting that a rule set built on 11 of "
     "13 attributes was more accurate than one built on all 13.")
table([
    ["Study", "Method", "Data", "Note"],
    ["Chuang et al. [9]", "Improved BPSO", "Gene expression", "Very high-dimensional data"],
    ["Elbedwehy et al. [10]", "BPSO + SVM", "Heart sounds", "Valve disease detection"],
    ["Pathan et al. [11]", "Feature-selection study", "Heart disease", "Effect on accuracy"],
    ["Fajri et al. [12]", "Bee swarm + Q-learning", "Coronary heart disease", "Hybrid selection"],
    ["Namlı et al. [13]", "BPSO + ensemble", "Coronary artery disease", "Two-stage PSO"],
    ["Zomorodi-Moghadam et al. [14]", "Hybrid PSO rules", "Coronary artery disease",
     "11 of 13 attributes"],
    ["This project", "BPSO vs 9 methods", "5 disease datasets", "Nested repeated CV + statistics"],
], [4.2, 3.6, 3.8, 4.0], "Summary of related work")
h2("2.4 Research Gap")
para("Most of these studies evaluate one dataset, often with a single train/test split, and "
     "compare BPSO only with the full feature set. Few compare BPSO with simple filter and "
     "embedded methods at the same subset size, test statistical significance across several "
     "datasets, or report selection stability. This project addresses that gap with a "
     "multi-dataset, multi-classifier comparison under nested cross-validation, using the "
     "statistical procedures recommended by Demšar [15].")

# ================================================================ CH 3
chapter("Chapter 3: System Requirements")
h2("3.1 Hardware Requirements")
table([
    ["Component", "Used in this project", "Minimum"],
    ["Processor", "12-thread x86-64 CPU", "Dual-core 64-bit CPU"],
    ["Memory", "16 GB RAM (approx.)", "4 GB RAM"],
    ["Storage", "~1 GB for code, data and results", "2 GB free"],
    ["Operating system", "Windows 11 with WSL2 (Linux) and Docker", "Windows, Linux or macOS"],
], [4.0, 6.5, 5.0], "Hardware requirements")
h2("3.2 Software Requirements")
table([
    ["Software / library", "Version", "Purpose"],
    ["Python", "3.11", "Programming language"],
    ["scikit-learn", "1.9", "Classifiers, preprocessing, cross-validation"],
    ["NumPy / pandas / SciPy", "2.4 / 3.1 / 1.17", "Numerical computing, data handling, statistics"],
    ["matplotlib / seaborn", "3.11 / –", "Figures"],
    ["FastAPI / Uvicorn", "≥ 0.110", "Prediction web API"],
    ["Streamlit", "≥ 1.47", "Interactive dashboard"],
    ["Docker", "–", "Packaging and deployment"],
    ["pytest, GitHub Actions", "–", "Automated testing"],
    ["Git / GitHub", "–", "Version control and code hosting"],
    ["Render", "Free tier", "Cloud hosting of the API and dashboard"],
], [4.6, 3.2, 7.7], "Software requirements")

# ================================================================ CH 4
chapter("Chapter 4: Datasets and Preprocessing")
h2("4.1 Datasets")
para("Five public binary disease-classification datasets were used (Table 4). In every dataset "
     "the positive class (label 1) indicates the presence of disease.")
table([
    ["Dataset", "Samples", "Features", "Disease (%)", "Missing cells", "Source"],
    ["Breast Cancer (WDBC)", "569", "30", "212 (37.3)", "0", "UCI / scikit-learn [16]"],
    ["Chronic Kidney Disease", "400", "24", "250 (62.5)", "1013", "UCI [17]"],
    ["Heart Disease (Cleveland)", "303", "13", "139 (45.9)", "6", "UCI [18]"],
    ["Indian Liver Patient (ILPD)", "583", "10", "416 (71.4)", "4", "UCI [19]"],
    ["Pima Indians Diabetes", "768", "8", "268 (34.9)", "652", "UCI / NIDDK [20]"],
], [4.4, 1.7, 1.8, 2.3, 2.1, 3.3], "Datasets used in the study")
bullets([
    "**Breast Cancer (WDBC):** 30 features computed from digitised images of fine-needle "
    "aspirates of breast masses; the task is malignant vs benign.",
    "**Chronic Kidney Disease:** 24 clinical and laboratory attributes, including several "
    "categorical variables and many missing values.",
    "**Heart Disease (Cleveland):** 13 attributes; the original 0–4 target is binarised into "
    "absence (0) or presence (1–4) of heart disease, following common practice [21].",
    "**Indian Liver Patient:** 10 attributes from liver-function tests; the classes are "
    "imbalanced (71.4 % positive).",
    "**Pima Indians Diabetes:** 8 attributes; physiologically impossible zero values of "
    "glucose, blood pressure, skin thickness, insulin and BMI are treated as missing.",
])
h2("4.2 Preprocessing")
para("All preprocessing steps are combined into a single scikit-learn pipeline that is fitted "
     "on the training fold only and then applied to the corresponding test fold:")
numbered([
    "Missing numeric values are imputed with the training-fold median.",
    "Missing categorical values are imputed with the training-fold most frequent value, and "
    "nominal variables are ordinal-encoded.",
    "All features are standardised to zero mean and unit variance.",
    "Input features are separated from the target label, and stratified partitions keep the "
    "class proportions similar in every fold.",
])
para("Fitting imputation or scaling statistics on the full dataset would leak information from "
     "the test data into training and overstate performance. Keeping these steps inside the "
     "pipeline prevents that.")

# ================================================================ CH 5
chapter("Chapter 5: Proposed Methodology")
h2("5.1 Overall Workflow")
para("The workflow consists of dataset loading, outer data splitting, preprocessing, BPSO "
     "feature selection, classifier training with hyperparameter tuning, and final evaluation "
     "(Figure 1). Steps 3–5 use only the outer training fold; the held-out fold is used once, "
     "for scoring.")
flow = doc.add_table(rows=1, cols=11)
flow.alignment = WD_TABLE_ALIGNMENT.CENTER
labels = ["Dataset", "Outer 10-fold split", "Preprocess (train fold)", "BPSO feature selection",
          "Train + tune classifier", "Test on held-out fold"]
for i, cell in enumerate(flow.rows[0].cells):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    if i % 2 == 0:
        cell.width = Cm(2.3)
        r = p.add_run(labels[i // 2])
        r.bold = True
        r.font.size = Pt(9.5)
        shade(cell, "D6F1EE" if i != 6 else "0B2545")
        if i == 6:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    else:
        cell.width = Cm(0.5)
        p.add_run("→").font.size = Pt(14)
caption("Figure", "Overall workflow; steps 2–6 repeat for 10 folds × 5 seeds per dataset")
h2("5.2 Binary Particle Swarm Optimization")
para("Each particle encodes a candidate feature subset as a binary position vector x_{i} and has "
     "a real-valued velocity vector v_{i}. At iteration t the velocity is updated as")
equation("v_{i}^{t+1} = w(t)·v_{i}^{t} + c_{1}r_{1}(p_{i} − x_{i}^{t}) + c_{2}r_{2}(g − x_{i}^{t})", 1)
para("where w is the inertia weight, c_{1} and c_{2} are acceleration coefficients, r_{1} and "
     "r_{2} are uniform random numbers in [0, 1], p_{i} is the particle’s personal best and g is "
     "the global best. The first term keeps the previous motion, the second pulls the particle "
     "toward its own best position and the third toward the swarm’s best. Each velocity is "
     "clamped to [−v_{max}, v_{max}], and the inertia weight decreases linearly:")
equation("w(t) = w_{max} − (w_{max} − w_{min}) · t / T", 2)
para("The velocity is converted to a probability with the sigmoid transfer function, and each "
     "bit is sampled from it:")
equation("S(v_{ij}) = 1 / (1 + e^{−v_{ij}}),     x_{ij} = 1 if u < S(v_{ij}), else 0,  u ~ U(0,1)", 3)
para("A large positive velocity therefore makes a feature very likely to be selected, and a "
     "large negative velocity makes it very likely to be dropped. If a particle selects no "
     "features, one randomly chosen feature is switched on (repair). When two solutions have the "
     "same fitness, the one with fewer features is preferred, and fitness values are cached per "
     "unique subset so that revisited subsets are not re-evaluated.")
h3("Algorithm: BPSO feature selection")
code("""
Input : training data (X, y), swarm size N, iterations T, alpha
Output: best feature mask g

1  Initialise N random binary masks x_i (each bit = 1 with prob. 0.5)
2  Initialise velocities v_i uniformly in [-v_max, v_max]; repair empty masks
3  Evaluate fitness F(x_i); set personal bests p_i = x_i; set global best g
4  for t = 1 .. T do
5      w <- linearly decreasing inertia weight
6      for each particle i do
7          v_i <- w*v_i + c1*r1*(p_i - x_i) + c2*r2*(g - x_i);  clip to [-v_max, v_max]
8          x_ij <- 1 if rand() < sigmoid(v_ij) else 0;  repair if empty
9          if F(x_i) better than F(p_i) then p_i <- x_i
10     g <- best of all p_i (ties broken by fewer features)
11 return g
""")
table([
    ["Parameter", "Value"],
    ["Swarm size", "20 particles"],
    ["Iterations T", "50"],
    ["Inertia weight w", "linear 0.9 → 0.4"],
    ["Acceleration c1, c2", "2.0, 2.0"],
    ["Velocity clamp v_max", "6.0"],
    ["Transfer function", "S-shaped (sigmoid)"],
    ["Initial bit probability", "0.5"],
    ["Empty-subset handling", "Repair (switch on one random feature)"],
    ["Fitness weight α", "0.99"],
    ["Wrapper classifier in fitness", "Logistic regression"],
    ["Inner CV for fitness", "5-fold stratified, accuracy"],
], [6.0, 7.0], "BPSO hyperparameters")
h2("5.3 Fitness Function")
para("The fitness function, which BPSO minimises, balances predictive performance and subset "
     "size:")
equation("F(z) = α · (1 − M(z)) + (1 − α) · |z| / d", 4)
para("where M(z) is the mean 5-fold stratified cross-validated accuracy of a logistic-regression "
     "model trained on the selected features (computed on the outer training fold only), |z| is "
     "the number of selected features, d is the total number of features and α ∈ [0, 1] sets the "
     "trade-off. With α = 0.99 accuracy dominates and subset size mainly breaks near-ties; "
     "smaller values of α favour smaller subsets. An empty subset receives the worst fitness, 1. "
     "For example, an accuracy of 0.80 with 4 of 8 features gives "
     "F = 0.99 × 0.20 + 0.01 × 0.5 = 0.203.")
h2("5.4 Comparison Methods")
table([
    ["Method", "Description"],
    ["All features", "No selection (reference)"],
    ["Binary GA", "Genetic algorithm with the same fitness; tournament size 3, crossover 0.9, "
     "mutation 1/d, one elite"],
    ["MI (k = BPSO)", "Top-k mutual-information features, k = median BPSO subset size"],
    ["MI (k by CV)", "Top-k mutual-information features, k chosen by inner CV"],
    ["ANOVA-F (k = BPSO)", "Top-k ANOVA F-score features, k = BPSO size"],
    ["ANOVA-F (k by CV)", "Top-k ANOVA F-score features, k chosen by inner CV"],
    ["RFE (k = BPSO)", "Recursive feature elimination to the BPSO size"],
    ["L1-LogReg", "Features with non-zero L1 logistic-regression coefficients (C by inner CV)"],
    ["Random (k = BPSO)", "Random subset of the BPSO size (sanity baseline)"],
], [4.0, 11.5], "Feature-selection methods compared with BPSO")
para("The “k = BPSO” variants test whether BPSO’s features are better than other features of "
     "the same count; the random baseline checks that BPSO is selecting meaningfully.")
h2("5.5 Classifiers")
table([
    ["Classifier", "Hyperparameter grid (inner CV)"],
    ["Logistic Regression", "C ∈ {0.01, 0.1, 1, 10}"],
    ["SVM (RBF kernel)", "C ∈ {0.1, 1, 10}, γ ∈ {scale, 0.01, 0.1}"],
    ["Random Forest", "200 trees, max depth ∈ {None, 5, 10}, min samples per leaf ∈ {1, 3}"],
    ["k-Nearest Neighbours", "k ∈ {3, 5, 7, 11}"],
    ["Gaussian Naive Bayes", "—"],
], [4.5, 11.0], "Classifiers and tuning grids")
h2("5.6 Evaluation Protocol")
bullets([
    "**Outer loop:** stratified 10-fold cross-validation repeated with 5 seeds (0–4), giving 50 "
    "test folds per dataset. All methods use identical folds, so comparisons are paired.",
    "**Inside each training fold:** preprocessing, feature selection and classifier grid "
    "search. The test fold is used only once, for scoring (nested cross-validation).",
    "**Scale:** 10 methods × 5 classifiers × 50 folds = 2,500 evaluations per dataset, 12,500 "
    "in total, plus an α-sensitivity study (α ∈ {0.80, 0.90, 0.95, 0.99}, 2 seeds × 10 folds).",
])
h2("5.7 Evaluation Metrics")
para("With TP, TN, FP and FN denoting true positives, true negatives, false positives and false "
     "negatives, and the disease class as positive:")
equation("Accuracy = (TP + TN) / (TP + TN + FP + FN)", 5)
equation("Precision = TP / (TP + FP),     Recall = TP / (TP + FN)", 6)
equation("F1 = 2 · Precision · Recall / (Precision + Recall)", 7)
para("ROC-AUC is computed from predicted probabilities. Every metric is reported as mean ± "
     "standard deviation over the 50 outer folds. Feature reduction is (d − k) / d × 100 %, where "
     "k is the number of selected features.")
h2("5.8 Statistical Analysis")
bullets([
    "Paired two-sided **Wilcoxon signed-rank tests** between BPSO and each other method over the "
    "50 folds, with **Holm correction** for multiple comparisons [22].",
    "A **Friedman test** over 25 blocks (5 datasets × 5 classifiers) with the **Nemenyi** "
    "critical difference (CD) for post-hoc comparison [15].",
    "**Selection stability:** mean pairwise Jaccard similarity of the subsets selected in "
    "different folds.",
])

# ================================================================ CH 6
chapter("Chapter 6: Implementation")
h2("6.1 Project Structure")
table([
    ["Path", "Purpose"],
    ["src/bpso_fs/selectors/bpso.py", "BPSO algorithm"],
    ["src/bpso_fs/fitness.py", "Fitness function with caching"],
    ["src/bpso_fs/selectors/", "Genetic algorithm, filters, RFE, L1 and random baselines"],
    ["src/bpso_fs/data/", "Dataset loaders and preprocessing pipeline"],
    ["src/bpso_fs/experiment.py", "Nested cross-validation experiment runner"],
    ["src/bpso_fs/reporting.py, plotting.py", "Statistical tests, tables and figures"],
    ["src/bpso_fs/api.py", "FastAPI prediction service"],
    ["app.py", "Streamlit dashboard"],
    ["scripts/", "Command-line entry points (run experiments, make tables and figures, train "
     "the demo model)"],
    ["configs/", "YAML experiment configurations"],
    ["tests/", "Automated tests (pytest)"],
    ["results_experiment/", "Per-fold results of the five-dataset run"],
    ["paper/", "Figures, tables, LaTeX paper and documentation"],
], [6.0, 9.5], "Project structure")
h2("6.2 Core BPSO Update")
para("The core of the BPSO implementation updates all particles at once with NumPy arrays:")
code("""
for t in range(self.n_iterations):
    w_t = self.inertia_at(t)
    r1 = rng.random((n, d))
    r2 = rng.random((n, d))
    vel = w_t * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest[None, :] - pos)
    np.clip(vel, -self.v_max, self.v_max, out=vel)

    rho = rng.random((n, d))
    pos = (rho < sigmoid(vel)).astype(np.int8)   # S-shaped transfer function
    pos = self._repair(pos, rng)                  # no empty subsets

    fit, score = self._evaluate(fit_fn, pos)      # cached fitness evaluations
    ...                                           # update personal and global bests
""")
h2("6.3 Experiment Pipeline")
para("The experiment runner loads each dataset, creates the repeated stratified outer folds and, "
     "for every fold, method and classifier, fits the preprocessing pipeline, the feature "
     "selector and a grid-searched classifier on the training fold before scoring the test fold. "
     "Seeds are derived deterministically from the (seed, fold) pair, so runs are reproducible. "
     "Per-fold metrics, predictions, selected features and BPSO/GA convergence histories are "
     "saved, and separate scripts produce the tables, figures and statistical tests used in this "
     "report.")
h2("6.4 Prediction API and Dashboard")
para("For demonstration, a logistic-regression model is trained on the full breast-cancer "
     "dataset using 16 of 30 features selected by BPSO, and served by a FastAPI application with "
     "three endpoints:")
table([
    ["Endpoint", "Method", "Description"],
    ["/health", "GET", "Reports whether the model is loaded"],
    ["/metadata", "GET", "Model details, including the selected features"],
    ["/predict", "POST", "Returns a label and disease probability for up to 1,000 records"],
], [3.0, 2.0, 10.5], "API endpoints")
para("A Streamlit dashboard provides four tabs: an overview, an interactive explorer of the "
     "cross-validation results (by dataset, classifier and metric), a prediction page that scores "
     "uploaded CSV files through the API, and a methodology and provenance page.")
h2("6.5 Deployment")
para("Both services are packaged in a Docker image and deployed on Render as two web services "
     "defined in a render.yaml blueprint. The API trains the demo model at start-up and exposes a "
     "health check. The live dashboard is available at https://bpso-showcase.onrender.com and the "
     "API documentation at https://bpso-api.onrender.com/docs. The source code is hosted on "
     "GitHub, which redeploys both services on every push to the main branch.")
h2("6.6 Testing")
para("The repository contains 18 automated pytest tests in four files, run by GitHub Actions on "
     "each push: BPSO behaviour (9 tests), the fitness function (6), absence of data leakage in "
     "the pipeline (2) and an end-to-end smoke test of the experiment (1).")

# ================================================================ CH 7
chapter("Chapter 7: Results and Discussion")
h2("7.1 Feature Reduction")
table([
    ["Measure", "WDBC", "CKD", "Heart", "ILPD", "Pima"],
    ["Original features d", "30", "24", "13", "10", "8"],
    ["Selected features k", "14.68 ± 2.17", "12.22 ± 2.04", "8.04 ± 1.55", "4.22 ± 1.18",
     "4.82 ± 1.04"],
    ["Feature reduction (%)", "51.1", "49.1", "38.2", "57.8", "39.8"],
    ["Stability (Jaccard)", "0.369", "0.482", "0.528", "0.474", "0.606"],
    ["Unique subsets evaluated", "1020", "1019", "786", "425", "190"],
    ["Runtime per fold (s)", "8.90 ± 0.57", "7.29 ± 0.21", "5.46 ± 0.48", "3.03 ± 0.22",
     "1.48 ± 0.10"],
], [4.1, 2.35, 2.35, 2.35, 2.35, 2.35], "BPSO feature reduction, stability and runtime "
   "(mean ± std over 50 folds)", font_size=10)
para("BPSO roughly halved the number of features on WDBC, CKD and ILPD and removed about 40 % on "
     "Heart and Pima.")
h2("7.2 Classification Performance")
table([
    ["Model", "Features", "WDBC", "CKD", "Heart", "ILPD", "Pima"],
    ["Logistic Regression", "All", "0.980", "0.996", "0.843", "0.722", "0.769"],
    ["Random Forest", "All", "0.961", "0.992", "0.821", "0.700", "0.761"],
    ["SVM", "All", "0.973", "0.999", "0.823", "0.709", "0.762"],
    ["Logistic Regression", "BPSO", "0.969", "0.982", "0.827", "0.723", "0.761"],
    ["Random Forest", "BPSO", "0.958", "0.994", "0.794", "0.700", "0.748"],
    ["SVM", "BPSO", "0.967", "0.988", "0.803", "0.707", "0.760"],
], [3.8, 1.9, 1.95, 1.95, 1.95, 1.95, 1.95], "Mean test accuracy with all features and with "
   "BPSO-selected features")
para("Accuracy changed by between −0.027 (random forest on Heart) and +0.002 (random forest on "
     "CKD); in most cases the change was within one standard deviation across folds.")
table([
    ["Dataset", "Classifier", "F1 (all)", "F1 (BPSO)", "ROC-AUC (all)", "ROC-AUC (BPSO)"],
    ["WDBC", "SVM", "0.963 ± 0.024", "0.956 ± 0.029", "0.996 ± 0.008", "0.994 ± 0.009"],
    ["CKD", "Random Forest", "0.994 ± 0.011", "0.996 ± 0.011", "1.000 ± 0.001", "1.000 ± 0.001"],
    ["Heart", "Logistic Reg.", "0.820 ± 0.074", "0.805 ± 0.065", "0.907 ± 0.051", "0.900 ± 0.052"],
    ["ILPD", "Logistic Reg.", "0.826 ± 0.024", "0.830 ± 0.024", "0.751 ± 0.060", "0.726 ± 0.062"],
    ["Pima", "Logistic Reg.", "0.626 ± 0.081", "0.614 ± 0.081", "0.833 ± 0.052", "0.824 ± 0.053"],
], [1.8, 2.7, 2.75, 2.75, 2.75, 2.75], "F1 and ROC-AUC with the best classifier for BPSO on each "
   "dataset", font_size=10)
para("After Holm correction, no F1 difference between BPSO and all features was significant on "
     "any dataset (p ≥ 0.06). ROC-AUC was significantly lower with BPSO on WDBC (p = 0.006), ILPD "
     "(p < 0.001) and Pima (p = 0.003).")
figure("method_comparison_breast_cancer.png", 15.5,
       "F1 and ROC-AUC over 50 folds for all ten methods on WDBC (SVM)")
h2("7.3 Comparison with Other Methods")
table([
    ["Dataset", "BPSO vs", "ΔF1", "Wins / ties / losses", "p (Holm)"],
    ["WDBC", "MI (k = BPSO)", "+0.028", "32 / 12 / 6", "0.0006"],
    ["WDBC", "ANOVA-F (k = BPSO)", "+0.028", "32 / 12 / 6", "0.0006"],
    ["WDBC", "MI (k by CV)", "−0.010", "6 / 22 / 22", "0.026"],
    ["WDBC", "ANOVA-F (k by CV)", "−0.008", "9 / 20 / 21", "0.040"],
    ["CKD", "Random (k = BPSO)", "+0.011", "22 / 25 / 3", "0.021"],
    ["Pima", "Random (k = BPSO)", "+0.068", "33 / 3 / 14", "0.0014"],
    ["Pima", "MI (k = BPSO)", "−0.017", "13 / 7 / 30", "0.039"],
], [2.2, 4.3, 2.0, 4.0, 2.5], "Significant paired Wilcoxon results (F1); all other comparisons "
   "were not significant")
para("BPSO’s subsets were significantly better than the top-ranked filter features of the same "
     "size on WDBC, whose 30 features are highly correlated, and better than random subsets on "
     "Pima and CKD. No difference between BPSO and the genetic algorithm was significant.")
table([
    ["Rank", "Method", "Average rank"],
    ["1", "All features", "3.68"], ["2", "ANOVA-F (k by CV)", "3.80"],
    ["3", "MI (k by CV)", "3.88"], ["4", "L1-LogReg", "4.08"], ["5", "RFE (k = BPSO)", "4.84"],
    ["6", "ANOVA-F (k = BPSO)", "5.70"], ["7", "MI (k = BPSO)", "5.78"],
    ["8", "BPSO (proposed)", "7.20"], ["9", "Binary GA", "7.28"], ["10", "Random (k = BPSO)", "8.76"],
], [1.5, 6.0, 3.0], "Average Friedman ranks by F1 over 25 dataset–classifier settings "
   "(χ² = 76.68, p = 7.3 × 10⁻¹³, CD = 2.71)")
figure("cd_diagram.png", 15.5, "Critical-difference diagram of average F1 ranks (Nemenyi, "
       "α = 0.05); methods joined by a bar are not significantly different")
para("Across all datasets and classifiers, BPSO ranked eighth of ten. It was statistically "
     "indistinguishable from the genetic algorithm, RFE, the fixed-size filters and random "
     "subsets, and significantly behind all features, both cross-validated filters and L1 "
     "logistic regression.")
h2("7.4 Convergence and Stability")
figure("convergence_heart_cleveland.png", 11.5,
       "Best fitness per iteration for BPSO and GA on the Heart dataset")
para("BPSO and the genetic algorithm reached almost the same best fitness, with most improvement "
     "in the first 15–20 iterations. The mean swarm fitness rose again after about 10 iterations, "
     "showing that the swarm kept exploring. Selection stability was moderate (Jaccard "
     "0.37–0.61), similar to the genetic algorithm and lower than L1 logistic regression "
     "(0.67–0.85) and the filters.")
h2("7.5 Selected Features")
table([
    ["Dataset", "Most frequently selected features (fraction of 50 folds)"],
    ["WDBC", "worst smoothness (0.90), worst symmetry (0.84), area error (0.76), worst concavity "
     "(0.72), mean texture (0.64)"],
    ["CKD", "specific gravity (1.00), haemoglobin (1.00), appetite (0.88), diabetes mellitus "
     "(0.88), hypertension (0.84)"],
    ["Heart", "number of major vessels (1.00), thalassemia (1.00), chest-pain type (0.82), "
     "maximum heart rate (0.76), sex (0.68)"],
    ["ILPD", "age (1.00), SGPT (0.86), alkaline phosphatase (0.56), total bilirubin (0.42)"],
    ["Pima", "glucose (1.00), BMI (1.00), diabetes pedigree (0.84), insulin (0.62), age (0.54)"],
], [2.2, 13.3], "Features most often selected by BPSO")
figure("feature_frequency_pima_diabetes.png", 12,
       "Selection frequency of each Pima feature across 50 BPSO runs")
para("Several of these agree with known clinical markers — glucose and BMI for diabetes, "
     "specific gravity and haemoglobin for chronic kidney disease, and vessel count and "
     "thalassemia for heart disease — which supports the interpretability of the selected "
     "subsets.")
h2("7.6 Sensitivity to α")
table([
    ["Dataset (classifier)", "α = 0.80", "α = 0.90", "α = 0.95", "α = 0.99"],
    ["WDBC (SVM)", "5.0 / 0.942", "6.8 / 0.953", "9.9 / 0.960", "14.3 / 0.961"],
    ["CKD (RF)", "3.9 / 0.977", "6.3 / 0.987", "7.7 / 0.995", "12.2 / 0.999"],
    ["Heart (LR)", "3.3 / 0.783", "4.0 / 0.790", "5.0 / 0.796", "7.7 / 0.808"],
    ["ILPD (LR)", "1.0 / 0.832", "1.4 / 0.833", "2.8 / 0.831", "4.2 / 0.832"],
    ["Pima (LR)", "1.0 / 0.574", "2.1 / 0.602", "2.7 / 0.609", "4.6 / 0.616"],
], [4.3, 2.8, 2.8, 2.8, 2.8], "Mean selected features / F1 for different α (20 runs each)")
figure("alpha_sensitivity_breast_cancer.png", 10.5,
       "Effect of α on F1, ROC-AUC and the number of selected features on WDBC")
para("Lowering α gives much smaller subsets at a modest cost. On WDBC, α = 0.90 keeps F1 at "
     "0.953 with about 7 of 30 features (77 % reduction). On ILPD a single feature gives the same "
     "F1 as four, but a much lower ROC-AUC (0.61 vs 0.72), showing that F1 alone can be "
     "misleading on imbalanced data.")
h2("7.7 Discussion")
para("The results answer the research question in two parts. First, BPSO does produce "
     "substantially smaller models: it removed 38–58 % of features on every dataset with no "
     "statistically significant loss in F1. For clinical use, where each feature may correspond "
     "to a laboratory test, this can lower data-collection cost and make models easier to "
     "inspect. Second, BPSO did not improve predictive performance: the full feature set and "
     "simple cross-validated filters ranked higher overall, and ROC-AUC was slightly but "
     "significantly lower on three datasets.")
para("Several factors explain this. The datasets are low-dimensional (8–30 features) and mostly "
     "contain informative clinical variables, so removing features mainly loses information "
     "rather than noise. The fitness function optimises inner accuracy with logistic regression, "
     "while evaluation uses F1 and ROC-AUC with five classifiers; this mismatch probably explains "
     "the ROC-AUC losses on the imbalanced ILPD and Pima datasets. Stochastic search variability "
     "is also visible in the moderate stability. Where BPSO helped — on the highly correlated "
     "WDBC features — it chose complementary features that outperformed same-size filter subsets. "
     "Finally, BPSO and the genetic algorithm gave practically identical results, suggesting that "
     "the fitness design matters more than the choice of metaheuristic.")

# ================================================================ CH 8
chapter("Chapter 8: Conclusion and Future Work")
h2("8.1 Conclusion")
para("This project applied Binary Particle Swarm Optimization to feature selection for disease "
     "prediction and evaluated it on five public datasets against nine alternatives, five "
     "classifiers and 50 repeated cross-validation folds per dataset. BPSO reduced the number of "
     "features by 38–58 % with a small, mostly non-significant loss in F1-score and selected "
     "clinically plausible features. However, it did not outperform the full feature set or simple "
     "cross-validated filters, and it ranked similarly to a genetic algorithm. On low-dimensional "
     "clinical data, BPSO is therefore best viewed as a tool for building compact and "
     "interpretable models rather than for improving accuracy. The complete system — research "
     "code, results, prediction API and dashboard — is publicly available and deployed online.")
h2("8.2 Limitations")
bullets([
    "Five small, public, low-dimensional datasets; results may differ on high-dimensional data.",
    "A reduced search budget (20 particles, 50 iterations, 5 repetitions).",
    "Simple median and mode imputation for missing values.",
    "The headline classifier per dataset was chosen by BPSO’s best ROC-AUC, which slightly favours "
    "BPSO in Table 12; the Friedman analysis over all classifiers does not have this bias.",
    "No external validation cohort; the models are not clinically validated.",
])
h2("8.3 Future Work")
bullets([
    "Optimise ROC-AUC or balanced accuracy in the fitness function.",
    "Evaluate V-shaped transfer functions and multi-objective (Pareto) BPSO.",
    "Add selection stability or the cost of each medical test to the objective.",
    "Test on high-dimensional datasets such as gene expression.",
    "Validate the workflow with external clinical data.",
])

# ================================================================ REFERENCES
chapter("References")
refs = [
    "J. Kennedy and R. Eberhart, “Particle swarm optimization,” in Proc. IEEE Int. Conf. Neural "
    "Networks, vol. 4, 1995, pp. 1942–1948.",
    "Y. Shi and R. Eberhart, “A modified particle swarm optimizer,” in Proc. IEEE Int. Conf. "
    "Evolutionary Computation, 1998, pp. 69–73.",
    "J. Kennedy and R. C. Eberhart, “A discrete binary version of the particle swarm algorithm,” "
    "in Proc. IEEE Int. Conf. Systems, Man, and Cybernetics, 1997, pp. 4104–4108.",
    "I. Guyon and A. Elisseeff, “An introduction to variable and feature selection,” Journal of "
    "Machine Learning Research, vol. 3, pp. 1157–1182, 2003.",
    "R. Kohavi and G. H. John, “Wrappers for feature subset selection,” Artificial Intelligence, "
    "vol. 97, no. 1–2, pp. 273–324, 1997.",
    "R. Tibshirani, “Regression shrinkage and selection via the lasso,” Journal of the Royal "
    "Statistical Society: Series B, vol. 58, no. 1, pp. 267–288, 1996.",
    "I. Guyon, J. Weston, S. Barnhill and V. Vapnik, “Gene selection for cancer classification "
    "using support vector machines,” Machine Learning, vol. 46, pp. 389–422, 2002.",
    "B. Xue, M. Zhang, W. N. Browne and X. Yao, “A survey on evolutionary computation approaches "
    "to feature selection,” IEEE Trans. Evolutionary Computation, vol. 20, no. 4, pp. 606–626, "
    "2016.",
    "L.-Y. Chuang, H.-W. Chang, C.-J. Tu and C.-H. Yang, “Improved binary PSO for feature "
    "selection using gene expression data,” Computational Biology and Chemistry, vol. 32, no. 1, "
    "pp. 29–38, 2008.",
    "M. N. Elbedwehy, H. M. Zawbaa, N. Ghali and A. E. Hassanien, “Detection of heart disease "
    "using binary particle swarm optimization,” in Proc. FedCSIS, 2012, pp. 177–182.",
    "M. S. Pathan, A. Nag, M. M. Pathan and S. Dev, “Analyzing the impact of feature selection on "
    "the accuracy of heart disease prediction,” Healthcare Analytics, vol. 2, art. 100060, 2022.",
    "Y. A. Z. A. Fajri, W. Wiharto and E. Suryani, “Hybrid model feature selection with the bee "
    "swarm optimization method and Q-learning on the diagnosis of coronary heart disease,” "
    "Information, vol. 14, no. 1, p. 15, 2023.",
    "Ö. H. Namlı, S. Yanık, A. Erdoğan and A. Schmeink, “A novel ensemble artificial intelligence "
    "approach for coronary artery disease prediction,” Int. J. Intelligent Computing and "
    "Cybernetics, vol. 17, no. 3, 2024.",
    "M. Zomorodi-Moghadam, M. Abdar, Z. Davarzani, X. Zhou, P. Pławiak and U. R. Acharya, "
    "“Hybrid particle swarm optimization for rule discovery in the diagnosis of coronary artery "
    "disease,” Expert Systems, vol. 38, art. e12485, 2021.",
    "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” Journal of "
    "Machine Learning Research, vol. 7, pp. 1–30, 2006.",
    "W. N. Street, W. H. Wolberg and O. L. Mangasarian, “Nuclear feature extraction for breast "
    "tumor diagnosis,” in Proc. SPIE, vol. 1905, 1993, pp. 861–870.",
    "L. Rubini, P. Soundarapandian and P. Eswaran, “Chronic Kidney Disease,” UCI Machine Learning "
    "Repository, 2015.",
    "A. Janosi, W. Steinbrunn, M. Pfisterer and R. Detrano, “Heart Disease,” UCI Machine Learning "
    "Repository, 1988, doi: 10.24432/C52P4X.",
    "B. Ramana and N. Venkateswarlu, “ILPD (Indian Liver Patient Dataset),” UCI Machine Learning "
    "Repository, 2012.",
    "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and R. S. Johannes, “Using the ADAP "
    "learning algorithm to forecast the onset of diabetes mellitus,” in Proc. Annu. Symp. Comput. "
    "Appl. Med. Care, 1988, pp. 261–265.",
    "R. Detrano et al., “International application of a new probability algorithm for the "
    "diagnosis of coronary artery disease,” American Journal of Cardiology, vol. 64, no. 5, "
    "pp. 304–310, 1989.",
    "S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of "
    "Statistics, vol. 6, no. 2, pp. 65–70, 1979.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning "
    "Research, vol. 12, pp. 2825–2830, 2011.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-1.0)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.add_run(f"[{i}]\t{r}")

# ================================================================ APPENDIX
chapter("Appendix A: Running the Project")
para("Install Python 3.10 or later, open a terminal in the project folder and run:")
code("""
pip install -e ".[serve,dev]"

# run the full five-dataset experiment (about 1 hour)
python scripts/run_experiments.py --config configs/all_datasets.yaml --dataset all --n-jobs 2 --results-dir results_experiment

# regenerate tables and figures
python scripts/make_tables.py  --results-dir results_experiment --tables-dir paper/tables
python scripts/make_figures.py --results-dir results_experiment --figures-dir paper/figures

# train the demo model and start the API and dashboard
python scripts/train_model.py --config configs/reduced.yaml --dataset breast_cancer --output models/disease_model.joblib
python -m uvicorn bpso_fs.api:app --port 8000
python -m streamlit run app.py
""")
para("With Docker installed, docker compose up --build starts the API (port 8000) and dashboard "
     "(port 8501) in one command.")
h2("Appendix B: Example API Request")
code("""
POST https://bpso-api.onrender.com/predict
Content-Type: application/json

{ "records": [ { "mean_radius": 17.99, "mean_texture": 10.38, ... } ] }

Response:
{ "predictions": [ { "label": 1, "probability": 0.99999 } ] }
""")
para("Label 1 means malignant for the breast-cancer demo model. The demo model was fitted on the "
     "full dataset, so its predictions on those records demonstrate the system rather than "
     "measure accuracy.")

# ---------------------------------------------------------------- page numbers
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_field(footer, "PAGE", "1")
sec.different_first_page_header_footer = True

doc.core_properties.title = TITLE
doc.core_properties.author = ", ".join(TEAM)
doc.save(OUT)
print("wrote", OUT)
