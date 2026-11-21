"""Build the project presentation: python paper/presentation/build_deck.py"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
OUT = HERE / "BPSO_Disease_Prediction_Presentation.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
TEAL = RGBColor(0x0E, 0x8C, 0x84)
MINT = RGBColor(0xD6, 0xF1, 0xEE)
CORAL = RGBColor(0xE4, 0x57, 0x4F)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x5B, 0x67, 0x78)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF7, 0xF9)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, opts) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
            if o.get("sup"):
                r._r.get_or_add_rPr().set("baseline", "30000")
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=INK, space=10):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        parts = it if isinstance(it, list) else [(it, {})]
        first = True
        for t, o in parts:
            r = p.add_run()
            r.text = ("•  " if first else "") + t
            first = False
            r.font.name = BODY
            r.font.size = Pt(size)
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def circle_num(slide, x, y, d, label, fill=TEAL, color=WHITE, size=18):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, color
    return c


def content_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    bg(s, WHITE)
    if kicker:
        text(s, 0.7, 0.45, 11.5, 0.35, kicker.upper(), size=12, color=TEAL, bold=True)
    text(s, 0.7, 0.75, 12.0, 0.8, title, size=34, color=NAVY, bold=True, font=HEAD)
    text(s, 0.7, 7.0, 8, 0.3, "Swarm Intelligence for Feature Selection in Disease Prediction",
         size=10, color=MUTED)
    text(s, 11.6, 7.0, 1.0, 0.3, str(len(prs.slides)), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)


def style_chart(chart, colors, legend=False, maxv=None, fmt='0.0'):
    chart.font.name = BODY
    chart.font.size = Pt(12)
    chart.font.color.rgb = MUTED
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    if maxv is not None:
        va.maximum_scale = maxv
    ca = chart.category_axis
    ca.format.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    ca.tick_labels.font.size = Pt(12)
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(11)
    dl.font.color.rgb = INK
    for ser, col in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col


# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
for i in range(8):  # motif: a row of 0/1 "particle" bits
    on = i in (1, 4, 5, 6)
    circle_num(s, 0.7 + i * 0.62, 1.1, 0.48, "1" if on else "0",
               fill=TEAL if on else RGBColor(0x1C, 0x3A, 0x5E), size=16)
text(s, 0.7, 2.05, 11.5, 1.9, "Swarm Intelligence for Feature Selection in Disease Prediction",
     size=42, color=WHITE, bold=True, font=HEAD)
text(s, 0.7, 4.05, 11.5, 0.5, "Binary Particle Swarm Optimization (BPSO) on five clinical datasets",
     size=20, color=MINT, italic=True)
text(s, 0.7, 5.15, 7.5, 1.4, [
    "Shashank Kushwaha  ·  Anshika Srivastava  ·  Muskan Mishra  ·  Roushan Kumar",
    [("Guide: ", {"bold": True, "color": MINT}), ("Niharika Chaudhary, Assistant Professor", {})],
    "Department of CSE (AIML), IILM University, Greater Noida",
], size=15, color=WHITE)
s.notes_slide.notes_text_frame.text = (
    "Introduce the team and the title. One line: we use a swarm-intelligence algorithm, BPSO, to "
    "pick a small set of clinical measurements for disease prediction, and we test it fairly on "
    "five datasets.")

# ------------------------------------------------------------------ 2 problem
s = content_slide("Not every clinical measurement is useful", "Problem and motivation")
cards = [
    ("Too many inputs", "Datasets hold 8–30 measurements; some are redundant or noisy."),
    ("Cost per feature", "Each measurement can mean a lab test, time and money for the patient."),
    ("Harder to explain", "Doctors trust models they can read; fewer inputs are easier to inspect."),
]
for i, (h, b) in enumerate(cards):
    x = 0.7 + i * 4.05
    box(s, x, 1.95, 3.75, 2.55, LIGHT)
    circle_num(s, x + 0.3, 2.2, 0.62, str(i + 1))
    text(s, x + 0.3, 3.0, 3.2, 0.45, h, size=20, color=NAVY, bold=True, font=HEAD)
    text(s, x + 0.3, 3.5, 3.2, 0.95, b, size=15, color=INK)
box(s, 0.7, 4.9, 11.9, 1.55, MINT)
text(s, 1.0, 5.05, 11.3, 1.25, [
    [("Research question  ", {"bold": True, "color": TEAL, "size": 15})],
    [("Can BPSO find a small set of features that predicts disease almost as well as using "
      "all features, and how does it compare with standard selection methods?",
      {"size": 19, "color": NAVY})],
], size=19)
s.notes_slide.notes_text_frame.text = (
    "Explain why feature selection matters in medicine: fewer tests, faster models, easier "
    "interpretation. Then read the research question.")

# ------------------------------------------------------------------ 3 objectives
s = content_slide("Project objectives", "What we set out to do")
objs = [
    ("Implement BPSO", "A binary swarm that searches feature subsets with a size-penalised fitness"),
    ("Test it fairly", "Nested cross-validation so test data never influences selection"),
    ("Compare widely", "Nine alternatives, five classifiers, five disease datasets"),
    ("Analyse honestly", "Significance tests, ranking, stability, runtime and α sensitivity"),
    ("Deploy it", "A live prediction API and web dashboard anyone can try"),
]
for i, (h, b) in enumerate(objs):
    y = 1.85 + i * 0.98
    circle_num(s, 0.8, y, 0.62, str(i + 1))
    text(s, 1.65, y - 0.02, 3.2, 0.62, h, size=20, color=NAVY, bold=True, font=HEAD,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 4.9, y - 0.02, 7.7, 0.62, b, size=17, color=INK, anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = "Walk through the five objectives briefly."

# ------------------------------------------------------------------ 4 feature selection
s = content_slide("Three ways to select features", "Background")
fams = [
    ("Filter", "Scores each feature alone with a statistic and keeps the top k. Very fast.",
     "MI, ANOVA-F"),
    ("Wrapper", "Trains a model on whole subsets. Slower, but finds features that work together.",
     "BPSO (ours), GA, RFE"),
    ("Embedded", "Selection happens while the model trains.", "L1 (Lasso) regression"),
]
for i, (h, b, ex) in enumerate(fams):
    x = 0.7 + i * 4.05
    hl = i == 1
    box(s, x, 1.95, 3.75, 3.3, NAVY if hl else LIGHT)
    text(s, x + 0.35, 2.25, 3.1, 0.55, h, size=26, color=WHITE if hl else NAVY, bold=True,
         font=HEAD)
    text(s, x + 0.35, 2.95, 3.1, 1.4, b, size=16, color=WHITE if hl else INK)
    text(s, x + 0.35, 4.45, 3.1, 0.6, [[("In this project: ", {"bold": True}), (ex, {})]],
         size=14, color=MINT if hl else TEAL)
text(s, 0.7, 5.65, 11.9, 0.9, [[
    ("Why search smartly?  ", {"bold": True, "color": CORAL}),
    ("30 features give 2", {}), ("30", {"sup": True}),
    (" ≈ 1 billion possible subsets — far too many to try one by one.", {}),
]], size=18, color=INK)
s.notes_slide.notes_text_frame.text = (
    "BPSO is a wrapper method: it judges a subset by actually training a model on it. That is "
    "why it can find features that work well together, and why it is slower than filters.")

# ------------------------------------------------------------------ 5 BPSO
s = content_slide("How Binary PSO searches", "Method")
text(s, 0.7, 1.75, 6.0, 2.2, [
    "A flock of birds looks for food. Each bird remembers its own best spot (personal best) and "
    "knows the flock's best spot (global best), and moves partly toward both.",
    "",
    "In BPSO each bird is a row of bits: 1 = use the feature, 0 = drop it.",
], size=16, color=INK)
names = ["Preg", "Gluc", "BP", "Skin", "Ins", "BMI", "Ped", "Age"]
bits = [0, 1, 0, 0, 1, 1, 1, 0]
for i, (n, b) in enumerate(zip(names, bits)):
    x = 0.7 + i * 0.74
    circle_num(s, x, 4.35, 0.6, str(b), fill=TEAL if b else RGBColor(0xCB, 0xD5, 0xE1),
               color=WHITE if b else INK, size=18)
    text(s, x - 0.1, 5.02, 0.8, 0.3, n, size=11, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 0.7, 5.5, 6.0, 0.8, "This particle uses Glucose, Insulin, BMI and Pedigree (Pima dataset).",
     size=14, color=MUTED, italic=True)
box(s, 7.25, 1.75, 5.4, 4.6, LIGHT)
text(s, 7.6, 2.0, 4.8, 0.5, "Each iteration", size=20, color=NAVY, bold=True, font=HEAD)
steps = [
    "Update velocity:  v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)",
    "Turn velocity into a probability:  S(v) = 1 / (1 + e^(−v))",
    "Set each bit to 1 with probability S(v)",
    "Score the subset; update personal and global bests",
]
for i, st in enumerate(steps):
    y = 2.65 + i * 0.9
    circle_num(s, 7.6, y, 0.46, str(i + 1), size=14)
    text(s, 8.25, y - 0.04, 4.15, 0.85, st, size=14, color=INK)
s.notes_slide.notes_text_frame.text = (
    "Use the bird analogy. Then show the bit row: this is one candidate subset. Velocity is "
    "turned into a probability with the sigmoid, so a large positive velocity means the feature "
    "is very likely kept. Settings: 20 particles, 50 iterations, inertia 0.9 to 0.4, c1 = c2 = 2.")

# ------------------------------------------------------------------ 6 fitness
s = content_slide("Scoring a subset: the fitness function", "Method")
box(s, 0.7, 1.85, 11.9, 1.25, NAVY)
text(s, 0.7, 1.85, 11.9, 1.25,
     "F(z) = α · (1 − accuracy) + (1 − α) · (selected ÷ total features)",
     size=28, color=WHITE, bold=True, font=HEAD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
parts = [
    ("Lower is better", "BPSO minimises F. Good accuracy and few features both lower it."),
    ("accuracy", "5-fold cross-validated accuracy of Logistic Regression, on training data only."),
    ("α = 0.99", "Accuracy dominates; size breaks near-ties. Lower α → smaller subsets."),
]
for i, (h, b) in enumerate(parts):
    x = 0.7 + i * 4.05
    text(s, x, 3.45, 3.75, 0.45, h, size=20, color=TEAL, bold=True, font=HEAD)
    text(s, x, 3.95, 3.75, 1.2, b, size=16, color=INK)
box(s, 0.7, 5.35, 11.9, 1.1, MINT)
text(s, 1.0, 5.35, 11.3, 1.1, [[
    ("Example  ", {"bold": True, "color": TEAL}),
    ("accuracy 0.80 with 4 of 8 features  →  F = 0.99 × 0.20 + 0.01 × 0.5 = 0.203", {}),
]], size=18, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = (
    "The fitness balances two goals. With alpha 0.99, accuracy matters most and the size term "
    "only breaks ties. We also tested smaller alpha values later.")

# ------------------------------------------------------------------ 7 datasets
s = content_slide("Five public disease datasets", "Data")
rows = [
    ("Dataset", "Patients", "Features", "Disease %", "Predicts"),
    ("Breast Cancer (WDBC)", "569", "30", "37.3", "Malignant tumour"),
    ("Chronic Kidney Disease", "400", "24", "62.5", "CKD"),
    ("Heart Disease (Cleveland)", "303", "13", "45.9", "Heart disease"),
    ("Indian Liver Patient", "583", "10", "71.4", "Liver disease"),
    ("Pima Indians Diabetes", "768", "8", "34.9", "Diabetes"),
]
widths = [3.9, 1.5, 1.5, 1.6, 3.4]
tbl = s.shapes.add_table(len(rows), 5, Inches(0.7), Inches(1.85), Inches(11.9),
                         Inches(0.55 * len(rows))).table
for c, wd in enumerate(widths):
    tbl.columns[c].width = Inches(wd)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c in (0, 4) else PP_ALIGN.CENTER
        f = para.runs[0].font
        f.name, f.size = BODY, Pt(16)
        f.bold = r == 0
        f.color.rgb = WHITE if r == 0 else INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL if r == 0 else (LIGHT if r % 2 else WHITE)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
text(s, 0.7, 5.5, 11.9, 1.0, [
    "Missing values filled with the training-fold median or mode  ·  Pima's impossible zeros "
    "treated as missing  ·  Heart label 0–4 turned into yes/no  ·  all features standardised",
], size=15, color=MUTED)
s.notes_slide.notes_text_frame.text = (
    "All datasets are public benchmarks, mostly from UCI. Label 1 always means disease. Note "
    "the different sizes and that ILPD is imbalanced (71 percent positive).")

# ------------------------------------------------------------------ 8 pipeline
s = content_slide("A leakage-free evaluation pipeline", "Methodology")
stages = ["Dataset", "Split into 10 folds", "Clean data\n(train fold only)", "BPSO selects\nfeatures",
          "Train and tune\nclassifier", "Test on held-out\nfold"]
bw, gap = 1.75, 0.28
for i, st in enumerate(stages):
    x = 0.7 + i * (bw + gap)
    hl = i == 3
    b = box(s, x, 2.15, bw, 1.45, NAVY if hl else MINT)
    tf = b.text_frame
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for j, line in enumerate(st.split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size, r.font.bold = BODY, Pt(15), True
        r.font.color.rgb = WHITE if hl else NAVY
    if i < len(stages) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.04), Inches(2.72),
                               Inches(gap - 0.08), Inches(0.3))
        a.fill.solid()
        a.fill.fore_color.rgb = TEAL
        a.line.fill.background()
box(s, 0.7, 4.1, 11.9, 2.3, LIGHT)
bullets(s, 1.0, 4.35, 11.3, 2.0, [
    [("Repeated 50 times per dataset: ", {"bold": True}), ("10 folds × 5 random seeds", {})],
    [("No leakage: ", {"bold": True}),
     ("imputation, scaling, selection and tuning all use the training fold only", {})],
    [("Same folds for every method, ", {"bold": True}), ("so comparisons are paired", {})],
], size=17)
s.notes_slide.notes_text_frame.text = (
    "This is nested cross-validation. The key point for evaluators: the test fold is used only "
    "once, at the end, so the results are not optimistic.")

# ------------------------------------------------------------------ 9 setup
s = content_slide("Experiment at a glance", "Experimental setup")
stats = [("5", "datasets"), ("10", "selection methods"), ("5", "classifiers"),
         ("12,500", "test evaluations")]
for i, (n, lab) in enumerate(stats):
    x = 0.7 + i * 3.0
    box(s, x, 1.85, 2.75, 2.0, NAVY if i == 3 else LIGHT)
    text(s, x, 2.0, 2.75, 1.1, n, size=48, color=WHITE if i == 3 else TEAL, bold=True,
         font=HEAD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, 3.1, 2.75, 0.5, lab, size=16, color=MINT if i == 3 else INK,
         align=PP_ALIGN.CENTER)
text(s, 0.7, 4.25, 5.8, 0.45, "Compared against", size=20, color=NAVY, bold=True, font=HEAD)
bullets(s, 0.7, 4.8, 5.8, 2.0, [
    "All features · Binary Genetic Algorithm",
    "MI and ANOVA-F filters (k = BPSO, k by CV)",
    "RFE · L1 Logistic Regression · Random subsets",
], size=16, space=6)
text(s, 6.9, 4.25, 5.7, 0.45, "Classifiers", size=20, color=NAVY, bold=True, font=HEAD)
bullets(s, 6.9, 4.8, 5.7, 2.0, [
    "Logistic Regression · SVM (RBF) · Random Forest",
    "k-Nearest Neighbours · Gaussian Naive Bayes",
    "Tuned by inner grid-search cross-validation",
], size=16, space=6)
s.notes_slide.notes_text_frame.text = (
    "5 datasets times 10 methods times 5 classifiers times 50 folds gives 12,500 test "
    "evaluations, about 51 minutes of compute. Statistics: Wilcoxon with Holm correction, "
    "Friedman with Nemenyi.")

# ------------------------------------------------------------------ 10 reduction chart
s = content_slide("BPSO cut 38–58 % of features on every dataset", "Result 1")
cd = CategoryChartData()
cd.categories = ["Breast Cancer", "Kidney", "Heart", "Liver", "Diabetes"]
cd.add_series("Feature reduction (%)", (51.1, 49.1, 38.2, 57.8, 39.8))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.75), Inches(7.4),
                        Inches(4.9), cd)
style_chart(gf.chart, [TEAL], maxv=70)
gf.chart.has_title = True
gf.chart.chart_title.text_frame.text = "Features removed by BPSO (%)"
gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
box(s, 8.5, 1.85, 4.1, 4.7, LIGHT)
text(s, 8.8, 2.1, 3.5, 0.45, "Features kept", size=20, color=NAVY, bold=True, font=HEAD)
kept = [("Breast Cancer", "30 → 14.7"), ("Kidney", "24 → 12.2"), ("Heart", "13 → 8.0"),
        ("Liver", "10 → 4.2"), ("Diabetes", "8 → 4.8")]
for i, (d, k) in enumerate(kept):
    y = 2.75 + i * 0.6
    text(s, 8.8, y, 1.9, 0.45, d, size=16, color=INK)
    text(s, 10.7, y, 1.6, 0.45, k, size=16, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)
text(s, 8.8, 5.85, 3.5, 0.6, "Average over 50 test folds", size=12, color=MUTED, italic=True)
s.notes_slide.notes_text_frame.text = (
    "The main positive result: BPSO roughly halves the number of features on every dataset.")

# ------------------------------------------------------------------ 11 accuracy chart
s = content_slide("F1-score stays close to the full model", "Result 2")
cd = CategoryChartData()
cd.categories = ["Breast Cancer", "Kidney", "Heart", "Liver", "Diabetes"]
cd.add_series("All features", (0.963, 0.994, 0.820, 0.826, 0.626))
cd.add_series("BPSO features", (0.956, 0.996, 0.805, 0.830, 0.614))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.75), Inches(7.6),
                        Inches(4.9), cd)
style_chart(gf.chart, [RGBColor(0x94, 0xA3, 0xB8), TEAL], legend=True, maxv=1.1, fmt="0.000")
gf.chart.plots[0].data_labels.font.size = Pt(10)
gf.chart.value_axis.minimum_scale = 0
gf.chart.value_axis.tick_labels.number_format = "0.0"
gf.chart.value_axis.tick_labels.number_format_is_linked = False
gf.chart.value_axis.major_unit = 0.2
box(s, 8.7, 1.85, 3.9, 4.7, LIGHT)
bullets(s, 9.0, 2.15, 3.35, 4.3, [
    [("F1-score", {"bold": True}), (" with the best classifier per dataset", {})],
    [("Change: −0.015 to +0.004", {"bold": True, "color": TEAL})],
    "No F1 difference is statistically significant",
    "ROC-AUC was slightly but significantly lower on 3 datasets",
], size=15, space=12)
s.notes_slide.notes_text_frame.text = (
    "F1 with the reported classifier: SVM for breast cancer, random forest for kidney, logistic "
    "regression for the rest. Wilcoxon tests with Holm correction show no significant F1 "
    "difference.")

# ------------------------------------------------------------------ 12 ranking
s = content_slide("But BPSO did not beat simpler methods", "Result 3")
picture(s, "cd_diagram.png", 0.7, 1.95, w=8.2)
box(s, 9.25, 1.85, 3.35, 4.7, NAVY)
text(s, 9.55, 2.1, 2.8, 1.0, "Rank 7.20", size=36, color=WHITE, bold=True, font=HEAD)
text(s, 9.55, 2.95, 2.8, 0.4, "of 10 methods (1 = best)", size=14, color=MINT)
bullets(s, 9.55, 3.6, 2.8, 2.9, [
    "Tied with the Genetic Algorithm (7.28)",
    "Behind all features (3.68) and CV-tuned filters (3.8–3.9)",
    "Friedman p < 0.001",
], size=14, color=WHITE, space=10)
text(s, 0.7, 5.5, 8.2, 0.9,
     "Average Friedman rank over 25 dataset–classifier settings (F1). A bar joins methods that "
     "are not significantly different (Nemenyi CD = 2.71).", size=13, color=MUTED, italic=True)
s.notes_slide.notes_text_frame.text = (
    "Be honest here: BPSO ranks 8th of 10. On low-dimensional data, keeping all features or using "
    "a CV-tuned filter is as good or better. This is a valid research finding.")

# ------------------------------------------------------------------ 13 features
s = content_slide("BPSO picks clinically sensible features", "Result 4")
picture(s, "feature_frequency_pima_diabetes.png", 0.7, 1.8, w=6.6)
text(s, 0.7, 6.25, 6.6, 0.4, "Pima diabetes: how often each feature was chosen in 50 runs",
     size=12, color=MUTED, italic=True)
items = [
    ("Diabetes", "Glucose, BMI (every run)"),
    ("Kidney", "Specific gravity, haemoglobin"),
    ("Heart", "Major vessels, thalassemia"),
    ("Breast", "Worst smoothness, worst symmetry"),
    ("Liver", "Age, SGPT (ALT)"),
]
for i, (d, f) in enumerate(items):
    y = 1.9 + i * 0.9
    box(s, 7.75, y, 4.85, 0.75, LIGHT)
    text(s, 8.0, y, 1.35, 0.75, d, size=16, color=TEAL, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.35, y, 3.15, 0.75, f, size=15, color=INK, anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = (
    "The selected features match known clinical markers, which supports interpretability. On "
    "breast cancer, BPSO's 15 features also beat the top 15 filter features (p < 0.001).")

# ------------------------------------------------------------------ 14 alpha
s = content_slide("α controls the size–accuracy trade-off", "Result 5")
picture(s, "alpha_sensitivity_breast_cancer.png", 0.7, 1.75, h=4.85)
box(s, 6.6, 1.85, 6.0, 4.7, LIGHT)
text(s, 6.95, 2.1, 5.3, 0.45, "Breast cancer (SVM)", size=20, color=NAVY, bold=True, font=HEAD)
rows = [("α", "Features", "F1"), ("0.80", "5.0", "0.942"), ("0.90", "6.8", "0.953"),
        ("0.95", "9.9", "0.960"), ("0.99", "14.3", "0.961")]
for i, row in enumerate(rows):
    y = 2.7 + i * 0.5
    for j, v in enumerate(row):
        text(s, 6.95 + j * 1.7, y, 1.6, 0.45, v, size=16, color=NAVY if i == 0 else INK,
             bold=i == 0 or i == 2)
text(s, 6.95, 5.3, 5.3, 1.1, "α = 0.90 keeps F1 at 0.953 with only about 7 of 30 features "
     "(77 % fewer).", size=16, color=TEAL, bold=True)
s.notes_slide.notes_text_frame.text = (
    "Lowering alpha gives much smaller models at a small cost in F1. This is the main tuning "
    "knob for practitioners.")

# ------------------------------------------------------------------ 15 system
s = content_slide("A working system you can try", "Implementation")
comps = [
    ("Research code", "Python, scikit-learn: BPSO, baselines, nested CV, statistics, figures"),
    ("Prediction API", "FastAPI service: /health, /metadata, /predict"),
    ("Web dashboard", "Streamlit: results explorer and CSV prediction upload"),
    ("Deployment", "Docker on Render, code on GitHub with automated tests"),
]
for i, (h, b) in enumerate(comps):
    x = 0.7 + (i % 2) * 6.05
    y = 1.85 + (i // 2) * 1.75
    box(s, x, y, 5.85, 1.5, LIGHT)
    circle_num(s, x + 0.3, y + 0.42, 0.66, str(i + 1))
    text(s, x + 1.2, y + 0.25, 4.4, 0.45, h, size=19, color=NAVY, bold=True, font=HEAD)
    text(s, x + 1.2, y + 0.72, 4.45, 0.7, b, size=14, color=INK)
box(s, 0.7, 5.5, 11.9, 1.0, NAVY)
text(s, 1.0, 5.5, 11.3, 1.0, [[
    ("Live demo  ", {"bold": True, "color": MINT}),
    ("bpso-showcase.onrender.com   ·   API: bpso-api.onrender.com/docs", {"color": WHITE}),
]], size=18, anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = (
    "Switch to the browser here: show the Evaluation results tab, then upload demo_patients.csv "
    "in Try a prediction. Say clearly that the demo model was trained on the full breast-cancer "
    "data, so predictions on those rows only show the system works.")

# ------------------------------------------------------------------ 16 conclusion
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, 0.7, 0.75, 12, 0.8, "Conclusion and future work", size=34, color=WHITE, bold=True,
     font=HEAD)
box(s, 0.7, 1.85, 5.8, 4.55, RGBColor(0x13, 0x33, 0x5C))
text(s, 1.0, 2.1, 5.2, 0.45, "What we found", size=20, color=MINT, bold=True, font=HEAD)
bullets(s, 1.0, 2.7, 5.25, 3.6, [
    "BPSO halves the features with almost no F1 loss",
    "It chooses medically sensible features",
    "It does not improve accuracy on these small datasets",
    "Best use: compact, cheaper, interpretable models",
], size=16, color=WHITE, space=12)
box(s, 6.8, 1.85, 5.8, 4.55, RGBColor(0x13, 0x33, 0x5C))
text(s, 7.1, 2.1, 5.2, 0.45, "Next steps", size=20, color=MINT, bold=True, font=HEAD)
bullets(s, 7.1, 2.7, 5.25, 3.6, [
    "Optimise ROC-AUC or balanced accuracy in the fitness",
    "Multi-objective (Pareto) and V-shaped BPSO",
    "High-dimensional data such as gene expression",
    "Include the cost of each medical test",
], size=16, color=WHITE, space=12)
text(s, 0.7, 6.75, 12, 0.4, "Research use only — models are not clinically validated.",
     size=12, color=MINT, italic=True)
s.notes_slide.notes_text_frame.text = (
    "Summarise: useful for reducing features, not for increasing accuracy, and we report this "
    "honestly with statistical tests.")

# ------------------------------------------------------------------ 17 thanks
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
for i in range(8):
    on = i in (1, 4, 5, 6)
    circle_num(s, 4.27 + i * 0.62, 2.0, 0.48, "1" if on else "0",
               fill=TEAL if on else RGBColor(0x1C, 0x3A, 0x5E), size=16)
text(s, 0.7, 2.85, 11.9, 1.0, "Thank you", size=48, color=WHITE, bold=True, font=HEAD,
     align=PP_ALIGN.CENTER)
text(s, 0.7, 3.95, 11.9, 0.5, "Questions?", size=24, color=MINT, align=PP_ALIGN.CENTER,
     italic=True)
text(s, 0.7, 5.2, 11.9, 0.9, [
    "github.com/shashank-nova/bpso-disease-prediction",
    "bpso-showcase.onrender.com",
], size=16, color=WHITE, align=PP_ALIGN.CENTER)

prs.core_properties.title = "Swarm Intelligence for Feature Selection in Disease Prediction"
prs.core_properties.author = "Kushwaha, Srivastava, Mishra, Kumar"
prs.save(OUT)
print("wrote", OUT, len(prs.slides), "slides")
