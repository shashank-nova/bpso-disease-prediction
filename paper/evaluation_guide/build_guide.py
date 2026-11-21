"""Build the evaluation guide PDF: python paper/evaluation_guide/build_guide.py"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
OUT = HERE / "BPSO_Project_Evaluation_Guide.pdf"

# Arial covers Greek letters, ±, → and ✓ (the built-in PDF fonts do not).
FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")

INK = colors.HexColor("#1f2937")
ACCENT = colors.HexColor("#1d4ed8")
SOFT = colors.HexColor("#eef2ff")
NOTE = colors.HexColor("#fff7ed")
NOTE_EDGE = colors.HexColor("#f59e0b")
GRID = colors.HexColor("#cbd5e1")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Body-Bold", fontSize=24,
                            leading=29, textColor=INK, spaceAfter=8),
    "sub": ParagraphStyle("s", fontName="Body", fontSize=12.5, leading=17,
                          textColor=colors.HexColor("#475569"), alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=21, textColor=ACCENT,
                         spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Body", fontSize=10.5, leading=15, textColor=INK,
                        spaceAfter=6),
    "b": ParagraphStyle("b", fontName="Body", fontSize=10.5, leading=15, textColor=INK,
                        leftIndent=14, bulletIndent=3, spaceAfter=3),
    "cell": ParagraphStyle("c", fontName="Body", fontSize=9.3, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Body-Bold", fontSize=9.3, leading=12,
                            textColor=colors.white),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=9, leading=12,
                          textColor=colors.HexColor("#475569"), alignment=TA_CENTER,
                          spaceAfter=10),
    "say": ParagraphStyle("say", fontName="Body-Italic", fontSize=10.5, leading=15,
                          textColor=colors.HexColor("#1e3a8a")),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=9, leading=12, textColor=INK),
    "box": ParagraphStyle("box", fontName="Body-Bold", fontSize=9.5, leading=12,
                          textColor=INK, alignment=TA_CENTER),
}

story = []


def h1(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def p(t):
    story.append(Paragraph(t, S["p"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["b"], bulletText="•"))
    story.append(Spacer(1, 4))


def steps(items):
    for i, it in enumerate(items, 1):
        story.append(Paragraph(it, S["b"], bulletText=f"{i}."))
    story.append(Spacer(1, 4))


def boxed(flowables, bg, edge):
    t = Table([[flowables]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def say(t):
    """A 'what to say to your guide' callout."""
    boxed([Paragraph("<b>Say to your guide:</b> " + t, S["say"])], SOFT, ACCENT)


def note(t):
    boxed([Paragraph(t, S["p"])], NOTE, NOTE_EDGE)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb"] if header and r == 0 else S["cell"]) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc"))
                  for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


def figure(name, width_cm, caption):
    path = FIG / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    story.append(KeepTogether([img, Spacer(1, 3), Paragraph(caption, S["cap"])]))


def pipeline():
    """Simple flow diagram built from table cells."""
    labels = ["1. Dataset\n(CSV / UCI)", "2. Split into\n10 folds", "3. Clean data\n(train fold only)",
              "4. BPSO picks\nfeatures", "5. Train\nclassifier", "6. Test on\nheld-out fold"]
    cells, widths = [], []
    for i, lab in enumerate(labels):
        cells.append(Paragraph(lab.replace("\n", "<br/>"), S["box"]))
        widths.append(2.35 * cm)
        if i < len(labels) - 1:
            cells.append(Paragraph("→", S["box"]))
            widths.append(0.45 * cm)
    t = Table([cells], colWidths=widths, rowHeights=[1.5 * cm])
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]
    for c in range(0, len(cells), 2):
        style += [("BACKGROUND", (c, 0), (c, 0), SOFT), ("BOX", (c, 0), (c, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Steps 2–6 repeat 50 times per dataset (10 folds × 5 random seeds). "
                           "The test fold is never seen during steps 3–5.", S["cap"]))


# ---------------------------------------------------------------- cover
story += [Spacer(1, 3.2 * cm),
          Paragraph("Swarm Intelligence for Feature Selection in Disease Prediction", S["title"]),
          Spacer(1, 6),
          Paragraph("Project explanation and evaluation guide", S["sub"]),
          Spacer(1, 1.2 * cm)]
table([
    ["Team", "Shashank Kushwaha, Anshika Srivastava, Muskan Mishra, Roushan Kumar"],
    ["Guide", "Niharika Chaudhary, Assistant Professor"],
    ["Institute", "Dept. of CSE (AIML), IILM University, Greater Noida"],
    ["Live dashboard", "https://bpso-showcase.onrender.com"],
    ["Live API", "https://bpso-api.onrender.com/docs"],
    ["Source code", "https://github.com/shashank-nova/bpso-disease-prediction"],
], [3.6, 12.8], header=False)
story.append(Spacer(1, 0.8 * cm))
p("<b>How to use this document.</b> Sections 1–3 explain the project from the basics. "
  "Sections 4–6 cover what was built and what was found. Section 7 is a step-by-step script for "
  "showing the project to your guide, and Section 8 lists questions you are likely to be asked, "
  "with answers. Blue boxes give sentences you can say directly; orange boxes are warnings.")
story.append(PageBreak())

# ---------------------------------------------------------------- 1
h1("1. The project in one minute")
p("Doctors decide whether a patient has a disease by looking at measurements such as blood "
  "glucose, blood pressure or tumour size. A machine-learning model can learn this decision from "
  "past patient records. But datasets often contain many measurements, and not all of them are "
  "useful. Some are redundant, some are noise, and each one may cost a lab test.")
p("<b>Our question:</b> can we automatically find a <b>small set of measurements</b> that still "
  "predicts the disease well? We use a swarm-intelligence algorithm, <b>Binary Particle Swarm "
  "Optimization (BPSO)</b>, to search for that small set, and we test it fairly on five real "
  "disease datasets against nine other methods.")
p("<b>Our answer:</b> BPSO removed <b>38–58 % of the features</b> on every dataset while the "
  "F1-score dropped by at most about 0.015. But it did <b>not</b> make predictions more accurate "
  "than using all features or simpler statistical selection methods. Its value is smaller, "
  "cheaper, easier-to-explain models.")
say("“Our project uses Binary Particle Swarm Optimization to choose a small set of clinical "
    "features for disease prediction. On five public datasets it cut the features roughly in "
    "half with almost no loss in F1-score, but it did not beat simpler methods on accuracy, so "
    "its main benefit is compact, interpretable models.”")

# ---------------------------------------------------------------- 2
h1("2. Basic concepts (explained simply)")
h2("2.1 Classification")
p("A <b>classification</b> model takes a patient's measurements and outputs a class: "
  "<b>1 = disease</b>, <b>0 = no disease</b>. It learns from labelled examples (patients whose "
  "outcome is known). We use five standard classifiers: Logistic Regression, Support Vector "
  "Machine (SVM), Random Forest, k-Nearest Neighbours (k-NN) and Gaussian Naive Bayes.")
h2("2.2 Features")
p("A <b>feature</b> is one input column, such as Glucose or BMI. The Pima diabetes dataset has "
  "8 features; the breast-cancer dataset has 30.")
h2("2.3 Feature selection")
p("<b>Feature selection</b> means choosing a subset of the columns. Fewer features give faster, "
  "simpler and cheaper models. There are three families of methods:")
table([
    ["Family", "How it works", "In our project"],
    ["Filter", "Scores each feature on its own with a statistic, keeps the top k. Very fast.",
     "Mutual Information (MI), ANOVA-F"],
    ["Wrapper", "Tries whole subsets by actually training a model on them. Slower but can "
     "find features that work well together.", "BPSO (ours), Genetic Algorithm, RFE"],
    ["Embedded", "Selection happens while training the model.", "L1 (Lasso) Logistic Regression"],
], [2.4, 9.0, 5.0])
p("With d features there are 2<super>d</super> possible subsets. For 30 features that is over "
  "one billion, so we cannot try them all. We need a smart search, and that is where BPSO comes in.")

h2("2.4 Particle Swarm Optimization (PSO) — the bird-flock idea")
p("Imagine a flock of birds searching for food. Each bird remembers the best place it has found "
  "itself (<b>personal best</b>) and knows the best place any bird has found (<b>global best</b>). "
  "At every step each bird moves partly in its current direction, partly towards its own best "
  "spot and partly towards the flock's best spot. Over time the flock gathers around good areas.")
p("In <b>Binary PSO</b> each bird (“particle”) is a row of 0s and 1s, one bit per feature: "
  "1 = use the feature, 0 = drop it. Example for Pima (8 features):")
table([
    ["Pregnan- cies", "Glucose", "Blood pressure", "Skin thickness", "Insulin", "BMI", "Pedigree",
     "Age"],
    ["0", "1", "0", "0", "1", "1", "1", "0"],
], [2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05])
p("This particle means “use Glucose, Insulin, BMI and Pedigree”. Each bit also has a "
  "<b>velocity</b>. A sigmoid function turns velocity into a probability of the bit being 1, so "
  "a large positive velocity means “very likely keep this feature”.")
p("<b>Update rule</b> (for each particle i): new velocity = w × old velocity + "
  "c<sub>1</sub>·r<sub>1</sub>·(personal best − position) + c<sub>2</sub>·r<sub>2</sub>·(global "
  "best − position). Then each bit is set to 1 with probability sigmoid(velocity). "
  "w is the inertia weight (0.9 → 0.4), c<sub>1</sub> = c<sub>2</sub> = 2 and r<sub>1</sub>, "
  "r<sub>2</sub> are random numbers between 0 and 1.")

h2("2.5 The fitness function — how BPSO scores a subset")
p("Every subset gets a score called <b>fitness</b>; lower is better:")
boxed([Paragraph("<b>F(z) = α · (1 − accuracy) + (1 − α) · (selected features ÷ total features)</b>",
                 ParagraphStyle("f", parent=S["p"], alignment=TA_CENTER, fontSize=11.5))],
      SOFT, ACCENT)
bullets([
    "<b>accuracy</b> = accuracy of a Logistic Regression model on the chosen features, measured "
    "with 5-fold cross-validation on the training data only.",
    "<b>α = 0.99</b> means accuracy matters most; the size term only breaks near-ties. A smaller α "
    "(e.g. 0.90) gives much smaller subsets.",
    "Example: accuracy 0.80 with 4 of 8 features → F = 0.99 × 0.20 + 0.01 × 0.5 = 0.203.",
])
h2("2.6 Cross-validation and data leakage")
p("To judge a model fairly we must test it on patients it has never seen. <b>10-fold "
  "cross-validation</b> splits the data into 10 parts, trains on 9 and tests on 1, rotating 10 "
  "times. We repeat this with 5 different random shuffles, giving <b>50 test results per "
  "dataset</b>. “Stratified” means every fold keeps the same disease percentage.")
p("<b>Data leakage</b> happens when test data influences training, for example by choosing "
  "features using the full dataset. That makes results look better than they really are. In our "
  "project <b>everything</b> (filling missing values, scaling, feature selection, tuning) is done "
  "inside the training folds only. This is called <b>nested cross-validation</b>.")
h2("2.7 Evaluation metrics")
table([
    ["Metric", "Meaning", "Formula"],
    ["Accuracy", "Share of all predictions that are correct", "(TP + TN) / all"],
    ["Precision", "Of the patients predicted sick, how many really are", "TP / (TP + FP)"],
    ["Recall (sensitivity)", "Of the truly sick patients, how many we caught", "TP / (TP + FN)"],
    ["F1-score", "Balance of precision and recall (main metric)", "2·P·R / (P + R)"],
    ["ROC-AUC", "How well the model ranks sick above healthy (0.5 = random, 1 = perfect)",
     "Area under ROC curve"],
], [3.3, 8.6, 4.5])
p("TP = true positive, TN = true negative, FP = false positive, FN = false negative.")
story.append(PageBreak())

# ---------------------------------------------------------------- 3
h1("3. Datasets")
p("All five are public benchmark datasets (mostly from the UCI Machine Learning Repository). "
  "Label 1 always means the disease is present.")
table([
    ["Dataset", "Patients", "Features", "Disease %", "Missing", "Predicts"],
    ["Breast Cancer (WDBC)", "569", "30", "37.3", "0", "Malignant vs benign tumour"],
    ["Chronic Kidney Disease", "400", "24", "62.5", "1013", "CKD vs not CKD"],
    ["Heart Disease (Cleveland)", "303", "13", "45.9", "6", "Heart disease present"],
    ["Indian Liver Patient", "583", "10", "71.4", "4", "Liver disease"],
    ["Pima Indians Diabetes", "768", "8", "34.9", "652", "Diabetes onset"],
], [4.4, 1.8, 1.8, 1.9, 1.6, 4.9])
bullets([
    "<b>Missing values</b> are filled with the median (numbers) or most common value (categories), "
    "learned from the training fold only.",
    "<b>Pima:</b> impossible zeros (glucose = 0, BMI = 0, …) are treated as missing.",
    "<b>Heart:</b> the original 0–4 severity label is turned into 0 = no disease, 1 = disease.",
    "All features are standardised (mean 0, standard deviation 1).",
])

# ---------------------------------------------------------------- 4
h1("4. How the system works")
h2("4.1 The experiment pipeline")
pipeline()
h2("4.2 What we compared BPSO against")
table([
    ["Method", "What it does"],
    ["All features", "No selection — the reference point"],
    ["BPSO (ours)", "Swarm of 20 particles, 50 iterations, fitness above"],
    ["Binary GA", "Genetic algorithm with the same fitness (crossover + mutation)"],
    ["MI / ANOVA-F (k = BPSO)", "Top features by statistic, same count as BPSO picked"],
    ["MI / ANOVA-F (k by CV)", "Top features by statistic, count chosen by cross-validation"],
    ["RFE (k = BPSO)", "Repeatedly removes the weakest feature"],
    ["L1-LogReg", "Lasso regression keeps features with non-zero weight"],
    ["Random (k = BPSO)", "Random features, same count — a sanity check"],
], [5.0, 11.4])
p("In total: <b>5 datasets × 10 methods × 5 classifiers × 50 folds = 12,500 test evaluations</b>, "
  "about 51 minutes of computing.")
h2("4.3 Project structure (what is in the code)")
table([
    ["Folder / file", "Purpose"],
    ["src/bpso_fs/selectors/bpso.py", "The BPSO algorithm"],
    ["src/bpso_fs/fitness.py", "The fitness function"],
    ["src/bpso_fs/selectors/", "GA, filters, RFE, L1 and random baselines"],
    ["src/bpso_fs/experiment.py", "Nested cross-validation experiment"],
    ["src/bpso_fs/data/", "Dataset loading and preprocessing"],
    ["scripts/run_experiments.py", "Runs the full experiment"],
    ["scripts/make_tables.py, make_figures.py", "Generate paper tables and charts"],
    ["src/bpso_fs/api.py", "FastAPI prediction service"],
    ["app.py", "Streamlit web dashboard"],
    ["results_experiment/", "Raw results of every fold"],
    ["paper/", "Figures, tables, LaTeX paper, this guide"],
    ["tests/", "Automated tests (pytest)"],
], [6.6, 9.8])
h2("4.4 Technology stack")
bullets([
    "<b>Language:</b> Python 3.11",
    "<b>Machine learning:</b> scikit-learn, NumPy, pandas, SciPy",
    "<b>Charts:</b> matplotlib, seaborn",
    "<b>Web:</b> FastAPI (prediction API) and Streamlit (dashboard)",
    "<b>Deployment:</b> Docker, hosted on Render; code on GitHub with GitHub Actions tests",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 5
h1("5. Results")
h2("5.1 Feature reduction")
table([
    ["Dataset", "Total features", "BPSO kept (avg)", "Reduction", "F1 all", "F1 BPSO"],
    ["Breast Cancer", "30", "14.7", "51 %", "0.963", "0.956"],
    ["Kidney Disease", "24", "12.2", "49 %", "0.994", "0.996"],
    ["Heart Disease", "13", "8.0", "38 %", "0.820", "0.805"],
    ["Liver (ILPD)", "10", "4.2", "58 %", "0.826", "0.830"],
    ["Diabetes (Pima)", "8", "4.8", "40 %", "0.626", "0.614"],
], [3.6, 2.6, 2.9, 2.3, 2.5, 2.5])
p("F1 uses the best classifier for each dataset (SVM, Random Forest, then Logistic Regression for "
  "the last three). On Kidney and Liver, BPSO is even slightly better; elsewhere slightly lower. "
  "None of the F1 differences is statistically significant.")
h2("5.2 Accuracy of three classifiers (all features → BPSO features)")
table([
    ["Classifier", "Breast", "Kidney", "Heart", "Liver", "Diabetes"],
    ["Logistic Regression", "0.980 → 0.969", "0.996 → 0.982", "0.843 → 0.827", "0.722 → 0.723",
     "0.769 → 0.761"],
    ["Random Forest", "0.961 → 0.958", "0.992 → 0.994", "0.821 → 0.794", "0.700 → 0.700",
     "0.761 → 0.748"],
    ["SVM", "0.973 → 0.967", "0.999 → 0.988", "0.823 → 0.803", "0.709 → 0.707", "0.762 → 0.760"],
], [3.6, 2.56, 2.56, 2.56, 2.56, 2.56])
h2("5.3 Overall ranking")
p("A <b>Friedman test</b> ranks all 10 methods across 25 settings (5 datasets × 5 classifiers). "
  "Rank 1 is best. Methods joined by a bar in the diagram are not significantly different.")
figure("cd_diagram.png", 16, "Figure 1. Average rank of each method (lower = better). "
       "BPSO ranks 7.20, next to the genetic algorithm (7.28).")
p("BPSO ranks 8th of 10. It is statistically tied with the genetic algorithm, RFE, the "
  "same-size filters and random subsets, and behind all-features (3.68), the cross-validated "
  "filters (3.80–3.88) and L1 (4.08).")
h2("5.4 Where BPSO did well")
bullets([
    "<b>Breast cancer:</b> BPSO's 15 features beat the 15 best features chosen by MI or ANOVA "
    "(F1 +0.028, p &lt; 0.001). These 30 features are highly correlated, and a wrapper picks "
    "features that work well <i>together</i>.",
    "<b>Diabetes and kidney:</b> BPSO was significantly better than random features of the same "
    "count, so it is choosing meaningfully.",
    "<b>Clinically sensible choices:</b> Glucose and BMI for diabetes, specific gravity and "
    "haemoglobin for kidney disease, major vessels and thalassemia for heart disease.",
])
figure("feature_frequency_pima_diabetes.png", 12.5,
       "Figure 2. How often BPSO chose each Pima feature across 50 runs. "
       "Glucose and BMI were chosen every time.")
h2("5.5 Effect of α (accuracy vs size trade-off)")
p("Lowering α makes BPSO prefer smaller subsets. On breast cancer, α = 0.90 keeps F1 at 0.953 "
  "with only about <b>7 of 30 features</b> (77 % fewer), versus F1 0.961 with 14 features at "
  "α = 0.99.")
figure("alpha_sensitivity_breast_cancer.png", 11,
       "Figure 3. Breast cancer: score (top) and number of features (bottom) for different α.")
h2("5.6 Convergence and runtime")
bullets([
    "BPSO finds its best subset within about 15–20 of the 50 iterations.",
    "It takes 1.5–8.9 seconds per run, about 1.3–2× the genetic algorithm and much slower than "
    "filters (under 1 second).",
    "Different folds pick somewhat different subsets (stability 0.37–0.61 on a 0–1 scale), "
    "because many subsets are almost equally good.",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 6
h1("6. Conclusion, limitations and future work")
h2("Conclusion")
p("BPSO is a good tool for building <b>compact models</b>: it roughly halves the number of "
  "clinical measurements needed with almost no loss in F1-score, and it picks medically sensible "
  "features. It does <b>not</b> improve accuracy over using all features or simple filters on "
  "these small datasets.")
h2("Why accuracy did not improve")
bullets([
    "The datasets have only 8–30 features, mostly useful ones, so removing features mostly "
    "removes information, not noise.",
    "BPSO optimises accuracy with Logistic Regression, but we evaluate F1 and ROC-AUC with five "
    "classifiers, which is a mismatch.",
    "Many subsets are almost equally good, so the search is somewhat random.",
])
h2("Limitations")
bullets([
    "Small, public, low-dimensional datasets; no real hospital data or external validation.",
    "Reduced search budget (20 particles, 50 iterations, 5 repeats).",
    "Simple median/mode imputation for missing values.",
    "Research use only — not a medical device.",
])
h2("Future work")
bullets([
    "Optimise ROC-AUC or balanced accuracy in the fitness function.",
    "Try V-shaped transfer functions and multi-objective (Pareto) BPSO.",
    "Test on high-dimensional data such as gene expression, where selection matters more.",
    "Include the cost of each medical test in the fitness function.",
])
say("“Our honest finding is that BPSO is useful for reducing features, not for increasing "
    "accuracy, on low-dimensional clinical data. We report this directly and used statistical "
    "tests so the conclusion is reliable.”")
story.append(PageBreak())

# ---------------------------------------------------------------- 7
h1("7. How to show the project to your guide")
note("<b>Do this 10 minutes before the evaluation:</b> open "
     "https://bpso-showcase.onrender.com and https://bpso-api.onrender.com/docs in your browser. "
     "The free hosting sleeps when unused, and the first load takes about 1 minute. Keep both tabs "
     "open, plus the GitHub repository and the paper PDF.")
h2("What to have ready")
table([
    ["Item", "Where"],
    ["Live dashboard", "https://bpso-showcase.onrender.com"],
    ["API documentation", "https://bpso-api.onrender.com/docs"],
    ["Code", "https://github.com/shashank-nova/bpso-disease-prediction"],
    ["Paper PDF", "Compile paper/swarm_paper_overleaf.zip on Overleaf"],
    ["Demo patient file", "paper/evaluation_guide/demo_patients.csv (6 breast-cancer records)"],
    ["This guide", "paper/evaluation_guide/BPSO_Project_Evaluation_Guide.pdf"],
], [4.2, 12.2])

h2("Suggested demo (10–12 minutes)")
p("<b>Step 1 — Introduce the problem (1 min).</b> Use the one-minute summary from Section 1.")
p("<b>Step 2 — Explain BPSO with the bird example (2 min).</b> Show the 0/1 particle table from "
  "Section 2.4 and the fitness formula from Section 2.5.")
p("<b>Step 3 — Dashboard, <i>Overview</i> tab (1 min).</b> Point to the four numbers at the top: "
  "datasets evaluated, methods, classifiers, and outer-fold runs per dataset (50).")
say("“These are real results from 12,500 test evaluations, not typed-in numbers. They are "
    "generated directly from the experiment output files.”")
p("<b>Step 4 — Dashboard, <i>Evaluation results</i> tab (3 min).</b> In <b>Evaluation set</b> "
  "choose <b>Main experiment</b>, then a Dataset (start with breast_cancer), a Classifier and "
  "the Metric f1. Compare BPSO with All features and the other methods, then switch the metric "
  "to n_selected to show how many features each method kept. Finally change Evaluation set to "
  "<b>α sensitivity</b> to show the size–accuracy trade-off.")
p("<b>Step 5 — Dashboard, <i>Try a prediction</i> tab (2 min).</b>")
steps([
    "Click <b>Browse files</b> and upload <b>demo_patients.csv</b>.",
    "Click <b>Score records</b>.",
    "The table shows a label (1 = malignant, 0 = benign) and a probability for each patient. "
    "Expected result: the first two are 1 (malignant), the other four are 0 (benign).",
])
note("<b>Important — say this clearly:</b> the demonstration model was trained on the full "
     "breast-cancer dataset, and these six patients are part of it. So this only shows that the "
     "system works end to end. The real accuracy evidence is the cross-validation in the "
     "Evaluation tab.")
p("<b>Step 6 — API documentation (1 min).</b> Open https://bpso-api.onrender.com/docs. Show "
  "<b>/health</b> (is the model loaded), <b>/metadata</b> (which 16 of 30 features BPSO "
  "selected for the deployed model) and <b>/predict</b>. Click /metadata → <b>Try it out</b> → "
  "<b>Execute</b> to show a live response.")
p("<b>Step 7 — Dashboard, <i>Methodology &amp; provenance</i> tab (1 min).</b> Show the exact "
  "settings used and how the dashboard numbers were produced from the experiment files.")
p("<b>Step 8 — Code and paper (1–2 min).</b> On GitHub open <b>src/bpso_fs/selectors/bpso.py</b> "
  "and show the velocity update and sigmoid. Then show Figure 1 (ranking) in the paper and end "
  "with the conclusion from Section 6.")

h2("If the internet or website does not work: run it on a laptop")
p("Install Python 3.10+ and open PowerShell in the project folder:")
for cmd in [
    'pip install -e ".[serve]"',
    "python scripts/train_model.py --config configs/reduced.yaml --dataset breast_cancer "
    "--output models/disease_model.joblib",
    "python -m uvicorn bpso_fs.api:app --port 8000",
    "python -m streamlit run app.py      (in a second PowerShell window)",
]:
    boxed([Paragraph(cmd, S["code"])], colors.HexColor("#f1f5f9"), GRID)
p("Then open http://localhost:8501 for the dashboard and http://localhost:8000/docs for the API. "
  "With Docker Desktop installed, the single command <b>docker compose up --build</b> does all "
  "of this.")
story.append(PageBreak())

# ---------------------------------------------------------------- 8
h1("8. Questions your guide may ask")
qa = [
    ("Why did you choose BPSO?",
     "Feature selection is a yes/no choice for each feature, so a binary search method fits "
     "naturally. PSO is simple, has few parameters and is widely used for wrapper feature "
     "selection, so it is a good method to evaluate carefully."),
    ("What is the difference between PSO and BPSO?",
     "PSO moves particles in continuous space. BPSO keeps a continuous velocity but turns it into "
     "a probability with the sigmoid function and samples each position bit as 0 or 1."),
    ("What does α do?",
     "It balances accuracy against the number of features. α = 0.99 mostly cares about accuracy; "
     "α = 0.90 accepts a little less accuracy for much smaller subsets."),
    ("How did you avoid data leakage?",
     "Missing-value filling, scaling, feature selection and parameter tuning all happen inside "
     "each training fold. The test fold is only used once, for the final score."),
    ("Why repeated cross-validation instead of one train/test split?",
     "One split depends on luck. 50 test folds give a mean and standard deviation and allow "
     "paired statistical tests."),
    ("Your method ranks 8th. Is the project a failure?",
     "No. The research question was whether BPSO can reduce features without losing much "
     "performance, and the answer is yes (38–58 % fewer features, F1 within about 0.015). It "
     "does not improve accuracy, and we report that honestly. Negative or neutral findings are "
     "valid research results."),
    ("Why compare with a random subset?",
     "As a sanity check: if BPSO were no better than random features of the same size, it would "
     "not be selecting meaningfully. It was significantly better on diabetes and kidney disease."),
    ("Why compare with a genetic algorithm?",
     "It is the other classic evolutionary method. Using the same fitness shows whether the swarm "
     "itself matters. Results were nearly identical, so the fitness design matters more."),
    ("What statistical tests did you use?",
     "Wilcoxon signed-rank tests (paired, per dataset) with Holm correction for multiple "
     "comparisons, and the Friedman test with the Nemenyi critical difference across all "
     "settings."),
    ("Why is F1 your main metric instead of accuracy?",
     "Several datasets are imbalanced (for example 71 % positive in the liver data). Accuracy can "
     "look good even when the minority class is missed; F1 focuses on the disease class."),
    ("What are the parameters of BPSO?",
     "20 particles, 50 iterations, inertia 0.9 to 0.4, c1 = c2 = 2, velocity limit 6, sigmoid "
     "transfer function, α = 0.99."),
    ("Can this be used in a hospital?",
     "Not yet. It uses public benchmark data with no external clinical validation. It is a "
     "research prototype."),
    ("What is the deployed model?",
     "A Logistic Regression trained on the full breast-cancer dataset using 16 of 30 features "
     "chosen by BPSO, served through FastAPI and shown in Streamlit."),
    ("What would you improve?",
     "Optimise ROC-AUC in the fitness, try multi-objective BPSO, test high-dimensional data, and "
     "add the cost of each test."),
    ("What did each team member do?",
     "Prepare this answer yourselves before the evaluation (literature review, coding, "
     "experiments, results analysis, paper writing, deployment)."),
]
for q, a in qa:
    story.append(KeepTogether([
        Paragraph(f"<b>Q. {q}</b>", S["p"]),
        Paragraph(a, ParagraphStyle("a", parent=S["p"], leftIndent=12)),
    ]))

# ---------------------------------------------------------------- 9
h1("9. Glossary")
table([
    ["Term", "Meaning"],
    ["Feature", "One input measurement (column), e.g. Glucose"],
    ["Feature selection", "Choosing a useful subset of features"],
    ["Wrapper method", "Judges a subset by training a model on it"],
    ["Filter method", "Ranks features with a statistic, without training a model"],
    ["Particle", "One candidate solution (a 0/1 feature mask) in the swarm"],
    ["Personal / global best", "Best subset found by one particle / by the whole swarm"],
    ["Inertia weight w", "How much a particle keeps its previous direction"],
    ["Sigmoid", "Function that turns any number into a probability between 0 and 1"],
    ["Fitness", "Score of a subset; lower is better in this project"],
    ["Fold", "One part of the data used for testing in cross-validation"],
    ["Nested cross-validation", "Inner CV for choices, outer CV for final testing"],
    ["Data leakage", "Test data accidentally influencing training"],
    ["Jaccard stability", "How similar the chosen subsets are across runs (0–1)"],
    ["p-value", "Chance of seeing the difference if there were none; below 0.05 = significant"],
    ["Friedman / Nemenyi", "Tests for ranking many methods over many datasets"],
    ["API", "A web address other programs call to get predictions"],
], [4.4, 12.0])

# ---------------------------------------------------------------- 10
story.append(PageBreak())
h1("10. Checklist before evaluation")
p("Tick each item before the evaluation starts.")
bullets([
    "Open the dashboard and API links 10 minutes early (they wake up in about a minute).",
    "Keep demo_patients.csv on the desktop.",
    "Compile the paper on Overleaf and keep the PDF open.",
    "Every team member can explain BPSO, the fitness formula and leakage in their own words.",
    "Know the key numbers: 5 datasets, 10 methods, 5 classifiers, 50 folds, 38–58 % reduction, "
    "rank 7.20.",
    "Decide who presents which part and what each member contributed.",
    "Have the laptop backup (Section 7) ready in case the internet fails.",
])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "BPSO Disease Prediction — Evaluation Guide")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="BPSO Project Evaluation Guide",
                        author="Kushwaha, Srivastava, Mishra, Kumar")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
