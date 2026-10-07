# Lab experiment contract (read fully before starting)

Root: E:\Plant-Disease-Detection-and-Analysis-System\LAB  (call it LAB)
Data:  LAB/data  (OpenCV sample images/videos, LAB/data/models/*.onnx|*.pt). PlantVillage leaf images: ../data/plantvillage/<class>/*.JPG
Never modify anything outside LAB. Never modify "LAB/index page.docx".

## Per experiment deliverables
1. Script: LAB/code/expNN_<slug>.py  (NN = two digits). Self-contained, runs with `python code/expNN_<slug>.py` from LAB.
   - Student lab-report style: short header docstring, clear comments, no argparse, no GUI windows (headless OpenCV: never call cv2.imshow/waitKey).
   - Compact but complete: target 80-180 lines, hard max 230 lines. Save figures with matplotlib (dpi=150) or cv2.imwrite.
   - All outputs go to LAB/outputs/expNN/. Use relative paths from LAB (os.chdir to LAB or build paths from __file__).
   - Use a fixed random seed where randomness is involved.
   - Print the key numeric results to stdout (they will be captured into console.txt).
2. Figures: LAB/outputs/expNN/figNN_<name>.png  (fig01_..., fig02_...). 2-5 figures per experiment. Use width <= 12 in at 150 dpi.
   Every figure must have readable titles/labels; grids of images should have subplot titles. No blank axes. Verify visually (Read the PNG).
3. Console: LAB/outputs/expNN/console.txt  = captured stdout of the final successful run (run: python code/expNN_x.py > outputs/expNN/console.txt 2>&1 ... then check it).
4. Content: LAB/outputs/expNN/content.json with EXACTLY this schema:
{
  "number": 1,
  "title": "Implement image preprocessing and Edge detection",   # exactly as in the index table
  "aim": "To ...",                                                # one sentence starting with 'To'
  "objectives": ["...", "..."],                                   # 2-4 bullet strings
  "theory": ["paragraph", "## Sub heading", "paragraph", ...],    # 4-9 items; items starting with '## ' are sub-headings. Plain text only, formulas written inline in plain text e.g. x = K [R | t] X. 350-600 words total. Correct, textbook level, no fluff.
  "tools": "Python 3.11, OpenCV 5.0, NumPy, Matplotlib",         # one line listing what the script actually uses
  "dataset": "one line describing the input data used and where it came from",
  "algorithm": ["Step 1 ...", "Step 2 ..."],                      # 5-10 numbered steps matching the script
  "code_file": "code/expNN_<slug>.py",
  "figures": [{"file": "figNN_x.png", "caption": "Figure NN.k: ..."}],   # in display order; captions 1-2 sentences, specific
  "results": ["observation paragraph with the actual numbers obtained", "..."],  # 2-4 items, must quote real numbers from console.txt
  "conclusion": "2-4 sentences.",
  "viva": [{"q": "question", "a": "short answer"}]                # exactly 3 items
}
Write content.json with json.dump(..., indent=2, ensure_ascii=False). Use plain ASCII quotes and hyphens in prose; no markdown other than the '## ' sub-heading marker; no em dashes.

## Quality bar
- Results must be real: run the script, read the numbers, put them in results.
- Look at every figure PNG with the Read tool and fix anything ugly (overlapping titles, tiny text, wrong colours from BGR/RGB mix-ups, empty panels).
- Prefer PlantVillage leaf images where a natural image is needed and the experiment is not tied to a specific dataset (the student's project is plant disease detection).
- GPU (RTX 3050, 6 GB) is available for torch; keep batch sizes small.
- Do not leave debug prints or half-finished sections. Do not create files outside LAB/code and LAB/outputs/expNN.
