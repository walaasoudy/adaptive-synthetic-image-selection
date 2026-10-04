"""Build the HAM10000 ASISM v2 walkthrough notebook: an explanation of the pipeline and a map of the
code that implements it. It does not copy the repository into the notebook.

Run from the repository root:

    python notebooks/build_ham10000_v2_walkthrough.py

The repository stays the source of truth. Every file and function the notebook names is looked up
here at build time, and every code excerpt is cut from the file it names; a renamed or removed
function fails the build instead of leaving a stale reference. The notebook holds no GPU step and no
pod command (those are in ham10000_asism_v2_learned_pod_walkthrough.ipynb).
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "ham10000_asism_v2_project_walkthrough.ipynb"

CODE_MAP: dict[str, list[str]] = {}          # every reference made below; the notebook re-checks it
_TREES: dict[str, tuple[ast.Module, list[str]]] = {}


def _parsed(relative: str) -> tuple[ast.Module, list[str]]:
    if relative not in _TREES:
        path = REPO / relative
        if not path.is_file():
            raise SystemExit(f"{relative}: not in this checkout")
        text = path.read_text(encoding="utf-8")
        _TREES[relative] = (ast.parse(text), text.splitlines())
    return _TREES[relative]


def _node(relative: str, name: str):
    """A top-level function or class, or Class.method."""
    scope = _parsed(relative)[0].body
    for part in name.split("."):
        found = [n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        if not found:
            raise SystemExit(f"{relative}: no function or class named {name!r}")
        scope = found[0].body
    return found[0]


def ref(relative: str, name: str | None = None) -> str:
    """Markdown for one reference: `name` and a link to the file with the line it starts at."""
    CODE_MAP.setdefault(relative, [])
    if name is None:
        if not (REPO / relative).is_file():
            raise SystemExit(f"{relative}: not in this checkout")
        return f"[`{relative}`](../{relative})"
    line = _node(relative, name).lineno
    if name not in CODE_MAP[relative]:
        CODE_MAP[relative].append(name)
    return f"`{name}` — [`{relative}:{line}`](../{relative})"


def excerpt(relative: str, name: str, start: str | None = None, end: str | None = None) -> str:
    """Lines of `name` cut from the file, from the first line containing `start` to the first later
    line containing `end` (both inside the function). A marker that is not found fails the build."""
    node = _node(relative, name)
    lines = _parsed(relative)[1]
    first, last = node.lineno - 1, node.end_lineno - 1
    if start is not None:
        hits = [i for i in range(first, last + 1) if start in lines[i]]
        if not hits:
            raise SystemExit(f"{relative}:{name}: start marker {start!r} not found")
        first = hits[0]
    if end is not None:
        hits = [i for i in range(first, last + 1) if end in lines[i]]
        if not hits:
            raise SystemExit(f"{relative}:{name}: end marker {end!r} not found")
        last = hits[0]
    body = textwrap.dedent("\n".join(lines[first:last + 1]))
    ref(relative, name)
    return (f"**من الكود الفعلي** — [`{relative}`](../{relative})، السطور {first + 1}–{last + 1}:\n\n"
            f"```python\n{body}\n```")


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def table(rows: list[tuple[str, str]]) -> str:
    return "| بيعمل إيه | فين في الكود |\n|---|---|\n" + "\n".join(f"| {what} | {where} |" for what, where in rows)


SECTIONS = [
    ("s1", "Dataset / Splits"), ("s2", "LoRA"), ("s3", "Synthetic Generation"),
    ("s4", "Candidate Pool / Safety"), ("s5", "Four Signals"), ("s6", "ASISM V2 — الصورة العامة"),
    ("s7", "Ranker Supervision"), ("s8", "Bootstrap / Lower Bound"), ("s9", "Adaptive Stopping"),
    ("s10", "C / D"), ("s11", "E4 — تجربة مستقلة لفحص الكمية"), ("s12", "Stage 4"), ("s13", "Metrics"),
    ("s14", "Final Comparison"),
]


def heading(anchor: str) -> str:
    number = [a for a, _ in SECTIONS].index(anchor) + 1
    return f'<a id="{anchor}"></a>\n\n---\n## {number}. {dict(SECTIONS)[anchor]}'


INTRO = """
# مشروع HAM10000 و ASISM v2 — شرح الـ pipeline وخريطة الكود

**الحالة يوم 2026-10-04:** مفيش أي تشغيل GPU متوافق عليه لـ v2. القياس (1,000 تدريب) و Stage 4 لسه ما اتعملوش، فالنوتبوك ده بيشرح **التصميم والكود**، مش نتايج v2.

**النوتبوك ده إيه:** مرجع تفتحيه فتفهمي المشروع كله وتشرحي الـ pipeline. لكل مرحلة: شرح بسيط، الملف والدالة المسؤولين عنها فعلًا، أهم الداخل والخارج، ومقطع قصير من الكود لو محتاج.

**النوتبوك ده مش إيه:**

- مش نسخة من الكود. الكود الأصلي في الـ repo هو المرجع، وكل مقطع هنا مقصوص منه وقت بناء النوتبوك.
- مش لتشغيل الـ GPU. أوامر الـ pod في [`ham10000_asism_v2_learned_pod_walkthrough.ipynb`](ham10000_asism_v2_learned_pod_walkthrough.ipynb) و [`docs/ham10000_asism_v2_learned_pod_commands.md`](../docs/ham10000_asism_v2_learned_pod_commands.md).
- خلايا الكود الثلاثة اللي فيه بتشتغل على CPU في ثواني ومفيهاش تدريب.

**مصادر الأرقام:** [`docs/ham10000_v2_experimental_contract.md`](../docs/ham10000_v2_experimental_contract.md) (العقد)، [`docs/ham10000_results_and_limitations.md`](../docs/ham10000_results_and_limitations.md)، [`docs/ham10000_status_2026-10-02.md`](../docs/ham10000_status_2026-10-02.md)، و [`configs/ham10000_asism_v2_ranker.yaml`](../configs/ham10000_asism_v2_ranker.yaml).

النوتبوك بيتبني بـ `python notebooks/build_ham10000_v2_walkthrough.py`. لو الكود اتغير، يتبني تاني؛ ما يتعدّلش بالإيد.
"""

DIAGRAM = """
## Architecture

```
 HAM10000  (10,015 images, 7,470 lesions, 7 classes)
     |
     v
 [1] SPLITS  lesion-level, six parts, frozen as "ham-stratified-v1"
     |
     |-- gen_train (3,586) ----------> [2] LoRA on SDXL  ("ham-lora-v1")
     |                                        |
     |                                        v
     |                                 [3] GENERATION  ->  3,168 synthetic candidates
     |                                        |
     |                                        v
     |                                 [4] CANDIDATE POOL + SAFETY FILTER  (removed 0)
     |                                        |
     |                                        v
     |                                 [5] FOUR SIGNALS per image
     |                                     similarity | IQA | uncertainty | explainability
     |                                        |
     |                                        v
     |          +-------------------- [6] ASISM V2 -----------------------+
     |          |                                                         |
     |-- classifier_train (1,641) --> [7] RANKER SUPERVISION              |
     |-- asism_tuning_heldout ------>     200 designed subsets x 5 seeds  |
     |      (1,377)                       = 1,000 small trainings         |
     |          |                         gate G1 -> fit -> acceptance    |
     |          |                                |                        |
     |          |                         [8] BOOTSTRAP (200 rankers)     |
     |          |                             lower 95% bound per image   |
     |          |                                |                        |
     |          |                         [9] ADAPTIVE STOPPING           |
     |          |                             which images AND how many   |
     |          +--------------------------------|------------------------+
     |                                           v
     |                                    [10] C = selected     D = random, same count per class
     |                                           |
     |-- classifier_train ---------------------->|
     |                                           v
     |                                    [12] STAGE 4  DenseNet-121, 20 seeds
     |                                         A real | B real+all | C real+selected | D real+random
     |                                           |
     |-- classifier_val (1,401) --------------->  monitoring only
     |                                           v
     |                                    [13] METRICS -> [14] COMPARISON
     |                                         confirmatory: C vs D, balanced accuracy
     |
     +-- final_eval_heldout (1,622) ---- protected; Stage 5 waits for the supervisor's policy


 [11] E4 (separate experiment): quantity curve at the Stage 4 recipe, 60 trainings.
      An independent check on HOW MANY. It does not train the ranker and does not set the count.
```
"""

BACKGROUND = """
## خلفية في فقرة: ليه v2؟

النسخة الأولى (v1) اشتغلت من أولها لآخرها واختارت 616 صورة من 3,168. على `final_eval_heldout`: A (حقيقي بس) 0.566، B (حقيقي + الكل) 0.645، C (حقيقي + المختار) 0.612 balanced accuracy. الاختبار التأكيدي C ضد B طلع −0.033 (p = 0.047)، يعني الاختيار ما غلبش «استخدم الكل». النتيجة دي مجمّدة وما بتتعدّلش.

التشخيص بعدها طلّع ثلاث مشاكل، وكل واحدة اتحوّلت لقاعدة في v2:

| المشكلة في v1 | القاعدة في v2 |
|---|---|
| C فيها 616 و B فيها 3,168: المقارنة خلطت «أنهي صور» مع «كام صورة» | حالة **D**: عشوائي بنفس عدد C لكل class |
| درجة المنفعة اللي اتعلم منها الـ ranker كانت أغلبها ضوضاء (تشغيلة واحدة لكل مجموعة) | **5 تكرارات** لكل مجموعة، وبوابة **G1** قبل أي تعلّم |
| العدد اتحدد عمليًا بحد أدنى يدوي | **قاعدة وقوف**: العدد ناتج من النموذج، مفيش K ولا نسبة |

كود v1 وتجارب التشخيص لسه في الـ repo، بس مش جزء من المسار اللي النوتبوك ده بيشرحه.
"""

CHECK_INTRO = """
## فحص خريطة الكود

الخلية دي بتتأكد إن كل ملف ودالة النوتبوك بيذكرهم موجودين في الـ checkout المفتوح. قراءة بس، من غير torch ولا GPU.
"""

CHECK_CODE = '''
import ast
from pathlib import Path

CODE_MAP = __CODE_MAP__


def find_repo_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "scripts").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Open this notebook from the project checkout.")


def defined_names(path):
    names = set()

    def walk(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(prefix + node.name)
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix + node.name + ".")

    walk(ast.parse(path.read_text(encoding="utf-8")).body)
    return names


REPO = find_repo_root()
missing = []
for relative, wanted in CODE_MAP.items():
    path = REPO / relative
    if not path.is_file():
        missing.append(relative)
    elif wanted:
        missing += [f"{relative}:{name}" for name in wanted if name not in defined_names(path)]
print(f"{len(CODE_MAP)} files, {sum(map(len, CODE_MAP.values()))} functions and classes referenced")
print("all present" if not missing else "MISSING (rebuild the notebook): " + ", ".join(missing))
'''

TOY_U_INTRO = """
### مثال توضيحي 1: شكل معادلة المنفعة

**الأرقام هنا مخترعة للشرح، مش قياسات.** الخلية بتحسب `U(S) = b + λ·log(1 + Σw)` لثلاث أنواع من الصور: متوسطة (`w = 1`)، قوية (`w = 1.5`)، وضعيفة (`w = 0.5`).

اللي المفروض تشوفيه: (1) كل مضاعفة للعدد بتضيف نفس المقدار تقريبًا، يعني الفايدة بتقلّ مع الزيادة؛ (2) الصور القوية بتوصل لنفس المنفعة بعدد أقل.
"""

TOY_U_CODE = '''
import numpy as np

b, lam = 0.90, 0.010                      # invented numbers, for illustration only
sizes = np.array([0, 125, 250, 500, 1000, 2000, 3168])
print(f"{'images':>7} | {'w=0.5':>7} | {'w=1.0':>7} | {'w=1.5':>7}")
for n in sizes:
    print(f"{n:>7} | " + " | ".join(f"{b + lam * np.log1p(w * n):7.4f}" for w in (0.5, 1.0, 1.5)))
'''

TOY_STOP_INTRO = """
### مثال توضيحي 2: الحد الأدنى 95% ونقطة الوقوف

**الأرقام مخترعة برضه.** class واحد فيه 300 صورة؛ وزنها الحقيقي نازل من +1 لـ −0.5. بنعمل 200 «نسخة ranker» كل واحدة شايفة الوزن بضوضاء، وبعدها نطبّق نفس فكرة القاعدة: نرتّب بالحد الأدنى، ونقف عند أول صورة حدها الأدنى مش فوق الصفر.

اللي المفروض تشوفيه: القاعدة بتاخد **أقل** من عدد الصور اللي متوسطها موجب، لأنها عايزة تقريبًا كل النسخ تتفق. كل ما الضوضاء تزيد، العدد يقلّ. ده مبسّط: الكود الحقيقي بيحسب الزيادة في `U` نفسها لكل نسخة ولكل الـ classes مع بعض.
"""

TOY_STOP_CODE = '''
import numpy as np

rng = np.random.default_rng(0)
true_w = np.linspace(1.0, -0.5, 300)                      # invented: 200 useful images, 100 harmful
for noise in (0.05, 0.20, 0.50):
    estimates = true_w + rng.normal(0, noise, size=(200, 300))    # 200 bootstrap rankers
    lower = np.quantile(estimates, 0.05, axis=0)                   # each image's lower 95% bound
    order = np.argsort(-lower)                                     # the class offers its best first
    stop = int(np.argmax(lower[order] <= 0)) if (lower <= 0).any() else len(order)
    print(f"noise {noise:.2f}: mean weight > 0 for {(estimates.mean(0) > 0).sum():3d} images, "
          f"the rule keeps {stop:3d}; of those truly useful: {(true_w[order[:stop]] > 0).sum():3d}")
'''


def build_sections() -> list[dict]:
    cells: list[dict] = []
    add = cells.append
    splits_py = "scripts/data/ham10000/01_build_splits.py"
    pre_py = "scripts/data/ham10000/02_preprocess_images.py"
    lora_py = "scripts/train/ham10000_train_lora_sdxl.py"
    recipes_py = "scripts/generate/ham10000_recipes.py"
    gen_py = "scripts/generate/ham10000_generate_synthetic_images.py"
    signals_py = "scripts/asism/ham10000_01_compute_signals.py"
    gonogo_py = "scripts/asism/ham10000_02_gonogo.py"
    sup_py = "scripts/asism_v2/supervision.py"
    pipe_py = "scripts/asism_v2/pipeline.py"
    gates_py = "scripts/asism_v2/gates.py"
    stop_py = "scripts/asism_v2/stopping.py"
    sel_py = "scripts/asism_v2/selection_files.py"
    util_py = "scripts/followup/ham10000_asism_v2_utility.py"
    learned_py = "scripts/followup/ham10000_asism_v2_learned_select.py"
    e4_py = "scripts/followup/ham10000_e4_quantity_curve.py"
    train_py = "scripts/classify/ham10000_train_conditions.py"
    agg_py = "scripts/classify/ham10000_aggregate_conditions.py"
    metrics_py = "scripts/utils/ham10000_metrics.py"
    cmp_py = "scripts/eval/ham10000_compare_conditions.py"

    # ------------------------------------------------------------------ 1
    add(markdown(heading("s1") + """

**الفكرة:** HAM10000 فيها 10,015 صورة لـ 7,470 lesion و 7 أنواع (`nv`، `mel`، `bkl`، `bcc`، `akiec`، `vasc`، `df`). نفس الـ lesion ممكن يتصور أكتر من مرة، فالتقسيم بيتعمل على مستوى الـ **lesion**: لو صورتين لنفس الـ lesion وقعوا في قسمين مختلفين، كل الأرقام اللي بعد كده تبقى منفوخة. بعد التقسيم فيه audit بيثبت على الملفات المكتوبة فعلًا إن مفيش تسريب، ولو لقى تسريب السكريبت بيفشل.

| القسم | عدد الصور | الاستخدام المسموح |
|---|---|---|
| `gen_train` | 3,586 | تدريب الـ LoRA، ومرجع للإشارات |
| `gen_val` | 388 | متابعة تدريب الـ LoRA بس |
| `classifier_train` | 1,641 | الجزء الحقيقي في كل classifier |
| `classifier_val` | 1,401 | متابعة Stage 4. **ممنوع في اختيار الصور** |
| `asism_tuning_heldout` | 1,377 | قياس المنفعة بس |
| `final_eval_heldout` | 1,622 | التقييم النهائي بس، ومحمي في الكود |

**ليه ستّ أقسام:** كل قرار بيتاخد على داتا غير اللي بيتحكم عليه بيها.

**المعالجة:** الصور بتتحط على مربع 512×512 بـ letterbox (تصغير + حشو رمادي)، من غير قص عشان الـ lesion اللي على الطرف ما يتقطعش، ومن غير تحويل لأبيض وأسود لأن اللون أهم علامة في الـ dermoscopy.

""" + table([
        ("تقسيم الـ lesions على الأقسام مع الحفاظ على نسب الأنواع", ref(splits_py, "partition_groups_stratified")),
        ("إثبات إن مفيش lesion ولا صورة في قسمين", ref(splits_py, "audit_splits")),
        ("الـ letterbox", ref(pre_py, "letterbox_resize_rgb")),
        ("حماية `final_eval_heldout`", ref("scripts/utils/splits.py", "assert_final_eval_access_allowed")),
        ("الإعدادات (النسب، الـ seed 42، اسم التقسيم)", ref("configs/splits_ham10000.yaml")),
    ]) + """

**الداخل:** ملف الـ metadata بتاع HAM10000 والصور الخام.
**الخارج:** `data/ham10000/processed/splits/ham-stratified-v1/<split>.csv` و `processed/images/ham-stratified-v1/<split>/`.
"""))
    add(markdown(excerpt(splits_py, "audit_splits", start="overlaps = ", end="count\"] = len(shared)")))

    # ------------------------------------------------------------------ 2
    add(markdown(heading("s2") + """

**الفكرة:** SDXL نموذج بيرسم صور من نص، ومتدرب على صور عامة. تدريبه كله غالي جدًا، فبنستخدم **LoRA**: أوزان SDXL بتفضل مجمّدة، وبنضيف جنبها مصفوفات صغيرة بتتعلم «الفرق» المطلوب بس. الـ `rank` هو حجم المصفوفات دي.

| البند | القيمة (مجمّدة) |
|---|---|
| النموذج الأساسي | SDXL base 1.0 |
| LoRA | rank 32، alpha 32، على الـ UNet بس |
| التدريب | 8,000 خطوة، 768×576، learning rate 1e-4 (cosine) |
| الداتا | `gen_train` للتدريب، `gen_val` للمتابعة |
| الناتج | `ham-lora-v1/final` |

**تفصيلة مهمة:** الـ LoRA بيتدرب على الصور **من غير الحشو الرمادي** وبنسبتها الأصلية 4:3. لو اتدرب على المربعات المحشوة هيتعلم يرسم شرايط رمادية.

""" + table([
        ("تجهيز صور التدريب (من غير حشو) والـ captions", ref("scripts/data/ham10000/04_prepare_lora_inputs.py", "prepare_split")),
        ("حلقة التدريب", ref(lora_py, "train_loop")),
        ("خطوة تدريب واحدة (إضافة ضوضاء وتوقّعها)", ref(lora_py, "training_step")),
        ("حفظ الـ adapter النهائي", ref(lora_py, "export_final")),
        ("الإعدادات", ref("configs/ham10000_stage1.yaml")),
    ]) + """

**الداخل:** صور `gen_train` والـ captions (اسم المرض في نص).
**الخارج:** ملف الـ LoRA adapter، ومعاه provenance (hash الإعدادات والتقسيم).

**حدود معروفة:** generator واحد بـ seed واحد و snapshot واحد، ومفيش إثبات مستقل إن كل صورة فعلًا شكل المرض المطلوب.
"""))

    # ------------------------------------------------------------------ 3
    add(markdown(heading("s3") + """

**الفكرة:** لكل class بنطلب من SDXL + LoRA عدد من الصور. الأنواع النادرة بتاخد أكتر، بس بحساب: المعادلة بتستخدم الجذر التربيعي (`rarity_exponent = 0.5`) عشان ما نطلبش من الـ generator تنوّع ما شافوش في class فيه حوالي 100 صورة حقيقية بس.

`target = 400 × (median_count / class_count) ^ 0.5`، محصور بين 100 و 1,500.

الناتج **3,168 صورة**:

| nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|
| 112 | 276 | 273 | 400 | 486 | 736 | 885 |

دي أعداد **candidates** مش أعداد مختارة: ASISM هو اللي بيقرر بعد كده. و v2 مش بيولّد صور جديدة.

كل صورة ليها seed خاص بيها محسوب من `sha256(seed:recipe_id)`، فأي صورة ممكن تتعاد لوحدها. الإعدادات: 40 خطوة، guidance 7.0، 768×576. بعد التوليد الصور بتعدّي على نفس دالة الـ letterbox بتاعة الصور الحقيقية.

""" + table([
        ("حساب عدد الصور لكل class", ref(recipes_py, "rarity_quotas")),
        ("seed لكل صورة", ref(recipes_py, "derive_image_seed")),
        ("التوليد نفسه", ref(gen_py, "run_generation")),
        ("فحص كل صورة ناتجة ضد الـ manifest", ref(gen_py, "validate_generation_outputs")),
        ("توحيد الهندسة مع الصور الحقيقية", ref("scripts/generate/ham10000_standardize_generated.py", "standardize_directory")),
        ("الإعدادات", ref("configs/ham10000_stage2.yaml")),
    ]) + """

**الداخل:** الـ LoRA adapter، وأعداد الأنواع في `gen_train` بس.
**الخارج:** `outputs/ham10000/stage2/ham-stratified-v1/all_candidates.csv` (ده كمان مدخل الحالة B) والصور.

**حدود معروفة (قياسات، مش أخطاء كود):** تنوّع أقل من الحقيقي؛ 84.8% من `mel` الصناعي الـ classifier المساعد قراه `nv`؛ و `bkl` الصناعي كل الـ judges اللي اتجربوا بيتعرفوا عليه وحش.
"""))
    add(markdown(excerpt(recipes_py, "rarity_quotas", start="present = ")))

    # ------------------------------------------------------------------ 4
    add(markdown(heading("s4") + """

**الفكرة:** قبل أي ترتيب، فلتر أمان بيشيل نوعين بس: الصورة اللي ما اتفتحتش أو ما اتقاستش (`iqa_valid` false)، والصورة اللي نسخة شبه مطابقة لصورة حقيقية (تشابه ≥ 0.95). الفلتر «fail closed»: الصورة اللي مالهاش صف في ملف الإشارات بتتعامل كإنها غير آمنة. علامات الجودة التانية (blur، تباين، حواف) **مش** شروط أمان؛ دي بتدخل كإشارة.

**النتيجة الفعلية:** الفلتر شال **صفر** صورة، فالـ 3,168 كلهم آمنين.

**التجميد:** ملف الـ candidates متثبّت بـ sha256 في الكود (`bf8047b5…`). أي نسخة تانية من الملف بتترفض، فما ينفعش مجموعة الصور تتغير من غير ما حد ياخد باله.

""" + table([
        ("قاعدة الأمان (المشتركة بين كل المراحل)", ref("scripts/asism/candidate_pool.py", "unsafe_candidates")),
        ("تحميل الـ pool لـ v2: فحص الـ hash، دمج الإشارات الأربعة، تطبيق الأمان", ref(sup_py, "load_signal_table")),
        ("فحص إن الإشارات كاملة ومفيش قيم ناقصة", ref("scripts/asism_v2/features.py", "validate_frame")),
    ]) + """

**الداخل:** `all_candidates.csv` وملفات الإشارات الأربعة (`*_scores.parquet`).
**الخارج:** جدول واحد في الذاكرة: `image_id`، `dx`، والإشارات الأربعة لكل صورة آمنة، وتقرير بالـ hashes.
"""))
    add(markdown(excerpt(sup_py, "load_signal_table", start="removed = {}", end="safe = pool.drop")))

    # ------------------------------------------------------------------ 5
    add(markdown(heading("s5") + """

**الفكرة:** كل صورة صناعية بتاخد أربع أرقام. كل إشارة بتتحفظ في ملف لوحدها ومعاها provenance، عشان أي واحدة تتعاد أو تتراجع من غير ما تبوّظ الباقي.

| الإشارة | العمود المستخدم | بتقيس إيه | إزاي |
|---|---|---|---|
| Similarity | `similarity_knn_mean` | شبه الصور الحقيقية من نفس الـ class | DINOv2 مجمّد، متوسط التشابه مع أقرب 15 صورة من `gen_train` |
| IQA | `iqa_composite` | جودة الصورة | حدّة، تباين، وخمس علامات عيوب |
| Uncertainty | `uncertainty_mutual_information` | الـ classifier محتار قد إيه | MC dropout، 20 مرة |
| Explainability | `explainability_calibrated_typicality` | الـ classifier بيبص فين، وهل ده طبيعي للـ class | Grad-CAM مقارنة بمرجع من `gen_train` |

**إشارة خامسة اتشالت (Agreement):** محتاجة «حَكَم» موثوق يقول الصورة من أنهي class. الحكمين اللي اتجربوا فشلوا في فحوصهم المكتوبة مسبقًا، فما اتستخدمتش.

**Go/No-Go:** فحص قبل أي تعلّم: كل إشارة سليمة تقنيًا، مفيش قيم ناقصة، ومش تكرار لإشارة تانية (الارتباط جوه الـ class أقل من 0.90). الأربعة عدّوا. النجاح هنا معناه إن الإشارات **مختلفة عن بعض**، مش إنها مفيدة؛ ده اللي v2 بيختبره.

**في v2 مفيش اتجاه مفروض:** محدش بيقول «uncertainty الأعلى أحسن» أو العكس. الـ ranker بيتعلم وزن واتجاه كل إشارة لكل class. والتطبيع (طرح المتوسط والقسمة على الانحراف) بيتحسب من صور الـ train بس.

""" + table([
        ("Similarity", ref(signals_py, "run_similarity")),
        ("IQA", ref(signals_py, "run_iqa")),
        ("Uncertainty (ومعاها Agreement في نفس الدالة)", ref(signals_py, "run_uncertainty_and_agreement")),
        ("Explainability", ref(signals_py, "run_explainability")),
        ("فحص التكرار بين الإشارات", ref(gonogo_py, "check_redundancy")),
        ("قرار كل إشارة", ref(gonogo_py, "decide")),
        ("التطبيع على صور الـ train بس", ref("scripts/asism_v2/features.py", "TrainingStandardizer")),
    ]) + """

**الداخل:** الصور الصناعية، `gen_train` كمرجع، والـ classifier المساعد (V3a).
**الخارج:** `similarity_scores.parquet`، `iqa_scores.parquet`، `uncertainty_scores.parquet`، `explainability_scores.parquet`، ولكل واحد `.provenance.json`.
"""))

    # ------------------------------------------------------------------ 6
    add(markdown(heading("s6") + """

**الفكرة:** ASISM v2 بيجاوب على سؤالين مع بعض: **أنهي صور** و**كام صورة**، من غير ما حد يحدد K أو نسبة. بيعمل كده على سبع خطوات، كل واحدة بترفض تشتغل لو اللي قبلها ما نجحتش:

| # | الخطوة | CPU / GPU | الملف الناتج |
|---|---|---|---|
| 1 | **plan**: رسم 200 مجموعة مصممة | CPU | `utility_plan.json` |
| 2 | **measure**: 1,000 تدريب صغير | **GPU** (مقفول) | `utility_runs_fit.jsonl`، `utility_runs_test.jsonl` |
| 3 | **gate (G1)**: هل القياس ثابت؟ | CPU | `g1_reliability.json` |
| 4 | **accept**: هل الـ ranker بيتوقع صح على مجموعات ما شافهاش؟ | CPU | `ranker_acceptance.json` |
| 5 | **fit**: الـ ranker و 200 نسخة bootstrap | CPU | `ranker_fit_seed42.json` |
| 6 | **select**: قاعدة الوقوف، وكتابة C و D | CPU | `c_selected.csv`، `d_selected_seed*.csv`، manifest |
| 7 | **stability**: نفس الاختيار بخمس fit seeds | CPU | `selection_stability.json` |

**التسجيل المسبق (pre-registration):** كل رقم بيأثر على النتيجة مكتوب في `configs/ham10000_asism_v2_ranker.yaml` ومنسوخ في الكود. لو الملف اختلف عن النسخة اللي في الكود، التحميل بيرفض. فما ينفعش رقم يتغير بتعديل ملف الإعدادات بس.

**القفل:** خطوة measure بترفض طول ما `MEASUREMENT_APPROVED = False`، حتى لو الـ flag اتكتب في الأمر.

""" + table([
        ("الإعدادات المعتمدة", ref("configs/ham10000_asism_v2_ranker.yaml")),
        ("رفض أي قيمة مختلفة عن المعتمد", ref("scripts/asism_v2/prereg.py", "load_prereg")),
        ("plan", ref(util_py, "run_plan")),
        ("measure", ref(util_py, "run_measure")),
        ("gate", ref(util_py, "run_gate")),
        ("accept", ref(learned_py, "run_accept")),
        ("fit", ref(learned_py, "run_fit")),
        ("select", ref(learned_py, "run_select")),
        ("stability", ref(learned_py, "run_stability")),
        ("الكتابة مرة واحدة (exclusive create)", ref("scripts/asism_v2/contracts.py", "write_new_json")),
    ]) + """

**الداخل:** الـ pool الآمن بإشاراته، `classifier_train`، `asism_tuning_heldout`.
**الخارج:** كله في `outputs/ham10000/stage3_asism_v2_ranker/ham-stratified-v1/`.
"""))
    add(markdown(excerpt(util_py, "run_measure", end="pass {REAL_MODELS_FLAG}")))

    # ------------------------------------------------------------------ 7
    add(markdown(heading("s7") + """

**الفكرة:** الـ ranker محتاج «إجابات صح» يتعلم منها. الإجابة هي: **لو ضفت المجموعة دي من الصور الصناعية للتدريب، الـ classifier بيبقى كويس قد إيه؟** بنقيس ده فعلًا بتدريب classifier صغير.

**1) المجموعات مصممة، مش عشوائية.** في v1 المجموعات كانت عشوائية وبحجم واحد، فكانت شبه بعض والفرق بينها أقل من الضوضاء. هنا:

- كل صورة بتاخد **دور** واحد: train (60%) أو validation (20%) أو test (20%)، جوه الـ class بتاعها. المجموعة فيها صور من دور واحد بس، فالثلاث مجموعات ما بيتشاركوش في أي صورة.
- **الأحجام مختلفة:** 125 / 250 / 500 / 1,000 للـ train، و 125 / 250 / 500 للباقي.
- **نسب الـ classes مختلفة** من مجموعة للتانية (Dirichlet).
- **نص المجموعات «مايلة»:** مسحوبة من النص الأعلى أو الأدنى لإشارة واحدة، عشان الإشارات يبقى ليها أثر ممكن يتقاس.
- العدد: 120 train + 40 validation + 40 test = **200 مجموعة**.

**2) القياس.** لكل مجموعة: تدريب DenseNet صغير (224 px، 300 خطوة) على `classifier_train` + المجموعة، وقياس macro AUROC على `asism_tuning_heldout`. بيتكرر بـ **5 seeds** (42–46)، والدرجة هي المتوسط. المجموع 200 × 5 = **1,000 تدريب**. الدرجة مطلقة، من غير طرح خط أساس.

**3) بوابة G1 (قبل أي تعلّم).** بتسأل: الفرق بين المجموعات حقيقي ولا ضوضاء seeds؟ بتحسب ثبات متوسط الـ 5 تكرارات، ولازم يبقى **≥ 0.80**. بتتحسب على مجموعات train و validation بس؛ الـ test لسه ما اتفتحش. ملحوظة مكتوبة في الكود: الرقم ده فيه أثر الحجم، ونفس الرقم جوه كل حجم بيتكتب جنبه بس مش هو اللي بيحكم.

**4) النموذج.**

`U(S) = b + λ · log(1 + Σ wᵢ)` ،  `wᵢ = 1 + score(xᵢ, classᵢ) + class term`

`wᵢ` هو «العدد الفعلي» للصورة: 1 صورة متوسطة، أكتر من 1 صورة بتتحسب بأكتر، صفر ما بتضيفش، وسالب بتضر. الـ `score` مجموع الإشارات الأربعة مضروبة في أوزان بتتعلم **لكل class**.

**5) accept (مرة واحدة).** بعد الـ fit، بنقرا مجموعات الـ test مرة واحدة والملف بيتكتب exclusive، فما ينفعش تتقري وتتعدّل وتتقري تاني. شرطين لازم الاتنين:

- ارتباط Spearman **جوه كل حجم** بين التوقع والقياس ≥ 0.50، و p ≤ 0.05 (10,000 permutation).
- خطأ الـ ranker على الـ test أقل من نموذج «الحجم والـ class بس» (من غير إشارات).

كمان بيتسجل للمقارنة: similarity لوحدها، والـ composite متساوي الأوزان.

**لو G1 أو accept فشل: الترتيب المتعلَّم ما بيتستخدمش، ودي النتيجة اللي بتتكتب.**

""" + table([
        ("توزيع الأدوار جوه كل class", ref(sup_py, "assign_roles")),
        ("رسم الـ 200 مجموعة", ref(sup_py, "build_plan")),
        ("فحص إن التصميم يقدر يحدد أثر الحجم والـ class", ref(sup_py, "check_plan")),
        ("التدريب الصغير نفسه", ref("scripts/asism/ham10000_03_build_utility_subsets.py", "_measure")),
        ("اشتراط الشبكة الكاملة (كل مجموعة × كل seed)", ref("scripts/asism_v2/contracts.py", "validate_measurements")),
        ("G1", ref(gates_py, "reliability_gate")),
        ("النموذج", ref(pipe_py, "AdditiveUtilityRanker")),
        ("الـ fit (مع early stopping على الـ validation)", ref(pipe_py, "fit_ranker")),
        ("Spearman جوه كل حجم", ref(gates_py, "stratified_spearman")),
        ("accept", ref(gates_py, "acceptance")),
    ]) + """

**الداخل:** الـ pool بإشاراته، الخطة، والـ 1,000 قياس.
**الخارج:** `g1_reliability.json`، `ranker_acceptance.json`.
"""))
    add(markdown(excerpt(pipe_py, "AdditiveUtilityRanker", start="def score", end="return self.intercept + self.log_count * torch.log1p")))
    add(markdown(excerpt(gates_py, "reliability_gate", start="groups = ")))
    add(markdown(TOY_U_INTRO))
    add(code(TOY_U_CODE))

    # ------------------------------------------------------------------ 8
    add(markdown(heading("s8") + """

**الفكرة:** ranker واحد بيدّي رقم واحد لكل صورة، ومش بيقول واثق فيه قد إيه. عشان نعرف الثقة بنعمل **bootstrap**:

1. من الـ 120 مجموعة train، بنسحب 120 مجموعة **مع الإرجاع** (فبعض المجموعات بتتكرر وبعضها بيغيب).
2. بندرّب ranker كامل على العيّنة دي، بنفس الإعدادات ونفس early stopping.
3. بنكرر **200 مرة**.

دلوقتي لكل صورة 200 تقدير لفايدتها. **الحد الأدنى 95%** هو القيمة اللي 5% بس من النسخ تحتها (5th percentile). لو الحد ده فوق الصفر، يبقى تقريبًا كل النسخ متفقة إن الصورة مفيدة.

**ليه الحد الأدنى مش المتوسط:** المتوسط ممكن يطلع موجب بالصدفة. الحد الأدنى بيطلب دليل، فالصورة المشكوك فيها ما بتتاخدش.

الـ ranker والـ 200 نسخة بيتحفظوا في ملف واحد، فالاختيار بيقرا الـ fit بدل ما يعيده، والملف بيرفض لو محتواه أو مصدره اتغير.

""" + table([
        ("السحب مع الإرجاع وتدريب النسخ (جوه `fit_ranker`)", ref(pipe_py, "fit_ranker")),
        ("تدريب نسخة واحدة", ref(pipe_py, "_train")),
        ("حساب الحد الأدنى لكل صورة (`own_lower`)", ref(stop_py, "progressive_select")),
        ("حفظ الـ ranker والنسخ", ref("scripts/asism_v2/persist.py", "save_fitted")),
        ("تحميله مع فحص المصدر", ref("scripts/asism_v2/persist.py", "load_fitted")),
    ]) + """

**الإعدادات المعتمدة:** 200 نسخة، Adam، learning rate 0.03، من غير weight decay، حد أقصى 5,000 epoch و patience 200، والـ fit على درجات مطبّعة.

**الداخل:** قياسات train و validation.
**الخارج:** `ranker_fit_seed42.json`.
"""))
    add(markdown(excerpt(pipe_py, "fit_ranker", start="for b in range(bootstrap)", end="ensemble.append(member)")))

    # ------------------------------------------------------------------ 9
    add(markdown(heading("s9") + """

**الفكرة:** دي القاعدة اللي بتحدد **كام صورة**. مفيش K ولا نسبة ولا حد أدنى للفايدة:

1. جوه كل class، الصور بتترتب بالحد الأدنى بتاعها (أحسن صورة الأول).
2. في كل خطوة، كل class لسه شغّال **بيعرض** صورته الجاية.
3. لكل عرض بنحسب: لو ضفنا الصورة دي للمجموعة الحالية، `U` هتزيد قد إيه؟ بنحسبها في الـ 200 نسخة وناخد الحد الأدنى 95%.
4. الـ class اللي حد عرضه **≤ 0** بيقف نهائيًا.
5. من الباقيين، بناخد صاحب أعلى حد أدنى، ونكرر.

كل class ممكن يقف عند عدد مختلف، وكل خطوة وكل وقفة بتتسجل بسببها (`trajectory` و `stops`).

**ثلاث نتايج ممكنة، وكلهم صالحين:** `subset` (جزء من الصور)، `all` (كل الصور)، `none` (ولا صورة).

**الثبات (stability):** الـ fit والاختيار كله بيتعاد بخمس fit seeds (42–46)، وبيتكتب: أقل وأكبر عدد، وتطابق كل اختيار مع الاختيار المعتمد (Jaccard). ده **تقرير بس**: الاختيار المستخدم هو بتاع seed 42، ومحدش بيختار من الخمسة.

**القاعدة بترفض لو النموذج ما يقدرش يحدد أثر الحجم** (كل المجموعات بحجم واحد) أو أثر الـ class، ولو النسخ أقل من الحد الأدنى للـ bootstrap.

""" + table([
        ("قاعدة الوقوف كلها", ref(stop_py, "progressive_select")),
        ("الزيادة في `U`", ref(stop_py, "_utility")),
        ("الثبات على الـ fit seeds", ref(gates_py, "selection_stability")),
    ]) + """

**الداخل:** الـ ranker المحفوظ والـ pool الآمن.
**الخارج:** قايمة الصور المختارة، العدد لكل class، وسجل الخطوات.
"""))
    add(markdown(excerpt(stop_py, "progressive_select", start="def offer(name)", end="active.discard(name)")))
    add(markdown(TOY_STOP_INTRO))
    add(code(TOY_STOP_CODE))

    # ------------------------------------------------------------------ 10
    add(markdown(heading("s10") + """

**الفكرة:** عشان نعرف إن الفايدة جاية من **أنهي صور** مش من **العدد**، لازم نقارن C بحالة فيها نفس العدد بس عشوائي.

- **C** = اللي قاعدة الوقوف اختارته.
- **D** = لكل class، نفس عدد C، مسحوب عشوائي من **نفس الـ pool الآمن**.

فالفرق الوحيد بين C و D هو الترتيب المتعلَّم.

**سحبة D لكل seed:** لكل seed من الـ 20 بتوع Stage 4 فيه سحبة D مستقلة، عشان النتيجة ما تبقاش مربوطة بسحبة عشوائية واحدة محظوظة أو منحوسة.

**لو النتيجة `all` أو `none`:** مفيش D، لأن C بقت هي B (أو A) بالتعريف. بيتدرب A و B بس، والنتيجة نفسها هي اللي بتتكتب.

| `selection_outcome` | البروتوكول | اللي بيتدرب | المقارنة التأكيدية |
|---|---|---|---|
| `subset` | `asism_v2_learned` | A, B, C, D | C ضد D |
| `all` / `none` | `asism_v2_learned_all_or_none` | A, B | مفيش |

""" + table([
        ("تحديد النتيجة وبناء C و D", ref(sel_py, "build_selection")),
        ("سحبة D بنفس عدد C لكل class", ref(sel_py, "draw_matched_random")),
        ("كتابة الملفات والـ manifest", ref(sel_py, "write_selection")),
        ("تعريف البروتوكولين والمقارنات", ref("scripts/utils/ham10000_conditions.py", "ConditionProtocol")),
    ]) + """

**الداخل:** الـ ranker المحفوظ، الـ pool، وقايمة seeds الـ Stage 4.
**الخارج:** `c_selected.csv`، `d_selected_seed42.csv` … `d_selected_seed61.csv`، و `asism_v2_learned_selection_manifest.json`.
"""))
    add(markdown(excerpt(sel_py, "draw_matched_random")))

    # ------------------------------------------------------------------ 11
    add(markdown(heading("s11") + """

> **E4 تجربة مستقلة.** مش جزء من تدريب الـ ranker، ومش بتدخل في قاعدة الوقوف، والعدد اللي ASISM بيختاره **مش** جاي منها. دورها في v2: فحص خارجي نحط جنبه العدد اللي ASISM طلّعه.

**السؤال:** لما نزوّد الصور الصناعية، الأداء بيتغير إزاي؟ وهل التغيير ده ظاهر فوق الضوضاء؟

**التصميم (متوافق عليه يوم 2026-10-02 قبل الكود):**

| البند | القيمة |
|---|---|
| الـ recipe | بتاع Stage 4 نفسه: 512 px، 3,000 خطوة |
| الأحجام | 0، 250، 500، 1,000، 2,000، وكل الـ pool |
| السحب | سلسلتين متداخلتين: كل سلسلة ترتيب عشوائي للـ pool، والحجم `s` هو أول `s` صورة |
| الـ seeds | الحقيقي بس: 42–51. باقي الأحجام: سلسلتين × 42–46 |
| العدد | 60 تدريب، 10 لكل حجم |
| المقياس | macro AUROC على `asism_tuning_heldout` |
| القاعدة | GO / COARSE / NO من اختبار Welch، مكتوبة قبل التشغيل |

**الفرق عن قياس الـ ranker:**

| | قياس الـ ranker (القسم 7) | E4 |
|---|---|---|
| الهدف | يعلّم الـ ranker | يرسم منحنى الكمية |
| الـ recipe | صغير: 224 px، 300 خطوة | الكامل: 512 px، 3,000 خطوة |
| الصور | مجموعات مصممة (أحجام ونسب وميل) | عشوائي متداخل |
| العدد | 1,000 تدريب | 60 تدريب |

**الحالة:** الـ 60 تدريب خلصوا على الـ pod. ملف التشغيلات مش موجود على اللابتوب وقت بناء النوتبوك، **فنتيجة E4 مش مكتوبة هنا**.

""" + table([
        ("السلسلتين المتداخلتين", ref(e4_py, "build_chains")),
        ("الـ 60 خلية", ref(e4_py, "plan_cells")),
        ("التأكد إن الـ recipe هو بتاع Stage 4 بالظبط", ref(e4_py, "check_stage4_recipe")),
        ("قاعدة القرار", ref(e4_py, "decide")),
        ("العدد `q*` اللي بيتحط جنب عدد ASISM", ref("scripts/followup/ham10000_e4_consequences.py", "q_star")),
        ("التصميم", ref("docs/ham10000_e4_quantity_design.md")),
    ])))

    # ------------------------------------------------------------------ 12
    add(markdown(heading("s12") + """

**الفكرة:** ده الاختبار الحقيقي. بندرّب الـ classifier الكامل على أربع حالات، **والفرق الوحيد بينهم هو داتا التدريب**:

| الحالة | بيتدرب على |
|---|---|
| A | `classifier_train` بس (1,641) |
| B | الحقيقي + كل الـ 3,168 |
| C | الحقيقي + اختيار ASISM |
| D | الحقيقي + عشوائي بنفس عدد C لكل class |

**الـ recipe (مجمّد):** DenseNet-121 (ImageNet)، 512 px، 3,000 خطوة، batch 32، learning rate 1e-4، softmax + cross-entropy، من غير class weighting، **20 seed** (42–61). عدد الخطوات ثابت في كل الحالات، فمفيش حالة بتاخد تدريب أكتر عشان داتاها أكبر.

**فحص العدالة قبل أي جدول:** الـ aggregator بيقارن إعدادات كل التدريبات. لو أي حاجة غير الداتا اختلفت بين حالتين، بيرفض يكتب الجدول. وكمان بيتأكد إن C و D بنفس العدد، وإن نتيجة الاختيار في الـ manifest مطابقة للبروتوكول.

**ملحوظتين:** التدريب مش bit-deterministic، فالنتايج بتتقارن كتوزيعات على الـ seeds. والـ classifier بيتدرب من غير augmentation؛ ده حد مسجّل من حدود البروتوكول.

""" + table([
        ("تجميع داتا الحالة (حقيقي + ملف الصناعي بتاعها)", ref(train_py, "build_condition_records")),
        ("اختيار ملف الصناعي حسب الحالة والـ seed (D ليها ملف لكل seed)", ref(train_py, "resolve_synthetic_manifest")),
        ("تدريب حالة × seed", ref(train_py, "run")),
        ("حلقة التدريب", ref("scripts/utils/ham10000_classifier.py", "train_classifier")),
        ("فحص إن الداتا بس هي اللي اختلفت", ref(agg_py, "check_fairness")),
        ("فحص إن نتيجة الاختيار مطابقة للبروتوكول", ref(agg_py, "check_selection_outcome")),
        ("الإعدادات والمسارات", ref("configs/ham10000_asism_v2_learned_stage4.yaml")),
    ]) + """

**الداخل:** `classifier_train`، `all_candidates.csv`، `c_selected.csv`، `d_selected_seed*.csv`.
**الخارج:** لكل حالة × seed في `outputs/ham10000/stage4_asism_v2_learned/ham-stratified-v1/<حالة>/seed<رقم>/`: النموذج، التوقعات على `classifier_val`، و `run_manifest.json`.
"""))
    add(markdown(excerpt(train_py, "resolve_synthetic_manifest")))

    # ------------------------------------------------------------------ 13
    add(markdown(heading("s13") + """

**الفكرة:** الداتا مش متوازنة (`nv` هو الأغلبية)، فالـ accuracy العادية بتخدع: نموذج بيقول `nv` دايمًا ياخد رقم عالي. عشان كده المقياس الأساسي هو **balanced accuracy**.

| المقياس | بيقيس إيه | الأحسن |
|---|---|---|
| **Balanced accuracy** (الأساسي والتأكيدي) | متوسط الـ recall على الأنواع السبعة؛ كل نوع ليه نفس الوزن | الأعلى |
| Macro-F1 | متوسط F1 على الأنواع | الأعلى |
| Macro AUROC | قدرة النموذج يرتّب الصح فوق الغلط، من غير عتبة | الأعلى |
| Accuracy | نسبة الصح الكلية | الأعلى |
| **Macro average precision** (جديد في v2) | زي AUROC بس أدق مع الأنواع النادرة | الأعلى |
| **Brier score** (جديد في v2) | بُعد الاحتمالات عن الحقيقة. 0 مثالي، 2 واثق وغلطان | **الأقل** |
| **ECE top-label** (جديد في v2) | هل ثقة النموذج صادقة؟ لو قال 80% يبقى صح 80% من المرات. 15 خانة متساوية العرض | **الأقل** |

**الثلاثة الجداد لـ v2 بس.** تقارير v1 بتطلّع نفس الأربعة اللي طلّعتهم بالظبط، وفيه اختبار بيتأكد من ده. عدد الخانات في ECE ثابت (15) لأن تغييره بيغيّر الرقم.

""" + table([
        ("Balanced accuracy", ref(metrics_py, "balanced_accuracy")),
        ("Macro average precision", ref(metrics_py, "macro_average_precision")),
        ("Brier", ref(metrics_py, "multiclass_brier_score")),
        ("ECE", ref(metrics_py, "top_label_ece")),
        ("أنهي مقاييس لأنهي بروتوكول", ref(cmp_py, "metric_functions")),
    ])))
    add(markdown(excerpt(metrics_py, "top_label_ece", start="confidence = ")))

    # ------------------------------------------------------------------ 14
    add(markdown(heading("s14") + """

**الفكرة:** فرق بين رقمين مش نتيجة لوحده؛ لازم نعرف هل هو أكبر من الصدفة.

**1) Bootstrap على مستوى الـ lesion.** بنسحب lesions مع الإرجاع 2,000 مرة ونعيد حساب المقياس كل مرة، فبيطلع فاصل ثقة 95%. السحب بالـ lesion مش بالصورة، لأن صور نفس الـ lesion مش مستقلة عن بعض. وللفرق بين حالتين السحب «paired»: نفس الـ lesions للحالتين.

**2) عيلتين من المقارنات.**

| العيلة | فيها إيه | التصحيح |
|---|---|---|
| **تأكيدية** | مقارنة واحدة: **C ضد D على balanced accuracy** | Holm |
| استكشافية | كل الباقي: C ضد B، C ضد A، B ضد A، D ضد A، D ضد B، كل المقاييس التانية، والـ recall لكل class | Benjamini–Hochberg |

المقارنة التأكيدية مكتوبة قبل ما أي نتيجة تتشاف. إضافة المقاييس الجديدة كبّرت العيلة الاستكشافية بس، وما لمستش التأكيدية.

**3) C ضد D بتختبر إيه وما بتختبرش إيه.** بتختبر **أنهي صور** بس، لأن العدد متساوي. **كام صورة** بتتقري من C ضد A و C ضد B، ومن عدد C جنب منحنى E4. دي قراءات استكشافية، والإحصائية التأكيدية للكمية لسه قرار عند المشرف.

**4) مصدرين للتوقعات.**

- `--source classifier_val`: مكتوب عليه **MONITORING** في كل مكان. مش التقييم النهائي ومش بيقرر حاجة.
- `--source final`: على `final_eval_heldout`. **مش متاح دلوقتي**: القسم ده اتقرا مرة في v1، وسياسة استخدامه في v2 مستنية قرار المشرف.

**5) عدم اليقين من الـ seeds.** الفاصل اللي فوق بيسحب lesions بس، فتغيّر النتيجة من seed لـ seed ما بيدخلش فيه، وده تغيّر كبير في الـ recipe ده. عشان كده فيه تحليلين معلنين مسبقًا بيتكتبوا **جنب** الفاصل الأساسي مش بداله: اختبار Welch على balanced accuracy لكل seed (الوحدة هي التدريبة)، و bootstrap على مستويين بيسحب lesions و seeds مع بعض.

""" + table([
        ("المقارنة كلها (فواصل، فروق، العيلتين)", ref(cmp_py, "compare")),
        ("فاصل الثقة بالـ lesion", ref("scripts/utils/metrics.py", "patient_level_bootstrap")),
        ("فرق paired بين حالتين", ref("scripts/utils/metrics.py", "paired_bootstrap_difference")),
        ("Holm", ref("scripts/utils/metrics.py", "holm_bonferroni")),
        ("Benjamini–Hochberg", ref("scripts/utils/metrics.py", "benjamini_hochberg")),
        ("تجميع توقعات `classifier_val` وتشغيل المقارنة", ref("scripts/followup/ham10000_asism_v2_compare.py", "run")),
        ("تحليل الثبات على الـ seeds", ref("scripts/eval/ham10000_seed_robustness.py", "robustness")),
    ]) + """

**الداخل:** ملفات التوقعات لكل حالة × seed.
**الخارج:** تقرير JSON وجدول CSV فيهم كل مقياس بفاصله، وكل فرق بـ p المصحّحة وحجم الأثر.
"""))
    add(markdown(excerpt(cmp_py, "compare", start="confirmatory, exploratory = {}, {}", end="exploratory[f\"{left}_vs_{right}:{name}\"] = _difference(left, right, metric_fn)")))
    return cells


CLOSING = """
---
## القواعد اللي المشروع ماشي بيها

- أي حد (threshold) بيتكتب قبل التشغيل اللي هيتحكم عليه بيه، وما بيترخّاش بعده.
- أي تغيير هو تعديل مكتوب بتاريخه **قبل** التشغيل اللي بعده.
- النتايج القديمة ما بتتعدّلش وما بتتمسحش؛ الجديد بيروح فولدرات جديدة.
- `final_eval_heldout` ما بيتقراش من أي كود جديد، و `classifier_val` ما بيتستخدمش في اختيار الصور.
- النتيجة السلبية بتتكتب زي ما هي.

## إيه اللي ممكن يوقّف v2، وكله نتيجة مش خطأ

| فين | الشرط | لو ما اتحققش |
|---|---|---|
| G1 | ثبات متوسطات المجموعات ≥ 0.80 | مفيش تعلّم من القياس |
| accept | Spearman ≥ 0.50، p ≤ 0.05، وأحسن من «الحجم والـ class بس» | الترتيب المتعلَّم ما بيتستخدمش |
| select | النتيجة `all` أو `none` | مفيش C و D؛ بيتدرب A و B بس |

**قراءتي أنا، مش نتيجة:** التشخيص القديم لقى إن القياس بالـ recipe الصغير ضوضاؤه عالية. v2 بيختلف في إن الأحجام متنوعة وكل مجموعة بتتكرر 5 مرات، بس ده ما يضمنش إن G1 يعدّي. فشل G1 أو accept احتمال حقيقي.

## اللي لسه مفتوح

| القرار | عند مين |
|---|---|
| الموافقة على القياس (GPU) | ولاء |
| الموافقة على Stage 4 | ولاء، بعد قراءة الاختيار |
| سياسة الـ test set في v2 | المشرف |
| الإحصائية التأكيدية للكمية | المشرف |
| نطاق الرسالة | المشرف |
"""


def build() -> dict:
    sections = build_sections()
    toc = "## المحتويات\n\n" + "\n".join(
        f"{i}. [{title}](#{anchor})" for i, (anchor, title) in enumerate(SECTIONS, start=1))
    overview = ("## كل مرحلة في سطر\n\n| # | المرحلة | بتاخد | بتطلّع |\n|---|---|---|---|\n"
                "| 1 | Splits | HAM10000 الخام | ستّ أقسام مجمّدة |\n"
                "| 2 | LoRA | `gen_train` | `ham-lora-v1` |\n"
                "| 3 | Generation | الـ LoRA | 3,168 صورة |\n"
                "| 4 | Pool / Safety | الصور وإشاراتها | الـ pool الآمن (3,168) |\n"
                "| 5 | Signals | الصور الصناعية | أربع أرقام لكل صورة |\n"
                "| 6 | ASISM V2 | الـ pool بإشاراته | C و D |\n"
                "| 7 | Supervision | 200 مجموعة × 5 seeds | درجة منفعة لكل مجموعة، G1، accept |\n"
                "| 8 | Bootstrap | القياسات | 200 ranker وحد أدنى لكل صورة |\n"
                "| 9 | Stopping | الحدود الدنيا | أنهي صور وكام |\n"
                "| 10 | C / D | الاختيار | ملفات C و D |\n"
                "| 11 | E4 (مستقلة) | الـ pool | منحنى الكمية |\n"
                "| 12 | Stage 4 | A, B, C, D | 80 classifier |\n"
                "| 13 | Metrics | التوقعات | سبع مقاييس |\n"
                "| 14 | Comparison | المقاييس | فواصل ثقة و p مصحّحة |")
    check = CHECK_CODE.replace("__CODE_MAP__", json.dumps(CODE_MAP, indent=4, ensure_ascii=False))
    cells = [markdown(INTRO), markdown(toc), markdown(DIAGRAM), markdown(overview), markdown(BACKGROUND),
             markdown(CHECK_INTRO), code(check), *sections, markdown(CLOSING)]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    notebook = build()
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"  {len(notebook['cells'])} cells, {len(CODE_MAP)} files and "
          f"{sum(map(len, CODE_MAP.values()))} functions and classes referenced, all found")
