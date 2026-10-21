"""Streamlit app for annotating the criteria in report.csv.

Pick a json file in the sidebar; its rows are laid out like report.csv (one row per learning
objective, the CSV fields as columns, multi-value text kept together and delimited by [SEP]).
Every criterion in the criteria columns shows the model's label (the ":1" / ":0" suffix in
report.csv) and a rater input:
-1 = unmarked (default), 1 = met, 0 = unmet.

"Export" downloads reportANN.csv: report.csv plus a "rater1" column holding the rater's input
for every criterion of the row, one per line as "<criterion>:<value>" (main criteria first, then
second criteria, in the same order as those columns). "Import" uploads a reportANN.csv exported
earlier and restores its rater1 values, so annotation can continue where it stopped.

Annotations live only in the browser session: on a hosted app (e.g. Streamlit Community Cloud)
they are lost when the tab is closed or the session times out, so export regularly. If a
reportANN.csv sits next to this script, it is loaded on start.

The app asks for a password first: set `password = "..."` in .streamlit/secrets.toml (locally)
or in the app's Secrets settings on Streamlit Community Cloud.

Usage:
    streamlit run annotate_report.py
    streamlit run annotate_report.py -- report.csv -o reportANN.csv
"""

import argparse
import csv
import hmac
import io
from pathlib import Path

import streamlit as st

SEP = " [SEP] "
UNMARKED = -1
RATER = "rater1"
KINDS = {"main": ("main criteria", "Main criteria"), "second": ("second criteria", "Follow-up criteria")}
LABELS = {-1: "−1", 1: "1", 0: "0"}
TEXT_COLS = ["json file name", "learning objective", "question", "reference answer", "student answer"]
WIDTHS = [1.3, 1.3, 2.4, 2.4, 2.4, 3, 3]
SEP_HTML = ' <b style="color:#2f5bd3">[SEP]</b> '
HERE = Path(__file__).parent


def md(text):
    """Escape Markdown so CSV text renders literally (e.g. "$" would start LaTeX)."""
    return "".join("\\" + ch if ch in "\\`*_{}[]()#+-.!$|<>~" else ch for ch in str(text))


def split(cell):
    """SEP-joined CSV cell -> list (empty cell -> [])."""
    return [s.strip() for s in cell.split(SEP)] if cell.strip() else []


def parse_labels(cell):
    """'<criterion>:1' lines -> [(criterion, label)]; label is None when missing."""
    out = []
    for line in filter(None, (l.strip() for l in (cell or "").splitlines())):
        text, _, label = line.rpartition(":")
        out.append((text, int(label)) if label.lstrip("-") in ("0", "1") else (line, None))
    return out


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def ident(row):
    """Identifies a report row independent of its position."""
    return row.get("json file name", ""), row.get("learning objective", "")


@st.cache_data
def load(csv_path):
    """report.csv -> (columns, rows, {file: [lo, ...]}) with each lo holding its criteria."""
    columns, rows = read_csv(csv_path)
    files = {}
    for r, row in enumerate(rows):
        questions, refs = split(row["question"]), split(row["reference answer"])
        los = files.setdefault(row["json file name"], [])
        los.append({
            "row": r, "id": ident(row), "n": len(los) + 1, "text": row["learning objective"],
            "questions": [(q, refs[i] if i < len(refs) else "N/A") for i, q in enumerate(questions)],
            "answers": split(row["student answer"]),
            "criteria": {kind: parse_labels(row[col]) for kind, (col, _) in KINDS.items()},
        })
    return columns, rows, files


def key(lo, kind, i):
    return (lo["row"], kind, i)


def row_keys(lo):
    """[(widget key, criterion text)] in rater1 order: main criteria, then second criteria."""
    return [(key(lo, kind, i), text) for kind in KINDS for i, (text, _) in enumerate(lo["criteria"][kind])]


def init_state(files, out_path):
    """Fill st.session_state.ann {criterion key: value}; resume from an existing reportANN.csv.

    Annotations live in their own dict, not in widget state: Streamlit drops a widget's state
    as soon as it is not drawn, which would wipe a file's inputs when switching files."""
    if st.session_state.get("ready"):
        return
    st.session_state.ann = {k: UNMARKED for los in files.values() for lo in los for k, _ in row_keys(lo)}
    if out_path.exists():
        apply_saved(read_csv(out_path)[1], files)
    st.session_state.ready, st.session_state.dirty = True, False


def apply_saved(saved_rows, files):
    """Reset st.session_state.ann to the rater1 values of reportANN.csv rows.

    Rows are matched by position when json file name + learning objective agree, otherwise by
    those two fields. Returns (report rows matched, criteria with a rating)."""
    by_id = {}
    for row in saved_rows:
        by_id.setdefault(ident(row), row)
    ann, matched = st.session_state.ann, 0
    for los in files.values():
        for lo in los:
            r = lo["row"]
            row = saved_rows[r] if r < len(saved_rows) and ident(saved_rows[r]) == lo["id"] else by_id.get(lo["id"])
            matched += row is not None
            prev = parse_labels(row.get(RATER, "")) if row else []
            by_text = {t: v for t, v in prev}
            for i, (k, text) in enumerate(row_keys(lo)):
                # Same position and text -> reuse; otherwise fall back to matching by text.
                v = prev[i][1] if i < len(prev) and prev[i][0] == text else by_text.get(text)
                ann[k] = v if v in LABELS else UNMARKED
    return matched, sum(v != UNMARKED for v in ann.values())


def import_csv(files):
    """file_uploader callback: replace the current annotations with an uploaded reportANN.csv."""
    upload = st.session_state.get("import_file")
    if upload is None:
        return
    try:
        reader = csv.DictReader(io.StringIO(upload.getvalue().decode("utf-8-sig"), newline=""))
        saved_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as e:
        st.session_state.import_msg = ("error", f"Could not read {upload.name}: {e}")
        return
    if RATER not in (reader.fieldnames or []):
        st.session_state.import_msg = ("error", f"{upload.name} has no '{RATER}' column.")
        return
    matched, rated = apply_saved(saved_rows, files)
    # Radio widgets keep their own state; drop it so they redraw from the imported values.
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith("w|")]:
        del st.session_state[k]
    st.session_state.dirty = False
    total = sum(len(los) for los in files.values())
    level = "success" if matched == total else "warning"
    st.session_state.import_msg = (level, f"Imported {upload.name}: {matched} / {total} rows matched, "
                                          f"{rated} criteria rated.")


def counts(los):
    keys = [k for lo in los for k, _ in row_keys(lo)]
    return sum(st.session_state.ann[k] != UNMARKED for k in keys), len(keys)


def to_csv(columns, rows, files):
    """reportANN.csv as bytes: the report rows plus the rater1 column."""
    out_cols = columns + [RATER] if RATER not in columns else columns
    out_rows = list(rows)
    for los in files.values():
        for lo in los:
            out_rows[lo["row"]] = {**rows[lo["row"]], RATER: "\n".join(
                f"{text}:{st.session_state.ann[k]}" for k, text in row_keys(lo))}
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=out_cols, restval="", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)
    return buf.getvalue().encode("utf-8-sig")


def mark_saved():
    st.session_state.dirty = False


def set_value(k, widget):
    st.session_state.ann[k] = st.session_state[widget]
    st.session_state.dirty = True


def step(names, d):
    i = names.index(st.session_state.file) + d
    if 0 <= i < len(names):
        st.session_state.file = names[i]


def cell(text):
    """Raw CSV cell as Markdown: text escaped, [SEP] delimiters highlighted."""
    return SEP_HTML.join(md(part) for part in text.split(SEP.strip())) if text.strip() else "—"


def criteria_cell(lo, kind):
    """Criteria column: each criterion with its model label and the rater input next to it."""
    items = lo["criteria"][kind]
    if not items:
        st.markdown("—")
    for i, (text, model) in enumerate(items):
        k = key(lo, kind, i)
        human, widget = st.session_state.ann[k], "w|%s|%s|%s" % k
        flag = " ⚠️" if human != UNMARKED and human != model else ""
        tag = f":{'green' if model == 1 else 'red'}[model: {model if model is not None else 'N/A'}]"
        st.markdown(f"{md(text)} {tag}{flag}")
        st.radio(text, list(LABELS), index=list(LABELS).index(human), key=widget, format_func=LABELS.get,
                 horizontal=True, label_visibility="collapsed", on_change=set_value, args=(k, widget))


def table(rows, los):
    """The file's rows in CSV layout: a header row, then one row per learning objective."""
    for col, name in zip(st.columns(WIDTHS), TEXT_COLS + [KINDS["main"][0], KINDS["second"][0]]):
        col.markdown(f"**{name}**")
    for lo in los:
        st.divider()
        cols, row = st.columns(WIDTHS), rows[lo["row"]]
        for col, name in zip(cols, TEXT_COLS):
            col.markdown(cell(row[name]), unsafe_allow_html=True)
        for col, kind in zip(cols[len(TEXT_COLS):], KINDS):
            with col:
                criteria_cell(lo, kind)


def check_password():
    """Block the app until the user enters the password set as `password` in Streamlit secrets.

    Locally: .streamlit/secrets.toml. On Streamlit Community Cloud: App settings > Secrets.
    Without a configured password the app stays locked."""
    if st.session_state.get("authenticated"):
        return
    try:
        # Checking first avoids Streamlit's own "No secrets found" error when there is no file.
        expected = st.secrets.get("password") if st.secrets.load_if_toml_exists() else None
    except Exception:
        expected = None
    if not expected:
        st.error('No password configured. Add `password = "..."` to .streamlit/secrets.toml '
                 "(or to the app's Secrets on Streamlit Cloud).")
        st.stop()
    st.title("Criteria annotation")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if hmac.compare_digest(entered.encode(), str(expected).encode()):
                st.session_state.authenticated = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=HERE / "report.csv")
    parser.add_argument("-o", "--output", type=Path, default=HERE / "reportANN.csv")
    args, _ = parser.parse_known_args()  # tolerate extra args passed by hosting platforms

    st.set_page_config(page_title="Criteria annotation", layout="wide")
    check_password()
    if not args.csv.exists():
        st.error(f"Report file not found: {args.csv.name}. Put report.csv next to {Path(__file__).name}.")
        st.stop()
    columns, rows, files = load(str(args.csv))
    rows = [dict(r) for r in rows]  # don't mutate the cached rows
    init_state(files, args.output)
    names = list(files)

    with st.sidebar:
        st.header("Criteria annotation")
        done, total = counts([lo for los in files.values() for lo in los])
        st.progress(done / total if total else 1.0, text=f"All files: {done} / {total} annotated")
        todo = st.checkbox("Unfinished files only")
        st.session_state.setdefault("file", names[0])
        progress = {n: counts(files[n]) for n in names}
        shown = [n for n in names if not todo or n == st.session_state.file or progress[n][0] < progress[n][1]]
        # Keep option labels static: Streamlit treats a selectbox whose labels change as a new
        # widget and resets it, so live counts here would jump back to the first file.
        st.selectbox("JSON file", shown, key="file")
        d, t = progress[st.session_state.file]
        st.caption(f"{'✓ ' if d == t else ''}This file: {d} / {t} · "
                   f"{sum(d == t for d, t in progress.values())} / {len(names)} files complete")
        b1, b2 = st.columns(2)
        b1.button("‹ Previous", on_click=step, args=(shown, -1), use_container_width=True)
        b2.button("Next ›", on_click=step, args=(shown, 1), use_container_width=True)

        st.divider()
        st.subheader("Save your work")
        st.download_button("⬇️ Export reportANN.csv", to_csv(columns, rows, files), file_name=args.output.name,
                           mime="text/csv", type="primary", on_click=mark_saved, use_container_width=True)
        st.caption("⚠️ Changes not exported yet" if st.session_state.dirty else "No unexported changes")
        st.file_uploader("⬆️ Import reportANN.csv", type="csv", key="import_file",
                         on_change=import_csv, args=(files,),
                         help="Replaces the current annotations with the rater1 values of an exported file.")
        if msg := st.session_state.pop("import_msg", None):
            getattr(st, msg[0])(msg[1])
        st.caption("Annotations are kept only while this tab is open. Export often, and import the "
                   "file to continue later.")

    st.title(st.session_state.file)
    st.caption("Rater input next to each criterion: −1 unmarked · 1 met · 0 unmet · ⚠️ differs from the model")
    table(rows, files[st.session_state.file])

main()
