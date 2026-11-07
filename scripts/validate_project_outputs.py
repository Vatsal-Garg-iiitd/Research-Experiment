"""Validate required artifacts and result integrity without altering raw data."""
import csv, math, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=['PROJECT_AUDIT.md','EXPERIMENTAL_METHODOLOGY.md','EXPERIMENT_COMPLETION_MATRIX.md','METHODOLOGY_GAPS.md','IMPLEMENTATION_GAPS.md','HARDWARE_BENCHMARK_REPORT.md','LLM_EVALUATION_REPORT.md','LITERATURE_VALIDATION_REPORT.md','STATISTICAL_ANALYSIS_REPORT.md','MARGINAL_GAIN_ANALYSIS.md','FINAL_RESEARCH_REPORT.md','REMAINING_WORK.md','EXECUTION_LOG.md','FINAL_COMPLETION_MATRIX.md','MACOS_SETUP.md','MACOS_EXECUTION_LOG.md','REPRODUCIBILITY.md','EXPERIMENT_RUN_REGISTRY.csv','FINAL_INDEPENDENT_AUDIT.md','results/processed/canonical_results.csv','results/llm/macos_full_wikitext2_evaluations.csv','results/llm/harness_full_summary.csv','results/literature/literature_field_evidence.csv']
errors=[];warnings=[]
for f in required:
    if not (ROOT/f).is_file(): errors.append(f'Missing deliverable: {f}')
statuses={'planned','implemented','pilot','running','completed','failed','superseded','resource_limited','literature_only','unknown'}
for p in sorted((ROOT/'results').rglob('*.csv')):
    if 'snapshots' in p.parts: continue
    try:
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames: errors.append(f'Empty/missing CSV header: {p.relative_to(ROOT)}'); continue
            records=list(reader)
        for n,r in enumerate(records,start=2):
            status=r.get('experiment_status') or r.get('status')
            if status and (not isinstance(status,str) or (status.lower() not in statuses and not status.lower().startswith('superseded_') and status.lower()!='local_measurement')): warnings.append(f'Unknown status {status!r}: {p.relative_to(ROOT)}:{n}')
            for key,val in r.items():
                if key.lower() in {'metric_value','evaluation_value','training_time_seconds','training_time','budget_value','test_accuracy','test_loss','validation_ari','mean_return','test_perplexity','test_cross_entropy'} and val not in (None,''):
                    try:
                        if not math.isfinite(float(val)): errors.append(f'Non-finite {key}: {p.relative_to(ROOT)}:{n}')
                    except ValueError: errors.append(f'Non-numeric {key}: {p.relative_to(ROOT)}:{n}')
    except Exception as e: errors.append(f'Cannot parse {p.relative_to(ROOT)}: {e}')
def load_csv(rel):
    with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

if (ROOT/'results/processed/canonical_results.csv').is_file():
    rows=load_csv('results/processed/canonical_results.csv');ids=[r.get('canonical_id') for r in rows]
    if len(ids)!=len(set(ids)):errors.append('Duplicate canonical_id in canonical_results.csv')
    print(f'INFO: Canonical result index rows: {len(rows)} (source CSVs remain authoritative)')
if (ROOT/'results/llm/macos_full_wikitext2_evaluations.csv').is_file():
    rows=load_csv('results/llm/macos_full_wikitext2_evaluations.csv');done=[r for r in rows if r.get('status')=='completed']
    if len(done)!=6:errors.append(f'Expected six completed full WikiText-2 checkpoint evaluations, found {len(done)}')
    if len({r.get('text_sha256') for r in done})!=1:errors.append('Full WikiText evaluations do not share a text hash')
    if any(r.get('context_length')!='1024' or r.get('stride')!='512' for r in done):errors.append('Full WikiText protocol mismatch')
if (ROOT/'results/llm/harness_full_summary.csv').is_file():
    rows=load_csv('results/llm/harness_full_summary.csv');tasks={r.get('task') for r in rows}
    if tasks!={'hellaswag','piqa','arc_easy'}:errors.append(f'Unexpected full harness task coverage: {sorted(tasks)}')
    for r in rows:
        if r.get('limit')!='none' or r.get('fewshot')!='0':errors.append(f'Full harness run has a limit or nonzero few-shot: {r.get("task")}')
        try:
            result=json.load((ROOT/r['result_json']).open(encoding='utf-8'))
            n=int(result['results'][r['task']]['sample_len'])
            if n!=int(r['evaluated_examples']):errors.append(f'Harness result count mismatch for {r["task"]}: summary={r["evaluated_examples"]}, JSON={n}')
            samples=list((ROOT/'results/llm/harness_full/pythia70m_step143000').glob(f'**/samples_{r["task"]}_*.jsonl'))
            if not samples or sum(1 for _ in samples[-1].open(encoding='utf-8'))!=n:errors.append(f'Harness per-sample log count mismatch for {r["task"]}')
        except Exception as e:errors.append(f'Cannot verify full harness JSON/sample counts for {r.get("task")}: {e}')
if (ROOT/'results/literature/literature_field_evidence.csv').is_file():
    rows=load_csv('results/literature/literature_field_evidence.csv');keys=[(r.get('model'),r.get('field')) for r in rows]
    classes={'reported','derived','estimated','uncertain'}
    if len(rows)!=300:errors.append(f'Expected 300 literature field rows, found {len(rows)}')
    if len(keys)!=len(set(keys)):errors.append('Duplicate model/field rows in literature evidence ledger')
    if any(r.get('evidence_class') not in classes for r in rows):errors.append('Unknown evidence class in literature ledger')
    if any(not r.get('source_url') or not r.get('locator') for r in rows):errors.append('Literature evidence row missing source URL or locator')
if (ROOT/'EXPERIMENT_RUN_REGISTRY.csv').is_file():
    rows=load_csv('EXPERIMENT_RUN_REGISTRY.csv');ids=[r.get('registry_id') for r in rows]
    if len(ids)!=len(set(ids)):errors.append('Duplicate registry_id in experiment run registry')
    if len(rows)!=3720:errors.append(f'Unexpected experiment run registry row count: {len(rows)}')
    for r in rows:
        source=r.get('source_file','')
        if source and not (ROOT/source).is_file():errors.append(f'Missing registry source path: {source}');break
if (ROOT/'FINAL_COMPLETION_MATRIX.md').is_file():
    matrix=(ROOT/'FINAL_COMPLETION_MATRIX.md').read_text(encoding='utf-8')
    if '**Strict objective completion: 0/50 configurations (0%).**' not in matrix:errors.append('Strict completion count missing or unexpectedly changed')
if (ROOT/'results/processed/strengthened_statistical_summaries.csv').is_file():
    rows=load_csv('results/processed/strengthened_statistical_summaries.csv')
    if len(rows)!=3614:errors.append(f'Unexpected grouped analysis count: {len(rows)}')
if (ROOT/'results/processed/strengthened_marginal_gains.csv').is_file():
    rows=load_csv('results/processed/strengthened_marginal_gains.csv')
    if len(rows)!=2407:errors.append(f'Unexpected marginal analysis count: {len(rows)}')
if (ROOT/'results/processed/strengthened_test_statistics.csv').is_file():
    rows=load_csv('results/processed/strengthened_test_statistics.csv')
    if len(rows)!=169:errors.append(f'Unexpected test endpoint count: {len(rows)}')
print(f'CSV validation: {len(errors)} errors, {len(warnings)} warnings')
for x in errors: print('ERROR:',x)
for x in warnings[:30]: print('WARN:',x)
raise SystemExit(1 if errors else 0)
