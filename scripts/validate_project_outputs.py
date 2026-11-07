"""Validate required artifacts and result integrity without altering raw data."""
import csv, math, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=['PROJECT_AUDIT.md','EXPERIMENTAL_METHODOLOGY.md','EXPERIMENT_COMPLETION_MATRIX.md','METHODOLOGY_GAPS.md','IMPLEMENTATION_GAPS.md','HARDWARE_BENCHMARK_REPORT.md','LLM_EVALUATION_REPORT.md','LITERATURE_VALIDATION_REPORT.md','STATISTICAL_ANALYSIS_REPORT.md','MARGINAL_GAIN_ANALYSIS.md','FINAL_RESEARCH_REPORT.md','REMAINING_WORK.md','EXECUTION_LOG.md','FINAL_COMPLETION_MATRIX.md','MACOS_SETUP.md','MACOS_EXECUTION_LOG.md','REPRODUCIBILITY.md','RESULT_PROVENANCE.md','IDENTIFIER_AUDIT.md','IDENTIFIER_RECONCILIATION.csv','EXPERIMENT_RUN_REGISTRY.csv','FINAL_INDEPENDENT_AUDIT.md','DATA_INVENTORY.md','DATA_QUALITY_REGISTER.md','FIGURE_CATALOGUE.md','scripts/plot_final_figures.py','scripts/build_final_report_pdf.py','scripts/build_verified_dataset.py','scripts/audit_identifiers.py','results/processed/canonical_results.csv','results/processed/verified_canonical_results.csv','results/processed/data_quality_register.csv','results/processed/figure_catalogue.csv','results/processed/identifier_audit.csv','results/processed/identifier_findings.csv','results/processed/foreign_key_audit.csv','results/llm/macos_full_wikitext2_evaluations.csv','results/llm/harness_full_summary.csv','results/literature/literature_field_evidence.csv','reports/FINAL_RESEARCH_REPORT.pdf']
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
    rows=load_csv('results/processed/canonical_results.csv');ids=[r.get('canonical_id') for r in rows];eids=[r.get('evaluation_id') for r in rows]
    if len(ids)!=len(set(ids)):errors.append('Duplicate canonical_id in canonical_results.csv')
    if len(eids)!=len(set(eids)) or any(not x for x in eids):errors.append('Canonical evaluation_id values are missing or non-unique')
    # Check every indexed row against its exact raw source row. raw_record_json
    # intentionally stores the original, malformed Double DQN source mapping.
    source_cache={}
    for r in rows:
        rel=r.get('source_file','');rownum=int(r.get('source_row') or 0)
        if not rel or not rownum:errors.append(f'Canonical source locator missing: {r.get("canonical_id")}');continue
        if rel not in source_cache:
            p=ROOT/rel
            try:source_cache[rel]=load_csv(rel)
            except Exception as e:errors.append(f'Cannot load canonical source {rel}: {e}');continue
        source_rows=source_cache[rel];idx=rownum-2
        if idx<0 or idx>=len(source_rows):errors.append(f'Canonical source row out of range: {rel}:{rownum}');continue
        try:embedded=json.loads(r.get('raw_record_json','{}'))
        except Exception as e:errors.append(f'Invalid embedded source JSON {rel}:{rownum}: {e}');continue
        if embedded!=source_rows[idx]:errors.append(f'Canonical/raw source row mismatch: {rel}:{rownum}')
        for key in ('configuration_id','checkpoint_path'):
            value=(r.get(key) or '').replace('\\','/')
            if value and not (ROOT/value).is_file():errors.append(f'Broken canonical {key} reference: {rel}:{rownum}: {value}')
    print(f'INFO: Canonical result index rows: {len(rows)} (all rows checked against source path and row)')
if (ROOT/'results/processed/verified_canonical_results.csv').is_file():
    rows=load_csv('results/processed/verified_canonical_results.csv')
    if len(rows)!=2612:errors.append(f'Unexpected verified index row count: {len(rows)}')
    if any(r.get('audit_eligibility') not in ('eligible_within_protocol','excluded','excluded_as_alias','excluded_from_completed_analysis') for r in rows):errors.append('Verified index has an unknown audit eligibility value')
    scopes={}
    for r in rows:
        key=(r.get('source_file'),r.get('experiment_id'))
        if key[1]:scopes[key]=scopes.get(key,0)+1
    if any(v>1 for v in scopes.values()):errors.append('Verified canonical experiment_id is repeated within a source-file scope')
    if sum(r.get('audit_disposition')=='duplicate_checkpoint_reuse' for r in rows)!=15:errors.append('Verified canonical index must mark 15 duplicate K-Means copies')
if (ROOT/'IDENTIFIER_RECONCILIATION.csv').is_file():
    rows=load_csv('IDENTIFIER_RECONCILIATION.csv')
    if len(rows)!=95:errors.append(f'Unexpected identifier reconciliation count: {len(rows)}')
    if any(not r.get('validation_status') for r in rows):errors.append('Identifier reconciliation has unvalidated correction')
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
    if len(rows)!=2417:errors.append(f'Unexpected marginal analysis count: {len(rows)}')
if (ROOT/'results/processed/strengthened_test_statistics.csv').is_file():
    rows=load_csv('results/processed/strengthened_test_statistics.csv')
    if len(rows)!=169:errors.append(f'Unexpected test endpoint count: {len(rows)}')
if (ROOT/'results/processed/data_size_scaling_summary.csv').is_file():
    rows=load_csv('results/processed/data_size_scaling_summary.csv')
    if any(r.get('budget_value')!='5' for r in rows):errors.append('Dataset-size summary must use matched five-epoch endpoints only')
    if any(int(r.get('n_seeds') or 0)!=3 for r in rows):errors.append('Dataset-size summary has an unexpected seed count')
if (ROOT/'results/processed/figure_catalogue.csv').is_file():
    rows=load_csv('results/processed/figure_catalogue.csv')
    if len(rows)!=25:errors.append(f'Expected 25 final figures, found {len(rows)}')
    for r in rows:
        for key in ('file_png','file_vector'):
            if not (ROOT/r[key]).is_file():errors.append(f'Missing catalogued figure: {r[key]}')
        if r.get('file_vector','').split('.')[-1] not in ('svg','pdf'):errors.append(f'Figure vector format missing: {r.get("file_vector")}')
        if not (ROOT/r.get('analysis_script','')).is_file():errors.append(f'Missing figure analysis script: {r.get("analysis_script")}')
if (ROOT/'FINAL_RESEARCH_REPORT.md').is_file():
    report=(ROOT/'FINAL_RESEARCH_REPORT.md').read_text(encoding='utf-8')
    expected=[f'## {i}.' for i in range(1,37)]
    if any(h not in report for h in expected):errors.append('Final report is missing one or more of its 36 required sections')
    for phrase in ('0 of 50','3,614','2,417','169','1,122.1625'):
        if phrase not in report:errors.append(f'Final report lacks reconciled key value: {phrase}')
pdf=ROOT/'reports/FINAL_RESEARCH_REPORT.pdf'
if pdf.is_file() and (pdf.stat().st_size<100000 or pdf.read_bytes()[:5]!=b'%PDF-'):errors.append('Final report PDF is invalid or unexpectedly small')
if (ROOT/'results/processed/identifier_findings.csv').is_file():
    findings=load_csv('results/processed/identifier_findings.csv')
    if any(r.get('severity')=='error' for r in findings):errors.append('Identifier audit contains unresolved error-level findings')
if (ROOT/'results/processed/foreign_key_audit.csv').is_file():
    refs=load_csv('results/processed/foreign_key_audit.csv')
    if any(r.get('validation')=='missing' for r in refs):errors.append('Identifier audit contains missing file foreign keys')
print(f'CSV validation: {len(errors)} errors, {len(warnings)} warnings')
for x in errors: print('ERROR:',x)
for x in warnings[:30]: print('WARN:',x)
raise SystemExit(1 if errors else 0)
