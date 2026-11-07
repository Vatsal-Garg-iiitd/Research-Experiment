"""Validate required artifacts and result integrity without altering raw data."""
import csv, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=['PROJECT_AUDIT.md','EXPERIMENTAL_METHODOLOGY.md','EXPERIMENT_COMPLETION_MATRIX.md','METHODOLOGY_GAPS.md','IMPLEMENTATION_GAPS.md','HARDWARE_BENCHMARK_REPORT.md','LLM_EVALUATION_REPORT.md','LITERATURE_VALIDATION_REPORT.md','STATISTICAL_ANALYSIS_REPORT.md','MARGINAL_GAIN_ANALYSIS.md','FINAL_RESEARCH_REPORT.md','REMAINING_WORK.md','EXECUTION_LOG.md','FINAL_COMPLETION_MATRIX.md','results/processed/canonical_results.csv']
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
print(f'CSV validation: {len(errors)} errors, {len(warnings)} warnings')
for x in errors: print('ERROR:',x)
for x in warnings[:30]: print('WARN:',x)
raise SystemExit(1 if errors else 0)
