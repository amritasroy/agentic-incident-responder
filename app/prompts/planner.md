Given the incident description and available tools, produce a short JSON plan:
- TRIAGE
- EVIDENCE (timeseries, logs)
- HYPOTHESES (ranked)
- VERIFY (extra evidence or human ask)
- REMEDIATION (candidate fix)
Return JSON with: {"steps":[], "stop_condition":"...", "confidence_hint":0-1}.
