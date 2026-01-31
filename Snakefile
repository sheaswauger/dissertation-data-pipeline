rule all:
    input:
        "data/processed/demo_tagged.parquet"

rule fetch_demo:
    output:
        "data/raw/demo.jsonl"
    shell:
        "python -m src.legal_pipeline.retrieve --provider openstates --state {wildcards.state if False else 'CO'} --limit 10 --out {output}"

rule normalize_demo:
    input: "data/raw/demo.jsonl"
    output: "data/processed/demo.parquet"
    shell:
        "python scripts/normalize_json.py {input} {output}"

rule tag_demo:
    input: "data/processed/demo.parquet"
    output: "data/processed/demo_tagged.parquet"
    shell:
        "python scripts/tag_metadata.py {input} {output}"
