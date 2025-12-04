School Safety Legislation Study Workflow
This diagram shows the complete data pipeline from initial setup to final enriched dataset.
graph TD
    Start([Setup Environment]) --> Setup[Install Dependencies<br/>./setup.sh]
    Setup --> Config[Configure API Keys<br/>API_keys.txt]
    
    Config --> Params[Define Search Parameters<br/>━━━━━━━━━━━━━━━━<br/>States: All 50<br/>Years: 2010-2025<br/>Terms: school shooting,<br/>school safety, school lockdown,<br/>school emergency]
    
    Params --> Collect[Data Collection<br/>━━━━━━━━━━━━━━━━<br/>legiscan_collector.py<br/>legiscan_search.py<br/>legiscan_bulk.py]
    
    Collect --> Raw[(Raw Data<br/>Multiple formats)]
    
    Raw --> Normalize[Normalize Data<br/>━━━━━━━━━━━━━━━━<br/>legiscan_normalizer.py]
    
    Normalize --> Combined[(legiscan_combined.csv<br/>Normalized dataset)]
    
    Combined --> Enrich[Enrich Missing Data<br/>━━━━━━━━━━━━━━━━<br/>legiscan_enricher.py<br/>Uses: LegiScan + OpenStates APIs]
    
    Enrich --> Final[(legiscan_enriched.csv/xlsx<br/>Final Dataset)]
    
    Final --> End([Analysis Ready])
    
    %% Supporting files
    Utils[src/utils.py<br/>src/text_processor.py] -.-> Collect
    Utils -.-> Normalize
    Utils -.-> Enrich
    
    Prompt[prompt_2.txt<br/>Enrichment specs] -.-> Enrich
    
    API[src/legiscan_api.py<br/>API wrapper] -.-> Collect
    
    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style Params fill:#fff4e1
    style Raw fill:#f0f0f0
    style Combined fill:#f0f0f0
    style Final fill:#e1e5f5
Pipeline Steps
	1	Setup Environment - Install dependencies and create virtual environment
	2	Configure API Keys - Add LegiScan and OpenStates API credentials
	3	Define Parameters - Set search scope (states, date range, keywords)
	4	Data Collection - Fetch raw bill data from LegiScan API
	5	Normalize Data - Standardize format across all collected bills
	6	Enrich Data - Fill missing fields using multiple API sources
	7	Final Dataset - Analysis-ready enriched dataset
Key Files
	•	Data Collection: legiscan_collector.py, legiscan_search.py, legiscan_bulk.py
	•	Processing: legiscan_normalizer.py, legiscan_enricher.py
	•	Support: src/utils.py, src/text_processor.py, src/legiscan_api.py
	•	Config: API_keys.txt, prompt_2.txt
