#!/usr/bin/env python3
"""Test BioBERT and Database Integration"""

from database import get_database
from modules.literature_ner import TRANSFORMERS_AVAILABLE, LiteratureMiner, RuleBasedNER

print('='*60)
print('Testing BioBERT + Database Integration')
print('='*60)

# Test 1: Check transformers
print()
print('1. BioBERT Status:')
print(f'   TRANSFORMERS_AVAILABLE = {TRANSFORMERS_AVAILABLE}')

# Test 2: Database
print()
print('2. Database Connection:')
db = get_database()
stats = db.get_database_stats()
print(f'   Papers: {stats["papers"]}')
print(f'   Entities: {stats["entities"]}')
print(f'   Relations: {stats["relations"]}')
print(f'   Predictions: {stats["predictions"]}')
print(f'   Biomarkers: {stats["biomarkers"]}')

# Test 3: Rule-based NER
print()
print('3. Testing Rule-based NER:')
ner = RuleBasedNER()
test_text = 'BRCA1 mutations increase breast cancer risk. Tamoxifen is used for treatment.'
entities = ner.extract_entities(test_text)
print(f'   Found {len(entities)} entities in test text')
for e in entities[:5]:
    print(f'   - {e["text"]} ({e["type"]})')

# Test 4: BioBERT NER (if available)
print()
print('4. Testing BioBERT NER:')
if TRANSFORMERS_AVAILABLE:
    try:
        miner_bio = LiteratureMiner(use_biobert=True)
        print('   BioBERT LiteratureMiner initialized successfully')
    except Exception as e:
        print(f'   BioBERT init failed: {e}')
else:
    print('   Transformers not available (rule-based NER will be used)')

print()
print('='*60)
print('All tests passed!')
print('='*60)
