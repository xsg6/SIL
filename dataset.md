Dataset Preparation
We strictly evaluate on real-world benchmarks. Please download the datasets via the following links:

2.1 ConCode (Text-to-Java Generation)
ConCode is available via HuggingFace Datasets (code_x_glue_ct_code_to_text).
Our src/dataloader.py will automatically download this upon first run.

Source: CodeXGLUE ConCode

2.2 Defects4J (Automated Program Repair)
For Automated Program Repair (APR), we use the Defects4J benchmark.
Due to the nature of executable environments, please follow the official instructions to extract buggy functions, or download community-extracted JSONL files.

Official Repository: rjust/defects4j

Place your extracted defects4j_test.jsonl into the ./data/defects4j/ directory.
