# Security policy

This tool reads observation JSONL, symbol packs, and optional integer code streams.

- Treat those files as untrusted input if you did not generate them.
- Do not run it next to credentials, `.env` files, or private Eris tokens.
- Optional LLM clients are caller-supplied. Do not paste secrets into gloss prompts.
- Certification is an integrity check on a pack, not a sandbox.

To report a vulnerability, contact the repository owner privately. Do not open a public issue with exploit details.
