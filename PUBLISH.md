# Publish this leaf as a public GitHub repo

The translator is a standalone MIT tree. The Cursor cloud GitHub App on Erisv2 **cannot create new repositories** (installation scope is `Bucktzu-Dev/Erisv2` only).

From a machine logged in as **Bucktzu-Dev** (or with a PAT that can create repos):

```bash
cd neuralese-translator
gh repo create Bucktzu-Dev/neuralese-translator \
  --public \
  --description "Neuralese to English Translator — reconstruct inner symbol alphabets, gloss to English, fail closed unless codes stay decodable." \
  --source=. \
  --remote=origin \
  --push
```

If the folder is already a git repo nested in Erisv2, use a fresh copy:

```bash
git clone --depth 1 https://github.com/Bucktzu-Dev/Erisv2.git
cd Erisv2/neuralese-translator
# or copy this directory out, then:
rm -rf .git
git init -b main
git add .
git commit -m "Publish Neuralese to English Translator"
gh repo create Bucktzu-Dev/neuralese-translator --public --source=. --remote=origin --push
```

After it exists, reviewers clone:

```bash
git clone https://github.com/Bucktzu-Dev/neuralese-translator.git
cd neuralese-translator
pip install -e ".[dev]"
python -m pytest -q
```
