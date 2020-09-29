# TF Datasets Tests


## First Install

1. Download from git
```
git clone https://github.com/zwerb/tf_datasets_tests.git tf_datasets_tests/

cd tf_datasets_tests/
```

2. a. Download virtualenv (if you haven't)
```
pip install virtualenv
```

2. b. Setup Virtual env
```
python3 -m venv ./
```

3. Start VirtualEnv and Install dependencies 
```
source bin/activate

pip install -r requirements.txt 
```

## Startup
```
source bin/activate

jupyter lab
```

### Other Notes / Commands

#### Convert from Jupyter Notebook 

-without comments

```
jupyter nbconvert --to python iris_walkthrough_copy.ipynb \
--TemplateExporter.exclude_markdown=True \
--TemplateExporter.exclude_output_prompt=True \
--TemplateExporter.exclude_input_prompt=True
```
