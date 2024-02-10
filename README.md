# Website for FoodAI

## Before You Start

### Download Model

Download [models.zip](https://drive.google.com/drive/folders/10BiLqVYzLJveoSXLv_BGgIwb1WzubwD0?usp=sharing), unzip it and put it the root.

### Setup Environment

Create a clean new environment if you're not familiar with what you have in your current Python environment. If you are using conda:

``` bash
conda create -n foodai python=3.7

source activate foodai
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

## Deploy

> Currently not for production deployment, large number of connections would potentially make the server crash.

### Local Debug Mode

This mode will refresh the page without rerunning the command below.

```bash
python app.py --debug=1
```

### Public Testing

```bash
python app.py --debug=0
```

If you have permission problems (mainly because you are trying to expose a port to public), run with `sudo`.
