# setup a virtual environment
python3 -m venv venv
# activate the virtual environment
source venv/bin/activate
# install the requirements
pip install -r requirements.txt
# obtain the token list and special token correspondence table
wget -P ./data https://github.com/tanreinama/Japanese-BPEEncoder_V2/raw/master/ja-swe24k.txt
wget -P ./data https://github.com/tanreinama/Japanese-BPEEncoder/raw/master/emoji.json
