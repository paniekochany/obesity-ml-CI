install:
	python3 -m pip install --upgrade pip &&\
	pip install -r requirements.txt
test:
	python3 -m pytest