.PHONY: setup demo eval clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

demo:
	. .venv/bin/activate && python app/main.py --scenario bearing_wear_03 --verbose

eval:
	. .venv/bin/activate && python eval/harness.py

clean:
	rm -rf .venv app/__pycache__ eval/__pycache__ models/* eval/results/* dist build
