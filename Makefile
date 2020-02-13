start.restapi:
	@python3 rest.py

curl.test:
	@curl -F "img=@1.jpg" http://0.0.0.0:5000/ocr