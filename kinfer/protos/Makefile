# Makefile

all: kinfer_pb2.py

%_pb2.py: %.proto
	protoc --proto_path=. --python_out=. --mypy_out=. -I. $<

clean:
	rm -f *.py *.pyi
