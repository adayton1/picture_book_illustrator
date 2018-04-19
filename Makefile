UNAME := $(shell uname)
ILLUSTRATOR_TENSORFLOW_MODEL_DIR ?= $(CURDIR)/deps/tensorflow_models
ILLUSTRATOR_PYTHONPATH ?= $(PYTHONPATH):$(CURDIR):$(ILLUSTRATOR_TENSORFLOW_MODEL_DIR):$(ILLUSTRATOR_TENSORFLOW_MODEL_DIR)/research:$(ILLUSTRATOR_TENSORFLOW_MODEL_DIR)/research/im2txt:$(ILLUSTRATOR_TENSORFLOW_MODEL_DIR)/research/slim
ILLUSTRATOR_MAIN ?= illustrator/illustrate.py

ifeq ($(UNAME), Linux)
	install_cmd = sudo apt install protobuf-compiler
endif
ifeq ($(UNAME), Darwin)
	install_cmd = brew install protobuf
endif

all: protobuf python cocoapi

update:
	git submodule update --merge

protobuf: update
	$(install_cmd)
	cd ./deps/tensorflow_models/research/ && \
		protoc ./object_detection/protos/*.proto --python_out=.

python:
	pip3 install numpy Cython wheel
	pip3 install -r requirements.txt
	test -d $(shell python3 -c 'import site; print(list(filter(lambda x: "site-packages" in x, site.getsitepackages()))[0])')/en_core_web_lg || python3 -m spacy download en_core_web_lg

cocoapi: update
	$(MAKE) -C ./deps/cocoapi/PythonAPI
	cp -r deps/cocoapi/PythonAPI/pycocotools $(ILLUSTRATOR_TENSORFLOW_MODEL_DIR)/research

run:
	@PYTHONPATH=$(ILLUSTRATOR_PYTHONPATH) python3 $(ILLUSTRATOR_MAIN) $(ILLUSTRATOR_INPUT)

.PHONY: cocoapi update protobuf python run all