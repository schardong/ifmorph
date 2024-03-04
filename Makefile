.PHONY: all clean train_initial_states train_warp

all: train_initial_states train_warp

install:
	@pip install -r requirements.txt
	@pip install -e .

package:
	@rm -Rf niif/__pycache__/
	@tar -cjf dist.tbz niif/ warp-train.py warp-inference-vid.py warp-inference-image.py setup.py requirements.txt README.md create-initial-states.py align.py
	@echo "Package created at: dist.tbz"

clean:
#	@rm -Rf pretrained/*
#	@rm -Rf results/*
	@rm -Rf __pycache__
	@rm -Rf ifmorph/__pycache__

train_initial_states: data/frll
	@python create-initial-states.py --nsteps 1000 --device cuda:0 experiments/initial_state_rgb_large_im.yaml data/frll/001_03.jpg data/frll/002_03.jpg

train_warp: results/fm_frll-001-002/weigths.pth
	@echo "FRLL 001->002 warp trained"

results/fm_frll-001-002/weights.pth: pretrained/001_03.pth pretrained/002_03.pth
	@python warp-train.py experiments/tem/fm_frll-001-002.yaml

landmark_models/shape_predictor_68_face_landmarks_GTX.dat:
	@echo "Downloading DLib GTX 68 landmarks detection model"
	@curl --location --remote-header-name --remote-name https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2
	@bunzip2 shape_predictor_68_face_landmarks_GTX.dat.bz2
	@mkdir -p landmark_models
	@mv shape_predictor_68_face_landmarks_GTX.dat landmark_models/
	@rm -Rf shape_predictor_68_face_landmarks_GTX.dat.bz2
	@echo "Done"

data/frll:
	@echo "Downloading the FRLL dataset"
	@curl --location --remote-header-name --remote-name https://figshare.com/ndownloader/files/8541961
	@unzip neutral_front.zip
	@mkdir -p data/frll
	@mv neutral_front/* data/frll/
	@rm -Rf __MACOSX neutral_front.zip
	@echo "Dataset downloaded"
