#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

# Paths
INPUT_VIDEO := data/match.mp4
OUTPUT_VIDEO := outputs/demo.mp4
OUTPUT_GIF := outputs/demo.gif
OUTPUT_TRACKS := outputs/tracks.jsonl
OUTPUT_METRICS := outputs/track_stats.png
CONFIG := config/config.yaml

#################################################################################
# COMMANDS                                                                      #
#################################################################################


help:	## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

##### DATA

.PHONY: data
data: download-video ## Download and prepare data
	@echo "Data ready."

.PHONY: download-video
download-video: ## Download source video clip from YouTube (CC-BY)
	bash scripts/download_video.sh

$(INPUT_VIDEO): download-video

##### PIPELINE

.PHONY: demo
demo: $(OUTPUT_VIDEO) $(OUTPUT_GIF) $(OUTPUT_TRACKS) ## Run full pipeline (video + GIF + tracks)
	@echo "Demo outputs ready in outputs/"

.PHONY: video
video: $(OUTPUT_VIDEO) ## Generate annotated output video

$(OUTPUT_VIDEO) $(OUTPUT_GIF) $(OUTPUT_TRACKS) $(OUTPUT_METRICS): $(INPUT_VIDEO) $(CONFIG)
	@echo "Running tracking pipeline..."
	python -m football_tracking_demo.run_demo --video $(INPUT_VIDEO) --config $(CONFIG)

.PHONY: gif
gif: $(OUTPUT_GIF) ## Generate demo GIF

.PHONY: tracks
tracks: $(OUTPUT_TRACKS) ## Generate track data (JSONL)

##### DEVELOPMENT

.PHONY: tests
tests: ## Run tests
	pytest tests/

.PHONY: clean-outputs
clean-outputs: ## Remove all generated outputs
	rm -f $(OUTPUT_VIDEO) $(OUTPUT_GIF) $(OUTPUT_TRACKS) $(OUTPUT_METRICS)
	@echo "Outputs cleaned."

##### QUICK START

.PHONY: all
all: data demo ## Download data and run full pipeline
