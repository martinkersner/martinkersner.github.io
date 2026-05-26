.PHONY: serve serve-drafts build clean

serve:
	hugo server

serve-drafts:
	hugo server -D

build:
	hugo --minify

clean:
	rm -rf public resources
