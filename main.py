from tasks import analysis
from tasks import image
import paths


def main(*args, **kwargs):

    for img in paths.INPUT_IMAGES.iterdir():
        analysis.analyze_image(image.load(img), window=True, predict=True)


if __name__ == "__main__":
    main()
