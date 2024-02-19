from tasks import analysis
from tasks import image
from pathlib import Path

IMG_IN_FOLD = Path("input_output/input_images")
IMG_OUT_FOLD = Path("input_output/output_images")


def main(*args, **kwargs):

    # i = 0
    # for img in IMG_IN_FOLD.glob("*.jpg"):
    #     analysis.analyze_image(image.load(str(img)), True)
    p = IMG_IN_FOLD / "pullup.jpg"
    analysis.analyze_image(image.load(str(p)), True)


if __name__ == "__main__":
    main()
