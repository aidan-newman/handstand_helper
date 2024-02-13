from tasks import video
from tasks import image
from tasks import landmark_analysis
from pathlib import Path

IMG_IN_FOLD = Path("input_output/input_images")
IMG_OUT_FOLD = Path("input_output/output_images")


def main(*args, **kwargs):
    # video.annotate_video()
    # landmark_analysis.check_form([1,2])

    i = 0
    for img in IMG_IN_FOLD.glob("*.jpg"):
        image.annotate_image(str(img), str(IMG_OUT_FOLD / str("annotated_image_" + str(i) + ".jpg")))
        i += 1


if __name__ == "__main__":
    main()
