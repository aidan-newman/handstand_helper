from tasks import video
from tasks import image
from tasks import landmark_analysis
from pathlib import Path


def main(*args, **kwargs):
    # video.annotate_video()
    # landmark_analysis.check_form([1,2])
    imgs_path = Path("input_output") / "input_images"

    i = 0
    for img in imgs_path.glob("*.jpg"):
        image.annotate_image(img, "input_output/output_images/annotated_image.jpg")
        i += 1


if __name__ == "__main__":
    main()
