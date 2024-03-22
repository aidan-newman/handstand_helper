import paths
from tasks import analysis
from neural_network.predict import load_corrections_model, load_identify_model
from pathlib import Path
from tasks import image


def main(*args, **kwargs):

    correction_model = load_corrections_model()
    identify_model = load_identify_model()
    
    analysis.analyze_video(
        filepath=paths.INPUT_VIDEOS / "vid.mp4",
        correction_model=correction_model,
        identify_model=identify_model,
        output_window=True,
        annotate=True,
        save_file=paths.OUTPUT_VIDEOS / "output_video.avi"
    )

    # count = 0
    # for img in paths.INPUT_IMAGES.iterdir():
    #     analysis.analyze_image(img=image.load(img),
    #                            identify_model=idt_model,
    #                            correction_model=correction_model,
    #                            window=True,
    #                            hold=True,
    #                            annotate=True,
    #                            count = count
    #                            )
    #     count += 1


if __name__ == "__main__":
    main()
