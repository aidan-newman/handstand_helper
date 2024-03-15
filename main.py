from tasks import analysis
from neural_network.predict import load_model
import paths
from tasks import image


def main(*args, **kwargs):

    model = load_model()
    # analysis.analyze_video(model=model, window=True)

    analysis.analyze_image(model=model, window=True, hold=True, img=image.load(paths.INPUT_IMAGES / "37.jpg"))


if __name__ == "__main__":
    main()
