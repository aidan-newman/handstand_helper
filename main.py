from tasks import analysis
from tasks import image
import paths


def main(*args, **kwargs):

    for vid in paths.INPUT_VIDEOS.iterdir():
        corrections, _ = analysis.analyze_video(window=True, predict=True)

    # i = 0
    # for correction in analysis.CORRECTIONS:
    #     if corrections[i] > 0.09:
    #         print(correction + ": " + str(corrections[i]))
    #     i += 1


if __name__ == "__main__":
    main()
