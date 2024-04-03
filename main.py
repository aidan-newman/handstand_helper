import time
import paths
from tasks import analysis
from tasks import image
from neural_network.predict import load_corrections_model, load_identify_model


LIVE_ROTATION = 270  # rotate live video 90d cw


def main(*args, **kwargs):

    # testing
    print("Waiting for TensorFlow...")
    time.sleep(2)

    while True:
        entry = input("Select an option:\n"
                      "[1] Analyze live video\n"
                      "[2] Analyze a stored video\n"
                      "[3] Analyze a stored image\n")

        # input check
        if str.isdigit(entry):
            entry = int(entry)
            if entry == 1 or entry == 2 or entry == 3:

                # create models
                correction_model = load_corrections_model()
                identify_model = load_identify_model()

                # start analyzing live video
                if entry == 1:
                    analysis.analyze_video(
                        correction_model=correction_model,
                        input_rotation=LIVE_ROTATION,
                        identify_model=identify_model,
                        display=True,
                        annotate=True
                    )

                # start analyzing a selected input video
                elif entry == 2:

                    names = []

                    # prompt user to select input video file
                    output = "Select an input video:\n"
                    i = 1
                    for suf in ("*.mp4", "*.mov", "*.avi"):
                        for file in paths.INPUT_VIDEOS.glob(suf):
                            output += ("[" + str(i) + "] " + file.name + "\n")
                            names.append(file.name)
                            i += 1

                    if len(names) == 0:
                        print("No input videos available.")
                        break
                    while True:
                        entry = input(output)
                        # input check
                        if str.isdigit(entry):
                            entry = int(entry)
                            if 1 <= entry <= i:
                                analysis.analyze_video(
                                    filepath=paths.INPUT_VIDEOS / names[entry-1],
                                    correction_model=correction_model,
                                    identify_model=identify_model,
                                    display=True,
                                    annotate=True
                                )
                                break

                # start analyzing a selected input image
                elif entry == 3:

                    names = []

                    # prompt user to select input video file
                    output = "Select an input image:"
                    i = 1
                    for suf in ("*.png", "*.jpg"):
                        for file in paths.INPUT_IMAGES.glob(suf):
                            output += ("\n[" + str(i) + "] " + file.name + "\n")
                            names.append(file.name)
                            i += 1

                    if len(names) == 0:
                        print("No input images available.")
                        break
                    while True:
                        entry = input(output)
                        # input check
                        if str.isdigit(entry):
                            entry = int(entry)
                            if 1 <= entry <= i:
                                analysis.analyze_image(
                                    img=image.load(paths.INPUT_IMAGES / names[entry-1]),
                                    correction_model=correction_model,
                                    identify_model=identify_model,
                                    display=True,
                                    annotate=True
                                )
                                break
                break
    exit(1)


if __name__ == "__main__":
    main()
