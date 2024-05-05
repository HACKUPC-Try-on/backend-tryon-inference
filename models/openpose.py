from preprocess.openpose.run_openpose import OpenPose


def get_openpose():
    openpose = OpenPose(0)
    openpose.preprocessor.body_estimation.model.to("cuda")
    return openpose
