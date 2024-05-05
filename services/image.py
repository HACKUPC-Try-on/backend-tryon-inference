from services.human import HumanService
from services.inference import InferenceService


class ImageService:
    @staticmethod
    def get_images(dict, garm_img):
        human_service = HumanService()
        human_img_orig = dict["background"].convert("RGB")
        human_img, crop_size = human_service.setup_human_image(
            dict, human_img_orig
        )
        mask = human_service.get_mask(human_img)
        pose_img = human_service.get_pose_img(human_img)

        inference_service = InferenceService(
            garm_img, pose_img, mask, human_img, crop_size
        )
        result = inference_service.inferenece()

        return human_service.get_new_img(human_img_orig, result)
