from utils_mask import get_mask_location
from models.openpose import get_openpose
from models.parsing import get_parsing
import apply_net
import apply_net
from services.tensor import TensorService
from PIL import Image
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)


class HumanService:
    crop_size = None
    openpose_model = None
    parsing_model = None
    tensor_transform = None
    left = None
    top = None

    def __init__(self):
        self.openpose_model = get_openpose()
        self.parsing_model = get_parsing()
        self.tensor_transform = TensorService.get_tensor_transform()

    def setup_human_image(self, dict, human_img_orig):
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        self.left = (width - target_width) / 2
        self.top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((self.left, self.top, right, bottom))
        self.crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
        return human_img, self.crop_size

    def get_mask(self, human_img):
        keypoints = self.openpose_model(human_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        return mask

    def get_pose_img(self, human_img):
        human_img_arg = self._get_human_args(human_img)
        args = apply_net.create_argument_parser().parse_args(
            (
                'show',
                './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                './ckpt/densepose/model_final_162be9.pkl',
                'dp_segm',
                '-v',
                '--opts',
                'MODEL.DEVICE',
                'cuda',
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        return pose_img

    def _get_human_args(self, human_img):
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        return convert_PIL_to_numpy(human_img_arg, format="BGR")

    def get_new_img(self, human_img_orig, out_img):
        return out_img
