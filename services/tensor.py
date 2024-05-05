from torchvision import transforms


class TensorService:

    @staticmethod
    def get_tensor_transform():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
