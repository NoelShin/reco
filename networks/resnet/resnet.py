import torch
import torch.nn as nn
from networks.resnet.resnet_backbone import ResNetBackbone


class ResNet50(nn.Module):
    def __init__(
            self,
            weight_type: str = "supervised",
            use_dilated_resnet: bool = False
    ):
        super(ResNet50, self).__init__()
        self.network = ResNetBackbone(backbone=f"resnet50{'_dilated8' if use_dilated_resnet else ''}", pretrained=None)
        self.n_embs = self.network.num_features
        self.use_dilated_resnet = use_dilated_resnet
        self._load_pretrained(weight_type)

    def _load_pretrained(self, training_method: str) -> None:
        curr_state_dict = self.network.state_dict()
        if training_method == "mocov2":
            state_dict = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
            )["state_dict"]

            for k in list(state_dict.keys()):
                if any([k.find(w) != -1 for w in ("fc.0", "fc.2")]):
                    state_dict.pop(k)

        elif training_method == "swav":
            # state_dict = torch.load("/users/gyungin/sos/networks/pretrained/swav_800ep_pretrain.pth.tar")
            state_dict = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
            )
            for k in list(state_dict.keys()):
                if any([k.find(w) != -1 for w in ("projection_head", "prototypes")]):
                    state_dict.pop(k)

        elif training_method == "supervised":
            # Note - pytorch resnet50 model doesn't have num_batches_tracked layers. Need to know why.
            # for k in list(curr_state_dict.keys()):
            #     if k.find("num_batches_tracked") != -1:
            #         curr_state_dict.pop(k)
            # state_dict = torch.load("../networks/pretrained/resnet50-pytorch.pth")
            from torchvision.models.resnet import resnet50
            resnet50_supervised = resnet50(True, True)
            state_dict = resnet50_supervised.state_dict()
            for k in list(state_dict.keys()):
                if any([k.find(w) != -1 for w in ("fc.weight", "fc.bias")]):
                    state_dict.pop(k)

        assert len(curr_state_dict) == len(state_dict), f"# layers are different: {len(curr_state_dict)} != {len(state_dict)}"
        for k_curr, k in zip(curr_state_dict.keys(), state_dict.keys()):
            curr_state_dict[k_curr].copy_(state_dict[k])
        print(f"ResNet50{' (dilated)' if self.use_dilated_resnet else ''} intialised with {training_method} weights is loaded.")

    def forward(self, x, norm: bool = False):
        features: torch.Tensor = self.network(x)[-1]  # return activation from the last layer only
        if norm:
            features = torch.nn.functional.layer_norm(features, features.shape[1:])
        return features


if __name__ == '__main__':
    resnet = ResNet50("mocov2")