import logging
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.convnext import ConvNeXt
import re
from functools import partial
from timm.models.swin_transformer import SwinTransformer
from timm.layers.classifier import create_classifier
# from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm
from timm.models import create_model
class Classifier(nn.Module):
    def __init__(self, args, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None

        if args.arch == "conv_tiny":
            # Use ConvNeXt Tiny as the backbone
            model = ConvNeXtBase(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
            bn = nn.BatchNorm1d(args.bottleneck_dim)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = args.bottleneck_dim
        elif args.arch == "swin_tiny":
            # Use Swin Tiny as the backbone
            model = swin_tiny(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
            # model = create_model("swin_tiny_patch4_window7_224.ms_in22k",pretrained=True,num_classes=args.bottleneck_dim)
            bn = nn.BatchNorm1d(args.bottleneck_dim)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = args.bottleneck_dim
        elif not self.use_bottleneck:
            model = models.__dict__[args.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        else:
            model = models.__dict__[args.arch](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
            bn = nn.BatchNorm1d(args.bottleneck_dim)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = args.bottleneck_dim

        self.fc = nn.Linear(self.output_dim, args.num_classes)


        ## Initialization and Masking 
        # for m in self.modules():
        #     # if isinstance(m, nn.Conv2d):
        #     #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal(m.weight.data)   # Initializing with orthogonal rows

        if self.use_weight_norm:
            self.fc = weight_norm(self.fc, dim=args.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        if isinstance(self.encoder[0], ConvNeXtBase):
            backbone_params.extend(self.encoder.parameters())
        elif isinstance(self.encoder[0], swin_tiny):
            backbone_params.extend(self.encoder.parameters())
        elif not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1].parameters())
            
        extra_params.extend(self.fc.parameters())

        # Exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0


class ConvNeXtBase(ConvNeXt):
    def __init__(self, conv_name="conv_tiny", pretrained=True):
        # ConvNeXt 모델 구성에 따라 깊이와 차원을 설정
        if conv_name == "conv_tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        else:
            # 다른 ConvNeXt 모델 구성이 필요한 경우 여기에 추가
            raise ValueError(f"Unsupported ConvNeXt model name: {conv_name}")

        super().__init__(depths=depths, dims=dims)

        self.output_dim = dims[-1]  # 마지막 차원을 출력 차원으로 설정
        if pretrained:
            self._load_pretrained_weights(conv_name)
        self.global_pool, self.fc = create_classifier(
            self.output_dim,
            self.num_classes,
            input_fmt='NHWC'
        )

    def _load_pretrained_weights(
        self,
        conv_name,
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    ):
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        if "model" in checkpoint:
            param_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            param_dict = checkpoint["state_dict"]
        else:
            param_dict = checkpoint
        load_flag = True
        import re

        for k, v in param_dict.items():
            k = k.replace("downsample_layers.0.", "stem.")
            k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
            k = re.sub(
                r"downsample_layers.([0-9]+).([0-9]+)", r"stages.\1.downsample.\2", k
            )
            k = k.replace("dwconv", "conv_dw")
            k = k.replace("pwconv", "mlp.fc")
            if "grn" in k:
                k = k.replace("grn.beta", "mlp.grn.bias")
                k = k.replace("grn.gamma", "mlp.grn.weight")
                v = v.reshape(v.shape[-1])
            k = k.replace("head.", "head.fc.")
            if k.startswith("norm."):
                k = k.replace("norm", "head.norm")
            if "head" in k:
                continue
            try:
                self.state_dict()[k].copy_(v)
            except:
                print("===========================ERROR=========================")
                print(
                    "shape do not match in k :{}: param_dict{} vs self.state_dict(){}".format(
                        k, v.shape, self.state_dict()[k].shape
                    )
                )
                load_flag = False

        # if ~load_flag:
        #     raise Exception(f'load_state_dict from {url} fail')

    def forward(self, x):
        # 모델의 특징 추출 부분만 사용
        x = self.forward_features(x)
        # 전역 평균 풀링 적용
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # 특징 벡터를 평탄화
        out = self.fc(x)
        return out

    @property
    def in_features(self):
        # 외부에서 모델의 입력 특징 차원에 접근할 수 있게 함
        return self.output_dim
    
    
class swin_tiny(SwinTransformer):
    def __init__(self, pretrained=True):
        super().__init__(patch_size=4, window_size = 7, embed_dim=96, depth=(2, 2, 6, 2), 
                         num_heads=(3, 6, 12, 24), mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.head = None
        self.output_dim = 768
        if pretrained:
            self._my_load_from_state_dict()
        self.global_pool, self.fc = create_classifier(
            self.output_dim,
            self.num_classes,
            input_fmt='NHWC'
        )
        
    def get_feature_dim(self):
        return self.output_dim
    
    def _my_load_from_state_dict(self, url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth'):
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint:
            param_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            param_dict = checkpoint['state_dict']
        else:
            param_dict = checkpoint
            
        load_flag = True
        for k, v in param_dict.items():
            if 'dist' in k:
                continue
            if k == 'head.weight':
                continue
            if k == 'head.bias':
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at without distillation pos
                v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
            elif "downsample" in k:
                pattern = r'layers\.(\d+)\.'
                def replace(match):
                    index = int(match.group(1))
                    return f'layers.{index + 1}.'
                k = re.sub(pattern, replace, k)

            if k in self.state_dict():
                try:
                    self.state_dict()[k].copy_(v)
                except Exception as e:
                    print(f'Error loading {k}: {e}')
            else:
                print(f'Skipping {k}: not in model\'s state_dict')


            # try:
            #     self.state_dict()[k].copy_(v)
            # except:
            #     print('===========================ERROR=========================')
            #     print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
            #     load_flag = False
        if load_flag:
            print(f'load_state_dict from {url} successful')
        else:
            raise Exception(f'load_state_dict from {url} fail')
    def forward(self, x):
        feat = super().forward_features(x)
        if feat.dim() == 3:
            feat = feat[:, 0]
        feat = self.global_pool(feat)
        out = self.fc(feat)
        return out
    
    @property
    def in_features(self):
        # 외부에서 모델의 입력 특징 차원에 접근할 수 있게 함
        return self.output_dim