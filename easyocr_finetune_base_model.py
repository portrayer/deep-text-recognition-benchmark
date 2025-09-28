import torch
import torch.nn as nn
import os
from collections import OrderedDict


# =====================================================================================
# --- 步骤 A: 确保模型定义与 EasyOCR 一致 (一次性修改) ---
# 这个脚本现在是自包含的，包含了之前对 feature_extraction.py 的修正。
# 您无需再修改仓库的其他文件。
# =====================================================================================

class VGG_FeatureExtractor_EasyOCR(nn.Module):
    """ Feature Extractor tailored for EasyOCR's zh_sim_g2.pth model """

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor_EasyOCR, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, output_channel, 2, 1, 0), nn.ReLU(True))


from modules.sequence_modeling import BidirectionalLSTM


class Model(nn.Module):
    """ A simplified Model class definition that matches our needs """

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        # We now use our custom, compatible VGG extractor
        self.FeatureExtraction = VGG_FeatureExtractor_EasyOCR(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size
        self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)

    def forward(self, input, text=None, is_train=True):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction


# =====================================================================================
# --- 步骤 B: 配置区 (您唯一需要修改的地方) ---
# =====================================================================================
class ModelOptions:
    def __init__(self):
        # 请将您自己数据集的专属字符集完整地粘贴到这里
        self.character = "()+123456ABCFHKMNOPSZaeglnu·↑→↓光加温点热照燃电通高"

        # 架构与 EasyOCR zh_sim_g2.pth 完全匹配
        self.Prediction = 'CTC'  # 保持 'CTC'

        # 超参数与 EasyOCR zh_sim_g2.pth 完全匹配
        self.output_channel = 256
        self.hidden_size = 256

        # 其他标准参数
        self.input_channel = 1
        self.imgH = 32
        self.imgW = 100
        self.num_class = len(self.character) + 1  # CTC blank token


# =====================================================================================
# --- 步骤 C: 一体化手术脚本 ---
# =====================================================================================
def create_compatible_finetune_model(opt, pretrained_model_path, new_model_path):
    print("=" * 50)
    print("--- 开始执行一体化模型手术 ---")
    print("=" * 50)

    # --- 1. 加载并脱壳 (移除 'module.' 前缀) ---
    if not os.path.exists(pretrained_model_path):
        print(f"错误: 预训练模型 '{pretrained_model_path}' 不存在。")
        return
    print(f"加载预训练模型: {pretrained_model_path}")
    pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')
    clean_state_dict = OrderedDict()
    for k, v in pretrained_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    print("步骤 1/4: '脱壳'完成，已净化层名称。")

    # --- 2. 创建新模型实例并换头 (迁移权重) ---
    print("\n步骤 2/4: 创建新模型实例并准备'换头'...")
    new_model = Model(opt)
    new_model_state_dict = new_model.state_dict()

    for name, param in clean_state_dict.items():
        if name in new_model_state_dict and new_model_state_dict[name].shape == param.shape:
            new_model_state_dict[name].copy_(param)

    new_model.load_state_dict(new_model_state_dict)
    print(" -> '换头'完成，兼容权重已加载到新模型。")

    # --- 3. 为新模型穿上新壳 (加回 'module.' 前缀) ---
    print("\n步骤 3/4: 为新模型'穿上新壳'以兼容DataParallel...")
    final_state_dict = OrderedDict()
    for k, v in new_model.state_dict().items():
        final_state_dict['module.' + k] = v
    print(" -> '穿壳'完成。")

    # --- 4. 保存最终模型 ---
    torch.save(final_state_dict, new_model_path)
    print("\n步骤 4/4: 保存最终模型。")
    print("=" * 50)
    print(f"🎉 终极手术成功！可以直接用于训练的模型已保存至: '{new_model_path}'")
    print("=" * 50)


if __name__ == '__main__':
    # 初始化配置
    config = ModelOptions()

    # 定义输入和输出文件
    base_model_file = 'zh_sim_g2.pth'
    finetune_starter_file = 'my_finetune_starter_final.pth'  # 使用新名字以示区别

    # 执行一体化手术
    create_compatible_finetune_model(config, base_model_file, finetune_starter_file)