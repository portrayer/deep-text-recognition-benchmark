import torch
from collections import OrderedDict
import os
from model import Model
from easyocr_finetune_base_model import ModelOptions

def inspect_model_shapes():
    print("=" * 80)
    print("--- 模型结构与形状诊断工具 ---")
    print("=" * 80)

    # --- 1. 加载并净化预训练模型 ---
    pretrained_model_path = 'zh_sim_g2.pth'
    if not os.path.exists(pretrained_model_path):
        print(f"错误: 预训练模型 '{pretrained_model_path}' 不存在。")
        return

    pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')

    clean_state_dict = OrderedDict()
    for k, v in pretrained_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    print(f"已加载并净化预训练模型 '{pretrained_model_path}'")

    # --- 2. 创建我们自己定义的新模型 ---
    opt = ModelOptions()
    new_model = Model(opt)
    new_model_state_dict = new_model.state_dict()
    print(f"已根据当前配置创建新模型实例。")
    print(f"  - 架构: {opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}")
    print(f"  - 超参数: output_channel={opt.output_channel}, hidden_size={opt.hidden_size}")

    # --- 3. 逐层对比形状 ---
    print("\n--- 开始逐层对比模型形状 ---")
    print(f"{'层名称':<50} | {'新模型形状':<25} | {'预训练模型形状':<25}")
    print("-" * 110)

    all_keys = set(clean_state_dict.keys()) | set(new_model_state_dict.keys())

    for name in sorted(list(all_keys)):
        new_shape = str(new_model_state_dict[name].shape) if name in new_model_state_dict else "----"
        pre_shape = str(clean_state_dict[name].shape) if name in clean_state_dict else "----"

        status = ""
        if new_shape != pre_shape:
            status = "  <--- [不匹配! MISMATCH!]"

        print(f"{name:<50} | {new_shape:<25} | {pre_shape:<25}{status}")

    print("\n诊断完成。请检查上面标记为 [不匹配! MISMATCH!] 的行，")
    print("并根据预训练模型的形状，调整 create_finetune_model.py 中的超参数。")


if __name__ == '__main__':
    inspect_model_shapes()