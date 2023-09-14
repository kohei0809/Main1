# AIを使って画像にキャプションを付ける
#from open_clip.src import open_clip  # pip install open_clip_torch
import open_clip
import torch
from PIL import Image

# GPUが利用できる場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AIモデルの読み込み
model, _, transform = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k",
    device=device,
)

# 画像の読み込み（必要に応じてファイル名を変更）
img = Image.open("./figure_caption/1.jpg").convert("RGB")

# 画像からキャプションを生成
im = transform(img).unsqueeze(0).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    generated = model.generate(im, seq_len=20)

# キャプションを人間が読める文章に変換して表示
caption = (
    open_clip.decode(generated[0].detach())
    .split("<end_of_text>")[0]
    .replace("<start_of_text>", "")
)
print(caption)