import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import torch.nn as nn

# ---- CONFIG ----
MODEL_PATH = "swinLung.pth"
IMG_SIZE = 224
HEAD = 0
TOKEN_INDEX = 1300  # index ของ token ที่จะดู attention ไปทุกตำแหน่ง

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, C//heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach().cpu()

        out = attn @ v  # [B, heads, N, C//heads]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MiniSwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=4, embed_dim=96, depth=4, num_heads=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.Sequential(
            *[SwinBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def get_attention_map(self):
        return self.blocks[-1].attn.attn_map

# ---- Load model ----
@st.cache_resource
def load_model():
    model = MiniSwinTransformer()  # สร้างโมเดล
    state_dict = torch.load('swinLung.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # โหลด weights
    model.eval()  # ตั้งให้โมเดลทำงานใน evaluation mode
    return model

# ---- ดึง attention map จากชั้น WindowAttention ล่าสุด ----
def get_last_attn_map(model):
    for module in reversed(list(model.modules())):
        if hasattr(module, 'attn_map') and module.attn_map is not None:
            return module.attn_map
    return None

# ---- Image Preprocessing ----
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# ---- แสดง Attention Map เป็นรูป ----
def plot_attention(attn_map, head=0, token_index=TOKEN_INDEX):
    mat = attn_map[0, head]  # (N, N)
    token_attn = mat[token_index].reshape(56, 56)  # (56x56) ขึ้นกับ patch ของคุณ
    fig, ax = plt.subplots()
    cax = ax.imshow(token_attn, cmap='jet')
    ax.set_title(f"Attention Head {head} – Token {token_index}")
    ax.axis('off')
    fig.colorbar(cax)
    return fig

# ---- Streamlit App ----
st.title("🧠 Image Classification with Attention Map")
st.write("Upload an image and see the predicted class with its attention map.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                logits = model(input_tensor)
                pred_class = logits.argmax(dim=1).item()
                attn_map = get_last_attn_map(model)
        if pred_class==0:
            prediction = "COVID-19"
        elif pred_class==1:
            prediction = "Lung Opacity"
        elif pred_class==2:
            prediction = "Normal"
        elif pred_class==3:
            prediction = "Viral Pneumonia"
        st.success(f"✅ **Predicted Class:** {prediction}")

        if attn_map is not None:
            fig = plot_attention(attn_map, head=HEAD)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No attention map found in model.")
