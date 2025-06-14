import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import urllib.request

# ---- CONFIG ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "swinLung.pth")
IMG_SIZE = 224
HEAD = 0
TOKEN_INDEX = 1300

# ----------------- Model Classes -----------------
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
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach().cpu()
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
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
        self.blocks = nn.Sequential(*[SwinBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def get_attention_map(self):
        return self.blocks[-1].attn.attn_map

# ----------------- Utility Functions -----------------
@st.cache_resource

def load_model():
    model = MiniSwinTransformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def get_last_attn_map(model):
    for module in reversed(list(model.modules())):
        if hasattr(module, 'attn_map') and module.attn_map is not None:
            return module.attn_map
    return None

def plot_attention(attn_map, head=0, token_index=TOKEN_INDEX):
    mat = attn_map[0, head]
    token_attn = mat[token_index].reshape(56, 56)
    fig, ax = plt.subplots()
    cax = ax.imshow(token_attn, cmap='jet')
    ax.set_title(f"Attention Head {head} – Token {token_index}")
    ax.axis('off')
    fig.colorbar(cax)
    return fig

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ----------------- Styling -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans Thai', sans-serif;
        background-color: #333333;
        color: #1e1e1e;
    }

    div.stButton > button {
        background-color: #5160df;     /* พื้นฟ้า */
        color: white !important;       /* ตัวอักษรสีขาว */
        padding: 1em 2em;
        font-size: 25px;
        border-radius: 25px;
        border: none;
        margin-top: 25px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        font-weight: bold;
        min-width: 200px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    div.stButton > button:hover {
        background-color: #4CAF50;     /* เปลี่ยนเป็นเขียว */
        color: white !important;       /* ให้ยังเป็นสีขาวตอน hover */
        transform: scale(1.03);
    }

    .centered-img {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    .uploaded-img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        width: 300px;
        height: 300px;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------- Streamlit UI -----------------
st.title("🫁 Radiography Classification with Attention map!")
st.caption("Upload a radiography you want to predict and plot attention map.")
st.caption("อัปโหลดภาพถ่ายรังสีทรวงอกเพื่อดูการวิเคราะห์และความสนใจของโมเดล")

# ----------------- ตัวอย่างภาพ -----------------
SAMPLE_IMAGES = {
    "ตัวอย่างที่ 1": "https://raw.githubusercontent.com/oo90x/swinLung/main/Pics/Samples/sample00.jpg",
    "ตัวอย่างที่ 2": "https://raw.githubusercontent.com/oo90x/swinLung/main/Pics/Samples/sample01.png",
    "ตัวอย่างที่ 3": "https://raw.githubusercontent.com/oo90x/swinLung/main/Pics/Samples/sample02.png",
    "ตัวอย่างที่ 4": "https://raw.githubusercontent.com/oo90x/swinLung/main/Pics/Samples/sample03.png",
    "ตัวอย่างที่ 5": "https://raw.githubusercontent.com/oo90x/swinLung/main/Pics/Samples/sample04.png",
}

use_sample = st.checkbox("Use sample images(ใช้ภาพตัวอย่าง) :หากต้องการทดสอบจากภาพตัวอย่างกรณีกด ✅ และหากประสงค์จะใช้ภาพที่นำมาเองกรุณานำ ✅ ออก")

if use_sample:
    sample_choice = st.selectbox("Choose a sample(เลือกภาพตัวอย่าง)", list(SAMPLE_IMAGES.keys()))
    sample_url = SAMPLE_IMAGES[sample_choice]
    
    try:
        with urllib.request.urlopen(sample_url) as response:
            uploaded_file = io.BytesIO(response.read())
    except Exception as e:
        st.error(f"ไม่สามารถโหลดภาพตัวอย่างได้: {e}")
        uploaded_file = None
else:
    uploaded_file = st.file_uploader("📁 เลือกภาพ JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_base64 = image_to_base64(image)
    st.markdown(
    f'''
    <div class="centered-img">
        <img src="data:image/png;base64,{image_base64}" class="uploaded-img" alt="Uploaded Image" />
    </div>
    ''',
    unsafe_allow_html=True
    )

    if st.button("🔍 ทำนายผล"):
        with st.spinner("กำลังประมวลผล..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                logits = model(input_tensor)
                pred_class = logits.argmax(dim=1).item()
                attn_map = get_last_attn_map(model)

        labels = ["COVID-19", "Lung Opacity", "Normal", "Viral Pneumonia"]
        st.success(f"✅ ผลการทำนาย: **{labels[pred_class]}**")

        if attn_map is not None:
            fig = plot_attention(attn_map, head=HEAD)
            st.pyplot(fig)
        else:
            st.warning("⚠️ ไม่พบ attention map จากโมเดล")
