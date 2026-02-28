import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Beam Analysis Module", layout="wide")
st.title("Beam Analysis Module（外伸梁结构分析）")

# --- 2. 初始化 Session State (用于保存动态状态) ---
if 'unit_system' not in st.session_state:
    st.session_state.unit_system = 'SI'
if 'num_loads' not in st.session_state:
    st.session_state.num_loads = 3

# --- 3. 顶部控制栏 & 单位设置 ---
col_preset1, col_preset2, _ = st.columns([1, 1, 4])
with col_preset1:
    if st.button("Preset: SI (m, kN)"):
        st.session_state.unit_system = 'SI'
        st.rerun()
with col_preset2:
    if st.button("Preset: US (ft, kips)"):
        st.session_state.unit_system = 'US'
        st.rerun()

# 根据当前单位系统定义单位后缀
if st.session_state.unit_system == 'SI':
    u_len, u_force, u_moment, u_dist, u_E, u_I, u_def = "m", "kN", "kN·m", "kN/m", "GPa", "cm4", "mm"
else:
    u_len, u_force, u_moment, u_dist, u_E, u_I, u_def = "ft", "kips", "kips·ft", "kips/ft", "ksi", "in4", "in"

# --- 4. 界面：梁属性与支座 ---
st.subheader("Beam Geometry & Supports")
col1, col2, col3, col4, col5 = st.columns(5)
L = col1.number_input(f"Length (L) [{u_len}]", min_value=0.1, value=10.0, step=1.0)
E_display = col2.number_input(f"Modulus (E) [{u_E}]", min_value=0.1, value=200.0, step=10.0)
I_display = col3.number_input(f"Inertia (I) [{u_I}]", min_value=0.1, value=5000.0, step=100.0)
xA = col4.number_input(f"Support A @ x [{u_len}]", min_value=0.0, max_value=float(L), value=0.0, step=1.0)
xB = col5.number_input(f"Support B @ x [{u_len}]", min_value=0.0, max_value=float(L), value=10.0, step=1.0)

# --- 5. 界面：载荷输入表 ---
st.subheader("Applied Loads")
col_add, col_reset, _ = st.columns([1, 1, 8])
with col_add:
    if st.button("➕ Add Load"):
        st.session_state.num_loads += 1
        st.rerun()
with col_reset:
    if st.button("🔄 Reset Loads"):
        st.session_state.num_loads = 3
        st.rerun()

st.markdown(f"**Positive = Downward**")

loads_data = []
for i in range(st.session_state.num_loads):
    c1, c2, c3 = st.columns(3)
    l_type = c1.selectbox("Type", ["Point Load (↓)", "Distributed Load (↓↓↓)"], key=f"type_{i}")
    pos = c2.text_input(f"Position x ({u_len})", value="5" if i==0 else "", key=f"pos_{i}", help="For dist load, can use 'start-end' like '0-10'")
    mag = c3.text_input(f"Magnitude", value="10" if i==0 else "", key=f"mag_{i}")
    loads_data.append((l_type, pos, mag))

st.divider()

# --- 6. 核心计算与绘图逻辑 ---
if xA >= xB or xB > L:
    st.error("❌ Support positions invalid (must be 0 <= A < B <= L)")
    st.stop()

# 离散化梁
n_pts = 1001
x = np.linspace(0, L, n_pts)
dx = L / (n_pts - 1)
q = np.zeros_like(x)

point_forces = []
applied_forces = []
applied_dist = []

# 解析载荷
try:
    for l_type, s_pos, s_mag in loads_data:
        if not s_pos or not s_mag: 
            continue
            
        mag = float(s_mag)
        
        if "Point" in l_type:
            pos = float(s_pos)
            applied_forces.append((pos, mag))
            idx = np.argmin(np.abs(x - pos))
            q[idx] += mag / dx 
            point_forces.append((pos, mag))
        else: # Distributed
            if "-" in s_pos:
                p1, p2 = map(float, s_pos.split("-"))
            else:
                start = float(s_pos)
                p1, p2 = start, L
            applied_dist.append((p1, p2, mag))
            mask = (x >= p1) & (x <= p2)
            q[mask] += mag
except ValueError:
    st.warning("⚠️ Please enter valid numbers for loads (e.g., '5' or '0-10').")
    st.stop()

# 计算支座反力
moment_sum_about_A = 0
force_sum = 0

for pos, P in applied_forces:
    force_sum += P
    moment_sum_about_A += P * (pos - xA)

for p1, p2, w in applied_dist:
    total_w = w * (p2 - p1)
    center = (p1 + p2) / 2
    force_sum += total_w
    moment_sum_about_A += total_w * (center - xA)

R_B = moment_sum_about_A / (xB - xA)
R_A = force_sum - R_B

# 显示反力结果
st.success(f"**Reactions:** $R_A = {R_A:.2f}$ {u_force} @ {xA}{u_len}  |  $R_B = {R_B:.2f}$ {u_force} @ {xB}{u_len}")

# 计算剪力 V(x)
V = np.zeros_like(x)
for p1, p2, w in applied_dist:
    mask1 = (x >= p1) & (x < p2)
    V[mask1] -= w * (x[mask1] - p1)
    mask2 = (x >= p2)
    V[mask2] -= w * (p2 - p1)

for pos, P in applied_forces:
    V[x >= pos] -= P
    
V[x >= xA] += R_A
V[x >= xB] += R_B

# 计算弯矩 M(x)
M = np.zeros_like(x)
for i in range(1, len(x)):
    M[i] = M[i-1] + (V[i-1] + V[i])/2 * dx

# 计算挠度 (Deflection)
Slope_EI = np.zeros_like(x)
for i in range(1, len(x)):
    Slope_EI[i] = Slope_EI[i-1] + (M[i-1] + M[i])/2 * dx

Def_EI_raw = np.zeros_like(x)
for i in range(1, len(x)):
    Def_EI_raw[i] = Def_EI_raw[i-1] + (Slope_EI[i-1] + Slope_EI[i])/2 * dx

yA_raw = np.interp(xA, x, Def_EI_raw)
yB_raw = np.interp(xB, x, Def_EI_raw)

slope_corr = (yA_raw - yB_raw) / (xB - xA)
c_corr = -yA_raw - slope_corr * xA
Def_EI = Def_EI_raw + slope_corr * x + c_corr

if st.session_state.unit_system == 'SI':
    real_EI = E_display * 1e6 * I_display * 1e-8
    Y_real = Def_EI / real_EI * 1000 
else:
    Y_real = Def_EI 

# --- 7. 绘图函数 (完全兼容 Matplotlib) ---
def draw_plots():
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})
    fig.patch.set_facecolor('#ffffff')
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)

    # Plot 1: Load Diagram
    ax0 = axs[0]
    ax0.set_title("Load Diagram", fontsize=10, fontweight='bold', pad=5)
    ax0.set_ylim(-1.5, 1.5)
    ax0.axis('off') 
    
    beam_L = x[-1]
    rect = Rectangle((0, -0.1), beam_L, 0.2, facecolor='gray', edgecolor='black')
    ax0.add_patch(rect)
    ax0.text(0-0.1, 0.5, "0", ha='center', fontsize=8, color='red')
    ax0.text(beam_L+0.1, 0.5, f"{beam_L}", ha='center',fontsize=8, color='red')
    
    def draw_support(ax, pos, type="pin"):
        tri_size = 6
        ground_len = 0.1
        ground_y = -0.2
        ground_lw = 1.5
        ax.plot(pos, -0.3, 'k^', markersize=tri_size)
        ax.plot([pos-ground_len, pos+ground_len], [ground_y-0.3, ground_y-0.3], 'k-', lw=ground_lw)
        if type == "pin":  
            for i in np.linspace(pos-0.1, pos+0.1, 4):
                ax.plot([i, i-0.06], [ground_y-0.3, ground_y-0.08-0.3], 'k-', lw=0.8)
        elif type == "roller": 
            ax.plot([pos-0.08, pos+0.08], [ground_y-0.3, ground_y-0.3], 'k-', lw=1.2)

    draw_support(ax0, xA, type="pin")
    draw_support(ax0, xB, type="roller")
                
    for pos, mag in point_forces:
        scale = 0.5
        ax0.annotate('', xy=(pos, 0.1), xytext=(pos, 0.1 + scale),
                     arrowprops=dict(facecolor='blue', shrink=0.05, width=1.0, headwidth=3,headlength=4))
        ax0.text(pos, 0.1 + scale + 0.1, f"{mag}", ha='center', color='blue', fontsize=8)
        
    for p1, p2, mag in applied_dist:
        for dp in np.linspace(p1, p2, 5):
             ax0.annotate('', xy=(dp, 0.1), xytext=(dp, 0.4),
                     arrowprops=dict(facecolor='green', alpha=0.5, width=0.8, headwidth=3, headlength=4, shrink=0.05))
        ax0.plot([p1, p2], [0.4, 0.4], 'g-', lw=1)
        ax0.text((p1+p2)/2, 0.5, f"w={mag}", ha='center', color='green', fontsize=8)

    def plot_diagram(ax, y_data, title, unit, color, fill_color):
        ax.plot(x, y_data, color=color, lw=1.5)
        ax.fill_between(x, y_data, 0, color=fill_color, alpha=0.3)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', lw=0.5)
        
        idx_max = np.argmax(np.abs(y_data))
        x_max = x[idx_max]
        y_max = y_data[idx_max]
        
        offset = max(y_data)*0.1 if max(y_data)!=0 else 1
        if y_max < 0: offset = -abs(min(y_data))*0.2
        
        ax.annotate(f"{y_max:.2f}", xy=(x_max, y_max), xytext=(x_max, y_max + offset),
                    arrowprops=dict(arrowstyle="->", color=color),
                    color=color, fontweight='bold', fontsize=9, ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))
        
        title_text = f"{title}  [Max: {np.max(y_data):.2f}, Min: {np.min(y_data):.2f}]"
        ax.text(0.02, 1.05, title_text, transform=ax.transAxes, fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plot_diagram(axs[1], V, "Shear Force (V)", f"Force ({u_force})", '#1976d2', '#bbdefb') 
    plot_diagram(axs[2], M, "Bending Moment (M)", f"Moment ({u_moment})", '#d32f2f', '#ffcdd2') 
    plot_diagram(axs[3], Y_real, "Deflection (w)", f"Disp. ({u_def})", '#388e3c', '#c8e6c9') 
    axs[3].set_xlabel(f"Position x ({u_len})", fontsize=10)
    
    return fig

# 渲染图表
st.pyplot(draw_plots())
