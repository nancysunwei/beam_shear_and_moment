import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

class BeamAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Beam Analysis Module（外伸梁结构分析）")
        self.root.geometry("1200x950")

        # --- 数据存储 ---
        self.entries_loads = []
        
        # --- 单位变量 ---
        self.u_len = tk.StringVar(value="m")
        self.u_force = tk.StringVar(value="kN")
        self.u_moment = tk.StringVar(value="kN·m")
        self.u_dist = tk.StringVar(value="kN/m")
        self.u_E = tk.StringVar(value="GPa")
        self.u_I = tk.StringVar(value="cm4")
        self.u_def = tk.StringVar(value="mm")

        self.setup_ui()

    def setup_ui(self):
        # 1. 顶部控制栏 (Top Bar)
        ctrl_frame = tk.Frame(self.root, pady=5, padx=5, relief="raised", bd=1)
        ctrl_frame.pack(fill="x")

        # 预设按钮
        tk.Label(ctrl_frame, text="Presets:").pack(side="left")
        tk.Button(ctrl_frame, text="SI (m, kN)", command=self.set_metric, bg="#e1f5fe").pack(side="left", padx=2)
        tk.Button(ctrl_frame, text="US (ft, kips)", command=self.set_imperial, bg="#e1f5fe").pack(side="left", padx=2)
        
        tk.Button(ctrl_frame, text="Compute & Plot", command=self.calculate, bg="#c8e6c9", font=('Arial', 10, 'bold')).pack(side="right", padx=10)
        tk.Button(ctrl_frame, text="Reset Loads", command=self.reset_loads).pack(side="right", padx=5)

        # 2. 主面板 (Main Pane)
        self.main_pane = tk.PanedWindow(self.root, orient="vertical")
        self.main_pane.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.top_input_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.top_input_frame)

        # --- 2.1 梁属性与支座 (Beam Properties & Supports) ---
        prop_frame = tk.LabelFrame(self.top_input_frame, text="Beam Geometry & Supports", padx=5, pady=5)
        prop_frame.pack(fill="x", padx=5, pady=5)
        
        # 辅助函数：创建输入框
        def create_entry(parent, label, var_unit, row, col, def_val):
            f = tk.Frame(parent); f.grid(row=row, column=col, padx=15, pady=2, sticky="w")
            tk.Label(f, text=label, font=('Arial',9,'bold')).pack(side="left")
            e = tk.Entry(f, width=8); e.insert(0, def_val); e.pack(side="left", padx=5)
            tk.Label(f, textvariable=var_unit).pack(side="left")
            return e

        self.e_L = create_entry(prop_frame, "Length (L):", self.u_len, 0, 0, "10")
        self.e_E = create_entry(prop_frame, "Modulus (E):", self.u_E, 0, 1, "200")
        self.e_I = create_entry(prop_frame, "Inertia (I):", self.u_I, 0, 2, "5000")
        
        # 支座位置
        tk.Label(prop_frame, text="|  Supports @ x =", fg="#666").grid(row=0, column=3, padx=5)
        self.e_sup1 = create_entry(prop_frame, "Support A:", self.u_len, 0, 4, "0")
        self.e_sup2 = create_entry(prop_frame, "Support B:", self.u_len, 0, 5, "10")

        # --- 2.2 载荷输入表 (Loads Table) ---
        load_frame_container = tk.LabelFrame(self.top_input_frame, text="Applied Loads", padx=5, pady=5)
        load_frame_container.pack(fill="x", padx=5, pady=5)
        
        # 表头
        headers = ["Type", f"Position x ({self.u_len.get()})", f"Magnitude (Force or Force/Length)", "Description"]
        for col, t in enumerate(headers):
            tk.Label(load_frame_container, text=t, font=('Arial',9,'bold'), fg="#3f51b5").grid(row=0, column=col, sticky="w", padx=10)

        self.load_table = tk.Frame(load_frame_container)
        self.load_table.grid(row=1, column=0, columnspan=4, sticky="we")
        
        # 添加增加载荷行的按钮
        btn_add = tk.Button(load_frame_container, text="+ Add Load", command=self.add_load_row, bg="#e0e0e0", width=10)
        btn_add.grid(row=2, column=0, pady=5, sticky="w")

        # 初始化3行载荷
        for _ in range(3): self.add_load_row()

        # --- 2.3 结果显示栏 ---
        res_frame = tk.LabelFrame(self.top_input_frame, text="Reaction Forces & Max Values", padx=5, pady=5, bg="#fff3e0")
        res_frame.pack(fill="x", padx=5, pady=5)
        
        self.lbl_reactions = tk.Label(res_frame, text="Reactions: N/A", bg="#fff3e0", font=('Consolas', 10))
        self.lbl_reactions.pack(side="left", padx=10)
        self.lbl_max_vals = tk.Label(res_frame, text="", bg="#fff3e0", font=('Arial', 9, 'bold'), fg="#d32f2f")
        self.lbl_max_vals.pack(side="right", padx=10)

        # 3. 绘图区域
        self.plot_frame = tk.Frame(self.main_pane, bg="white", bd=2, relief="sunken")
        self.main_pane.add(self.plot_frame)

    def set_metric(self):
        self.u_len.set("m"); self.u_force.set("kN"); self.u_dist.set("kN/m")
        self.u_moment.set("kN·m"); self.u_E.set("GPa"); self.u_I.set("cm4"); self.u_def.set("mm")
        self.reset_headers()

    def set_imperial(self):
        self.u_len.set("ft"); self.u_force.set("kips"); self.u_dist.set("kips/ft")
        self.u_moment.set("kips·ft"); self.u_E.set("ksi"); self.u_I.set("in4"); self.u_def.set("in")
        self.reset_headers()

    def reset_headers(self):
        # 简单刷新界面文本
        pass 

    def add_load_row(self):
        row = len(self.entries_loads)
        
        # 载荷类型下拉
        cb_type = ttk.Combobox(self.load_table, values=["Point Load (↓)", "Distributed Load (↓↓↓)"], width=18, state="readonly")
        cb_type.set("Point Load (↓)")
        cb_type.grid(row=row, column=0, padx=5, pady=2)
        
        # 位置 (Start Pos)
        e_pos = tk.Entry(self.load_table, width=10); e_pos.grid(row=row, column=1, padx=5)
        e_pos.insert(0, "5" if row==0 else "")
        
        # 大小
        e_mag = tk.Entry(self.load_table, width=15); e_mag.grid(row=row, column=2, padx=5)
        e_mag.insert(0, "10" if row==0 else "")
        
        # 说明
        lbl_desc = tk.Label(self.load_table, text="Positive = Downward", fg="gray", font=('Arial', 8))
        lbl_desc.grid(row=row, column=3, padx=5)

        self.entries_loads.append((cb_type, e_pos, e_mag, lbl_desc))

    def reset_loads(self):
        for w in self.load_table.winfo_children(): w.destroy()
        self.entries_loads.clear()
        for _ in range(3): self.add_load_row()

    def calculate(self):
        try:
            # 1. 获取梁参数
            L = float(self.e_L.get())
            E_display = float(self.e_E.get()) # GPa or ksi
            I_display = float(self.e_I.get()) # cm4 or in4
            xA = float(self.e_sup1.get())
            xB = float(self.e_sup2.get())

            if xA >= xB or xB > L:
                raise ValueError("Support positions invalid (must be 0 <= A < B <= L)")

            # 离散化梁 (用于数值计算)
            n_pts = 1001
            x = np.linspace(0, L, n_pts)
            dx = L / (n_pts - 1)
            
            # 初始化载荷分布数组 q(x) (Positive Downward)
            q = np.zeros_like(x)
            
            # 记录集中力用于剪力修正
            point_forces = [] # (pos, mag)

            # 2. 读取载荷
            applied_forces = [] # for reaction calc: (pos, mag)
            applied_dist = []   # for reaction calc: (start, end, mag) - assume full length for simple implementation if dist
            
            # 简单的矩形分布载荷逻辑：
            # 如果是分布载荷，暂时默认施加在全长或用户指定点（简化处理：输入位置视为分布载荷中心？不，改为全长或分段）
            # 为了程序健壮性，本例中分布载荷输入Position格式为 "start-end" 或者 单一数值(视为全长)
            
            for cb, e_p, e_m, _ in self.entries_loads:
                s_pos = e_p.get()
                s_mag = e_m.get()
                if not s_pos or not s_mag: continue
                
                mag = float(s_mag)
                l_type = cb.get()

                if "Point" in l_type:
                    pos = float(s_pos)
                    applied_forces.append((pos, mag))
                    # 在离散数组中添加
                    idx = np.argmin(np.abs(x - pos))
                    q[idx] += mag / dx # 近似为分布载荷脉冲
                    point_forces.append((pos, mag))
                else: # Distributed
                    # 解析位置 "0-10" 或 "5"
                    if "-" in s_pos:
                        p1, p2 = map(float, s_pos.split("-"))
                    else:
                        # 如果只输一个数，暂且视为从该点到梁尾，或者若输入0则为全长
                        start = float(s_pos)
                        p1, p2 = start, L
                    
                    applied_dist.append((p1, p2, mag))
                    # 填充离散数组
                    mask = (x >= p1) & (x <= p2)
                    q[mask] += mag

            # 3. 计算支座反力 (Equilibrium Equations)
            # Sigma M_A = 0 -> R_B * (xB - xA) = sum(P * (xp - xA)) + sum(w * len * (center - xA))
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
            
            # 显示反力
            self.lbl_reactions.config(text=f"Reactions: RA = {R_A:.2f} {self.u_force.get()} @ {xA}m,  RB = {R_B:.2f} {self.u_force.get()} @ {xB}m")

            # 将反力添加到载荷数组中 (Upward is negative in our q array which is Downward Positive)
            # q array is defined as Downward (+). Reactions are usually Upward (-).
            # So we add -R_A and -R_B
            idx_A = np.argmin(np.abs(x - xA))
            idx_B = np.argmin(np.abs(x - xB))
            
            # 注意：数值积分对集中力处理较差，剪力图容易斜坡。
            # 我们改用半解析法：先积分q得到V_raw，再手动叠加集中力产生的阶跃。
            
            # --- 4. 计算剪力 V(x) ---
            # V(x) = int(-q) dx. Convention: Upward force on left face is positive shear.
            # Standard beam convention: dV/dx = -w(x). 
            # So V(x) = Integral(-w) + Reactions
            
            V = np.zeros_like(x)
            
            # 处理分布载荷贡献
            for p1, p2, w in applied_dist:
                # w is downward positive. V slope is -w.
                # For x > p1: add -w * (x - p1). Cap at p2.
                mask1 = (x >= p1) & (x < p2)
                V[mask1] -= w * (x[mask1] - p1)
                mask2 = (x >= p2)
                V[mask2] -= w * (p2 - p1)

            # 处理集中力贡献 (含反力)
            # Point load P (downward) causes V to drop by P at x.
            # Reaction R (upward) causes V to jump by R.
            
            # 所有外力 (Down)
            for pos, P in applied_forces:
                V[x >= pos] -= P
                
            # 反力 (Up)
            V[x >= xA] += R_A
            V[x >= xB] += R_B
            
            # --- 5. 计算弯矩 M(x) ---
            # M(x) = Integral(V) dx
            M = np.zeros_like(x)
            # 使用梯形积分
            for i in range(1, len(x)):
                M[i] = M[i-1] + (V[i-1] + V[i])/2 * dx

            # --- 6. 计算挠度 (Deflection) ---
            # EI * d2y/dx2 = M(x)  (assuming positive M causes compression on top, smile shape curvature)
            # Note on signs: Usually EI y'' = -M or M depending on convention.
            # Standard: M positive = sagging (smile). y positive = upward? usually y positive is downward in code or upward.
            # Let's assume standard calculus: y is positive UP.
            # EI y'' = M(x).
            # Slope Theta = Int(M) / EI
            # Deflection y = Int(Theta)
            
            # 单位换算：我们需要统一到基本单位 (N, m, Pa) 或保持一致
            # 假设用户输入的是 consistent units，或者我们只做定性形状，最后乘系数。
            # 为了数值准确，做简单转换：
            # Force=kN(1e3), Len=m, E=GPa(1e9), I=cm4(1e-8 m4)
            # EI unit = 1e9 * 1e-8 = 10 N·m2 = 0.01 kN·m2
            
            scale_EI = 1.0
            if self.u_len.get() == 'm':
                # SI: E(GPa)->kN/m2 is 1e6. I(cm4)->m4 is 1e-8.
                # EI (kN m2) = E * 1e6 * I * 1e-8 = E * I * 0.01
                scale_EI = E_display * I_display * 0.01
            else:
                # US: E(ksi)->kips/in2. I(in4). L(ft).
                # Convert everything to ft, kips? Or in, kips.
                # Let's stick to Length Unit as base. L in ft.
                # E(ksi) = 144 * E(ksf). I(in4) = I/20736 ft4.
                # This is messy. Let's just output y * EI for demonstration or assume EI=1 for shape
                # 实际演示中，挠度数值非常敏感。这里为了演示图表，使用 EI=const 进行归一化积分，
                # 并在显示时标注 "Calculated with EI=..."
                scale_EI = 1.0 # Placeholder
            
            # 积分 M 得到 Slope*EI
            Slope_EI = np.zeros_like(x)
            for i in range(1, len(x)):
                Slope_EI[i] = Slope_EI[i-1] + (M[i-1] + M[i])/2 * dx
            
            # 积分 Slope 得到 Deflection*EI (Raw)
            Def_EI_raw = np.zeros_like(x)
            for i in range(1, len(x)):
                Def_EI_raw[i] = Def_EI_raw[i-1] + (Slope_EI[i-1] + Slope_EI[i])/2 * dx

            # 修正边界条件: y(A) = 0, y(B) = 0
            # 当前 Def_EI_raw 是一条飘在空中的曲线。
            # 实际挠度 y(x) = y_raw(x) + C1*x + C2
            # 利用 y(xA)=0, y(xB)=0 解 C1, C2
            yA_raw = np.interp(xA, x, Def_EI_raw)
            yB_raw = np.interp(xB, x, Def_EI_raw)
            
            # linear correction term: Y_corr(x) = m*x + c
            # m*xA + c = -yA_raw
            # m*xB + c = -yB_raw
            # Subtracting: m(xB-xA) = -(yB_raw - yA_raw) -> m = (yA_raw - yB_raw) / (xB-xA)
            slope_corr = (yA_raw - yB_raw) / (xB - xA)
            c_corr = -yA_raw - slope_corr * xA
            
            Def_EI = Def_EI_raw + slope_corr * x + c_corr
            
            # 真实的挠度值 (approximate unit conversion)
            # 如果 SI: M in kN.m. Numerator int(int(M)) is kN.m3
            # EI is kN.m2. Result is m.
            # We used scale_EI = 1. So we divide by real EI now.
            if self.u_len.get() == 'm':
                real_EI = E_display * 1e6 * I_display * 1e-8 # kN m2
                Y_real = Def_EI / real_EI * 1000 # convert m to mm
            else:
                Y_real = Def_EI # 暂不处理英制复杂换算，显示相对值

            # 7. 绘图
            self.draw_plots(x, q, V, M, Y_real, xA, xB, applied_forces, applied_dist)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw_plots(self, x, q, V, M, Y, xA, xB, points, dists):
        # 清除旧图
        for widget in self.plot_frame.winfo_children(): widget.destroy()
        
        # 创建画布 (4行：载荷, 剪力, 弯矩, 挠度)
        fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})
        fig.patch.set_facecolor('#f5f5f5')
        plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)

        # --- Plot 1: Load Diagram ---
        ax0 = axs[0]
        ax0.set_title("Load Diagram", fontsize=9, fontweight='bold', pad=3)
        ax0.set_ylim(-1.5, 1.5)
        ax0.axis('off') # 关闭坐标轴，手动画梁
        
        # 画梁
        beam_L = x[-1]
        rect = Rectangle((0, -0.1), beam_L, 0.2, facecolor='gray', edgecolor='black')
        ax0.add_patch(rect)
        # 标注梁起点x=0
        ax0.text(0-0.1, 0.5, "0", ha='center', fontsize=8, color='red')
        # 对应梁终点标注（对比参考）
        ax0.text(beam_L+0.1, 0.5, f"{beam_L}", ha='center',fontsize=8, color='red')
        # 画支座
        def draw_support(ax, pos, type="pin"):
            # 基础尺寸（统一缩小）
            tri_size = 6
            ground_len = 0.1
            ground_y = -0.2
            ground_lw = 1.5
            
            # 绘制三角指针
            ax.plot(pos, -0.3, 'k^', markersize=tri_size)
            # 绘制地面基线
            ax.plot([pos-ground_len, pos+ground_len], [ground_y-0.3, ground_y-0.3], 'k-', lw=ground_lw)
            
            # 不同支座类型的纹理区分
            if type == "pin":  # 铰支座（原样式，缩小版）
                for i in np.linspace(pos-0.1, pos+0.1, 4):
                    ax.plot([i, i-0.06], [ground_y-0.3, ground_y-0.08-0.3], 'k-', lw=0.8)
            elif type == "roller":  # 滚动支座（简化为单条竖线）
                #ax.plot([pos, pos], [ground_y-0.5, ground_y-0.08-0.5], 'k-', lw=1.2)
                ax.plot([pos-0.08, pos+0.08], [ground_y-0.3, ground_y-0.3], 'k-', lw=1.2)

        # 调用示例：xA为铰支座，xB为滚动支座
        draw_support(ax0, xA, type="pin")
        draw_support(ax0, xB, type="roller")
                    
        # 画载荷箭头
        max_load = 10
        if points: max_load = max([p[1] for p in points])
        
        for pos, mag in points:
            # 向下的力
            scale = 0.5
            ax0.annotate('', xy=(pos, 0.1), xytext=(pos, 0.1 + scale),
                         arrowprops=dict(facecolor='blue', shrink=0.05, width=1.0, headwidth=3,headlength=4))
            ax0.text(pos, 0.1 + scale + 0.1, f"{mag}", ha='center', color='blue', fontsize=8)
            
        for p1, p2, mag in dists:
            # 画一排箭头
            for dp in np.linspace(p1, p2, 5):
                 ax0.annotate('', xy=(dp, 0.1), xytext=(dp, 0.4),
                         arrowprops=dict(facecolor='green', alpha=0.5,
                                 width=0.8, headwidth=3, headlength=4, shrink=0.05))
            # 画横杠
            ax0.plot([p1, p2], [0.4, 0.4], 'g-', lw=1)
            ax0.text((p1+p2)/2, 0.5, f"w={mag}", ha='center', color='green', fontsize=8)

       # ax0.text(0, -0.5, "0", ha='center')
        #ax0.text(beam_L, -0.5, f"{beam_L}", ha='center')

        # 通用绘图函数
        def plot_diagram(ax, y_data, title, unit, color, fill_color):
            ax.plot(x, y_data, color=color, lw=1.5)
            ax.fill_between(x, y_data, 0, color=fill_color, alpha=0.3)
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axhline(0, color='black', lw=0.5)
            
            # 标注最大值
            idx_max = np.argmax(np.abs(y_data))
            x_max = x[idx_max]
            y_max = y_data[idx_max]
            
            # 使用 annotate 避免遮挡
            offset = max(y_data)*0.1 if max(y_data)!=0 else 1
            if y_max < 0: offset = -abs(min(y_data))*0.2
            
            ax.annotate(f"{y_max:.2f}", xy=(x_max, y_max), xytext=(x_max, y_max + offset),
                        arrowprops=dict(arrowstyle="->", color=color),
                        color=color, fontweight='bold', fontsize=8, ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))
            
            # 在图表标题旁写极值范围
            title_text = f"{title}  [Max: {np.max(y_data):.2f}, Min: {np.min(y_data):.2f}]"
            ax.text(0.02,1.1 , title_text, transform=ax.transAxes, fontsize=9, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # --- Plot 2: Shear Diagram (SFD) ---
        plot_diagram(axs[1], V, "FS", f"Force ({self.u_force.get()})", '#1976d2', '#bbdefb') # Blue

        # --- Plot 3: Bending Moment (BMD) ---
        # 注意：弯矩图正负习惯。工程习惯上，受拉侧画图。
        # 此处采用标准数学坐标：M正值为"下凸"（sagging），画在正轴。
        plot_diagram(axs[2], M, "M", f"Moment ({self.u_moment.get()})", '#d32f2f', '#ffcdd2') # Red

        # --- Plot 4: Deflection ---
        plot_diagram(axs[3], Y, "w", f"Disp. ({self.u_def.get()})", '#388e3c', '#c8e6c9') # Green
        axs[3].set_xlabel(f"Position x ({self.u_len.get()})")
        
        # 嵌入到GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = BeamAnalysisApp(root)
    root.mainloop()
