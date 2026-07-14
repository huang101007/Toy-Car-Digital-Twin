# 第五版
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import threading
import time



#新增改善文字變方框(用全域變數的方法)
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 支援中文
plt.rcParams['axes.unicode_minus'] = False  # 修正負號變問號的問題

class ClockworkCarSimulator:
    #基礎設定(初始化之類的)
    def __init__(self, root):
        self.root = root
        self.root.title("機械概論第五組 發條車數位雙生模型")
        self.root.geometry("1700x1050")
        self.root.configure(bg='#2c3e50')



        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.setup_styles()

        # 新增參數
        self.model = None
        self.model_lateral = None
        self.current_model_type = "XGBoost"
        self.current_model_type_lateral = "Random Forest"
        self.feature_names = [
            'GearRatio', 'TransmissionSide', 'GearsNum', 'WheelNum',
            'MainWheelNum', 'NonmainWheelNum', 'DrivePosition',
            'WheelThickness', 'LoadPlate', 'Bearings', 'Weight', 'WindingLength'
        ]
        self.data = None
        self.X_test = None
        self.y_test = None
        self.X_test_lateral = None
        self.y_test_lateral = None
        self.feature_importance = None
        self.feature_importance_lateral = None
        self.target_var = "Distance"



        # 動畫相關變數
        self.is_simulating = False
        self.animation_thread = None

        self.encoding_maps = {
            'GearRatio': {0: '齒輪比1:1', 1: '齒輪比1:2', 2: '無齒輪'},
            'TransmissionSide': {0: '雙邊傳動', 1: '單邊傳動', 2: '無齒輪'},
            'GearsNum': {0: '三個齒輪', 1: '兩個齒輪', 2: '無齒輪'},
            'WheelNum': {0: '4輪', 1: '3輪'},
            'MainWheelNum': {0: '驅動輪直徑45mm', 1: '驅動輪直徑35mm'},
            'NonmainWheelNum': {0: '非驅動輪直徑45mm', 1: '非驅動輪直徑35mm'},
            'DrivePosition': {0: '前驅', 1: '後驅'},
            'WheelThickness': {0: '輪子厚度3.5mm', 1: '輪子厚度7mm'},
            'LoadPlate': {0: '有負載板', 1: '無負載板'},
            'Bearings': {0: '無軸承', 1: '有軸承'},
            'Weight': {0: '0g', 1: '50g', 2: '100g', 3: '150g'}
        }

        self.setup_gui()
    #風格樣式設定(自型顏色大小)
    def setup_styles(self):
        self.style.configure('Title.TLabel', font=('Microsoft JhengHei', 16, 'bold'),
                             foreground='white', background='#2c3e50')
        self.style.configure('Header.TLabel', font=('Microsoft JhengHei', 14, 'bold'),
                             foreground='#2c3e50', background='white')
        self.style.configure('Info.TLabel', font=('Microsoft JhengHei', 12),
                             foreground='#34495e', background='white')
        self.style.configure('Result.TLabel', font=('Microsoft JhengHei', 20, 'bold'),
                             foreground='#e74c3c', background='#ecf0f1')
        self.style.configure('Success.TLabel', font=('Microsoft JhengHei', 12),
                             foreground='#27ae60', background='white')
        self.style.configure('Custom.TNotebook', background='#34495e', tabposition='n')
        self.style.configure('Custom.TNotebook.Tab', padding=[20, 10],
                             font=('Microsoft JhengHei', 12, 'bold'))
    #主介面設計(最上面名稱和有四個分頁)
    def setup_gui(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = tk.Label(title_frame, text="機械概論第五組 發條車數位雙生模型",
                               font=('Microsoft JhengHei', 22, 'bold'),
                               fg='white', bg='#2c3e50')
        title_label.pack(expand=True, pady=(15, 5))

        subtitle_label = tk.Label(title_frame, text="Clockwork Car Digital Twin Simulation Platform",
                                  font=('Microsoft JhengHei', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack()

        main_container = tk.Frame(self.root, bg='#34495e')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        notebook.pack(fill='both', expand=True)

        design_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(design_frame, text='🎛️ 設計參數')

        result_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(result_frame, text='📊 模擬結果')

        # 新增車輛模擬頁面
        simulation_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(simulation_frame, text='🚗 車輛模擬')

        analysis_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(analysis_frame, text='📈 視覺化分析')

        self.setup_design_page(design_frame)
        self.setup_result_page(result_frame)
        self.setup_simulation_page(simulation_frame)
        self.setup_analysis_page(analysis_frame)
    #第一個介面{參數設定}設計版面
    def setup_design_page(self, parent):
        control_frame = tk.LabelFrame(parent, text="🎛️ 控制面板",
                                      font=('Microsoft JhengHei', 14, 'bold'),
                                      bg='white', fg='#2c3e50', bd=3, relief='groove')
        control_frame.pack(fill='x', padx=15, pady=15)

        top_control = tk.Frame(control_frame, bg='white')
        top_control.pack(fill='x', padx=15, pady=15)

        data_section = tk.Frame(top_control, bg='white')
        data_section.pack(fill='x', pady=(0, 10))

        tk.Button(data_section, text="📁 載入實驗資料", command=self.load_data,
                  bg='#3498db', fg='white', font=('Microsoft JhengHei', 12, 'bold'),
                  relief='flat', padx=25, pady=8, cursor='hand2').pack(side='left')

        self.data_status = tk.Label(data_section, text="等待載入實驗資料...",
                                    font=('Microsoft JhengHei', 12), fg='#7f8c8d', bg='white')
        self.data_status.pack(side='left', padx=(15, 0))

        # 預測目標選擇
        target_section = tk.Frame(top_control, bg='white')
        target_section.pack(fill='x', pady=(0, 10))

        tk.Label(target_section, text="🎯 預測目標:",
                 font=('Microsoft JhengHei', 12, 'bold'),
                 bg='white', fg='#2c3e50').pack(side='left')

        self.target_var_tk = tk.StringVar(value="Distance")
        target_combo = ttk.Combobox(target_section, textvariable=self.target_var_tk,
                                    values=["Distance", "LateralDeviation"],
                                    state="readonly", width=18, font=('Microsoft JhengHei', 11))
        target_combo.pack(side='left', padx=(15, 0))
        target_combo.bind("<<ComboboxSelected>>", self.on_target_change)

        # 模型選擇
        model_section = tk.Frame(top_control, bg='white')
        model_section.pack(fill='x')

        tk.Label(model_section, text="🤖 距離模型:",
                 font=('Microsoft JhengHei', 12, 'bold'),
                 bg='white', fg='#2c3e50').pack(side='left')

        self.model_var = tk.StringVar(value="XGBoost")
        model_combo = ttk.Combobox(model_section, textvariable=self.model_var,
                                   values=["XGBoost", "Random Forest", "Gradient Boosting"],
                                   state="readonly", width=18, font=('Microsoft JhengHei', 11))
        model_combo.pack(side='left', padx=(15, 0))

        tk.Label(model_section, text="🤖 偏移模型:",
                 font=('Microsoft JhengHei', 12, 'bold'),
                 bg='white', fg='#2c3e50').pack(side='left', padx=(30, 0))

        self.model_var_lateral = tk.StringVar(value="Random Forest")
        model_combo_lateral = ttk.Combobox(model_section, textvariable=self.model_var_lateral,
                                           values=["XGBoost", "Random Forest", "Gradient Boosting"],
                                           state="readonly", width=18, font=('Microsoft JhengHei', 11))
        model_combo_lateral.pack(side='left', padx=(15, 0))

        tk.Button(model_section, text="🔧 訓練模型", command=self.train_model,
                  bg='#27ae60', fg='white', font=('Microsoft JhengHei', 12, 'bold'),
                  relief='flat', padx=20, pady=6, cursor='hand2').pack(side='right')

        self.model_status = tk.Label(model_section, text="",
                                     font=('Microsoft JhengHei', 11),
                                     fg='#27ae60', bg='white')
        self.model_status.pack(side='right', padx=(0, 15))

        param_container = tk.Frame(parent, bg='#ecf0f1')
        param_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        left_params = tk.LabelFrame(param_container, text="⚙️ 設計參數設定",
                                    font=('Microsoft JhengHei', 14, 'bold'),
                                    bg='white', fg='#2c3e50', bd=3, relief='groove')
        left_params.pack(side='left', fill='both', expand=True, padx=(0, 8))

        right_result = tk.LabelFrame(param_container, text="🎯 即時預測結果",
                                     font=('Microsoft JhengHei', 14, 'bold'),
                                     bg='white', fg='#2c3e50', bd=3, relief='groove')
        right_result.pack(side='right', fill='y', padx=(8, 0))

        self.param_frame = left_params
        self.result_display_frame = right_result
        self.setup_result_display()
        #新增這個
        self.create_parameter_controls()
    #側向偏移及距離顯示(藍色及橘色區域綁訂到self.distance_var，和self.lateral_var
    def setup_result_display(self):
        # 距離預測顯示
        distance_container = tk.Frame(self.result_display_frame, bg='white')
        distance_container.pack(fill='x', padx=20, pady=20)

        tk.Label(distance_container, text="Predicted Distance",
                 font=('Microsoft JhengHei', 16, 'bold'),
                 bg='white', fg='#2c3e50').pack()

        self.distance_var = tk.StringVar(value="0.00")
        distance_display = tk.Frame(distance_container, bg='#3498db', relief='raised', bd=3)
        distance_display.pack(pady=15, padx=10)

        distance_value = tk.Label(distance_display, textvariable=self.distance_var,
                                  font=('Microsoft JhengHei', 28, 'bold'), fg='white', bg='#3498db')
        distance_value.pack(padx=30, pady=15)

        tk.Label(distance_display, text="cm",
                 font=('Microsoft JhengHei', 14, 'bold'),
                 fg='#ecf0f1', bg='#3498db').pack(pady=(0, 10))

        # 側向偏移預測顯示
        lateral_container = tk.Frame(self.result_display_frame, bg='white')
        lateral_container.pack(fill='x', padx=20, pady=10)

        tk.Label(lateral_container, text="Predicted Lateral Deviation",
                 font=('Microsoft JhengHei', 16, 'bold'),
                 bg='white', fg='#2c3e50').pack()

        self.lateral_var = tk.StringVar(value="0.00")
        lateral_display = tk.Frame(lateral_container, bg='#e67e22', relief='raised', bd=3)
        lateral_display.pack(pady=10, padx=10)

        lateral_value = tk.Label(lateral_display, textvariable=self.lateral_var,
                                 font=('Microsoft JhengHei', 28, 'bold'), fg='white', bg='#e67e22')
        lateral_value.pack(padx=30, pady=15)

        tk.Label(lateral_display, text="cm",
                 font=('Microsoft JhengHei', 14, 'bold'),
                 fg='#ecf0f1', bg='#e67e22').pack(pady=(0, 10))

    # 這部分和之後一直失敗，沒有開始鍵
    #"""設置車輛模擬頁面"""
    def setup_simulation_page(self, parent):
        """模擬頁面：顯示不可手動更改的預測距離與偏移 + 開始模擬按鈕 + 畫布動畫區域"""

        # ➤ 輸出預測值區塊（模擬顯示用）
        output_frame = tk.Frame(parent, bg='white')
        output_frame.pack(pady=20)

        # 距離顯示（藍色框）
        distance_display = tk.Frame(output_frame, bg='#3498db', relief='raised', bd=3)
        distance_display.grid(row=0, column=0, padx=30, pady=10)

        tk.Label(distance_display, text="預測距離", font=('Microsoft JhengHei', 14, 'bold'),
                 bg='#3498db', fg='white').pack(pady=(10, 0))

        self.sim_distance_var = tk.StringVar(value="0.00")
        tk.Label(distance_display, textvariable=self.sim_distance_var,
                 font=('Microsoft JhengHei', 28, 'bold'),
                 bg='#3498db', fg='white').pack(padx=30, pady=10)

        tk.Label(distance_display, text="cm", font=('Microsoft JhengHei', 12),
                 bg='#3498db', fg='white').pack(pady=(0, 10))

        # 側向偏移顯示（橘色框）
        lateral_display = tk.Frame(output_frame, bg='#e67e22', relief='raised', bd=3)
        lateral_display.grid(row=0, column=1, padx=30, pady=10)

        tk.Label(lateral_display, text="側向偏移", font=('Microsoft JhengHei', 14, 'bold'),
                 bg='#e67e22', fg='white').pack(pady=(10, 0))

        self.sim_lateral_var = tk.StringVar(value="0.00")
        tk.Label(lateral_display, textvariable=self.sim_lateral_var,
                 font=('Microsoft JhengHei', 28, 'bold'),
                 bg='#e67e22', fg='white').pack(padx=30, pady=10)

        tk.Label(lateral_display, text="cm", font=('Microsoft JhengHei', 12),
                 bg='#e67e22', fg='white').pack(pady=(0, 10))

        # ➤ 開始模擬按鈕
        self.start_sim_btn = tk.Button(parent,
                                       text="🚗 開始模擬",
                                       font=('Microsoft JhengHei', 13, 'bold'),
                                       bg='#27ae60', fg='white',
                                       command=self.start_car_simulation)
        self.start_sim_btn.pack(pady=10)

        # ➤ 動畫畫布區域
        sim_frame = tk.LabelFrame(parent, text="動畫區域", font=('Microsoft JhengHei', 12, 'bold'),
                                  bg='white', fg='black', bd=2)
        sim_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.sim_fig, self.sim_ax = plt.subplots(figsize=(10, 5))
        self.sim_fig.patch.set_facecolor('#f8f9fa')
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, sim_frame)
        self.sim_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.initialize_simulation_display()

    #初始化模擬
    def initialize_simulation_display(self):
        """初始化模擬顯示"""
        self.sim_ax.clear()

        # 設置基本跑道
        track_length = 100  # 預設100cm
        track_width = 8

        # 繪製跑道背景
        self.sim_ax.add_patch(patches.Rectangle((-5, -track_width / 2),
                                                track_length + 10, track_width,
                                                facecolor='#bdc3c7', alpha=0.3,
                                                edgecolor='#7f8c8d', linewidth=2))

        # 起點線
        self.sim_ax.axvline(x=0, color='#27ae60', linewidth=4, label='起點')

        # 終點線
        self.sim_ax.axvline(x=track_length, color='#e74c3c', linewidth=4, label='終點')

        # 繪製初始車輛
        self.draw_car(0, 0)

        # 設置刻度
        self.setup_track_scale(track_length)

        # 設置軸屬性
        self.sim_ax.set_xlim(-10, track_length + 10)
        self.sim_ax.set_ylim(-10, 10)
        self.sim_ax.set_xlabel('距離 (cm)', fontsize=14, fontweight='bold')
        self.sim_ax.set_ylabel('側向偏移 (cm)', fontsize=14, fontweight='bold')
        self.sim_ax.set_title('發條車模擬動畫 - 等待開始', fontsize=18, fontweight='bold')
        self.sim_ax.grid(True, alpha=0.3)
        self.sim_ax.legend(loc='upper right', fontsize=12)

        self.sim_canvas.draw()
    #車道
    def setup_track_scale(self, max_distance):
        """設置跑道刻度"""
        # 動態計算刻度間隔
        if max_distance <= 100:
            interval = 10
        elif max_distance <= 500:
            interval = 50
        elif max_distance <= 1000:
            interval = 100
        else:
            interval = 200

        # 繪製刻度線和標籤
        for i in range(0, int(max_distance) + interval, interval):
            if i <= max_distance:
                self.sim_ax.axvline(x=i, color='#95a5a6', alpha=0.6, linestyle='--', linewidth=1)
                self.sim_ax.text(i, -8, f'{i}cm', ha='center', va='top',
                                 fontsize=10, fontweight='bold', color='#2c3e50')
    #畫車
    def draw_car(self, position, lateral_offset):
        """畫一台固定尺寸的車：不會因為坐標軸縮放而變形"""
        from matplotlib.transforms import Affine2D

        # 實際位置 (車中心座標)
        car_x, car_y = position, lateral_offset

        # 設定畫面中固定大小 (例如寬3cm、高1.5cm)
        car_width = 3
        car_height = 1.5

        # 使用轉換，使車輛大小不會隨座標變化
        trans = Affine2D().translate(car_x - car_width / 2, car_y - car_height / 2) + self.sim_ax.transData

        car = patches.Rectangle((0, 0), car_width, car_height,
                                facecolor='#2980b9', edgecolor='black',
                                linewidth=1.5, transform=trans, zorder=5)

        self.sim_ax.add_patch(car)

    #開始按鍵
    def start_car_simulation(self):
        """開始簡化車輛動畫模擬"""
        if self.is_simulating:
            return

        try:
            distance = float(self.distance_var.get())
            lateral = float(self.lateral_var.get())
            self.sim_distance_var.set(f"{distance:.1f}")
            self.sim_lateral_var.set(f"{lateral:.1f}")
        except:
            distance, lateral = 100.0, 0.0
            self.sim_distance_var.set("100.0")
            self.sim_lateral_var.set("0.0")

        self.is_simulating = True
        self.start_sim_btn.config(text="🔄 模擬中...", state='disabled', bg='#95a5a6')

        threading.Thread(target=self.run_car_animation, args=(distance, lateral)).start()
    #模擬畫面
    def run_car_animation(self, target_distance, target_lateral):
        """運行簡化車輛動畫"""
        steps = 30
        delay = 0.1

        # 動態調整座標軸範圍
        x_margin = 10
        y_margin = 5
        x_limit = target_distance * 1.2 + x_margin
        y_limit = max(abs(target_lateral) * 1.2 + y_margin, 10)  # 確保最小高度

        for step in range(steps + 1):
            if not self.is_simulating:
                break

            progress = step / steps
            x = target_distance * progress
            y = target_lateral * progress

            self.sim_ax.clear()

            # 繪製背景跑道（根據動態大小）
            self.sim_ax.add_patch(patches.Rectangle(
                (-5, -y_limit),
                x_limit + 10, 2 * y_limit,
                facecolor='#ecf0f1', alpha=0.3,
                edgecolor='#7f8c8d', linewidth=2
            ))

            # 起點 & 終點線
            self.sim_ax.axvline(x=0, color='green', linewidth=3, label='起點')
            self.sim_ax.axvline(x=target_distance, color='red', linewidth=3, label='目標')

            # 車輛行進軌跡
            if step > 0:
                trajectory_x = [target_distance * i / steps for i in range(step + 1)]
                trajectory_y = [target_lateral * i / steps for i in range(step + 1)]
                self.sim_ax.plot(trajectory_x, trajectory_y, 'r--', linewidth=2, alpha=0.7, label='軌跡')

            self.draw_car(x, y)
            self.setup_track_scale(target_distance)

            # 動態設定座標軸範圍
            self.sim_ax.set_xlim(-10, x_limit)
            lateral_margin = max(abs(target_lateral) * 1.2, 10)
            self.sim_ax.set_ylim(-y_limit, y_limit)

            self.sim_ax.set_xlabel('距離 (cm)', fontsize=12)
            self.sim_ax.set_ylabel('側向偏移 (cm)', fontsize=12)
            self.sim_ax.set_title(f'位置: {x:.1f} cm, 偏移: {y:.1f} cm', fontsize=14)
            self.sim_ax.grid(True, alpha=0.3)
            self.sim_ax.legend(loc='upper right')

            self.sim_canvas.draw()
            time.sleep(delay)

        self.is_simulating = False
        self.start_sim_btn.config(text="🚗 開始動畫模擬", state='normal', bg='#27ae60')

    #text = "📈 模型效能評估",
    def setup_result_page(self, parent):
        performance_frame = tk.LabelFrame(parent, text="📈 模型效能評估",
                                          font=('Microsoft JhengHei', 14, 'bold'),
                                          bg='white', fg='#2c3e50', bd=3, relief='groove')
        performance_frame.pack(fill='x', padx=15, pady=15)

        self.performance_text = tk.Text(performance_frame, height=12, width=80,
                                        font=('Consolas', 11), bg='#f8f9fa',
                                        relief='flat', bd=2, wrap='word')
        self.performance_text.pack(padx=15, pady=15, fill='x')

        importance_frame = tk.LabelFrame(parent, text="🔍 特徵重要性詳細分析",
                                         font=('Microsoft JhengHei', 14, 'bold'),
                                         bg='white', fg='#2c3e50', bd=3, relief='groove')
        importance_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        self.importance_text = tk.Text(importance_frame, height=15, width=80,
                                       font=('Microsoft JhengHei', 11), bg='#f8f9fa',
                                       relief='flat', bd=2, wrap='word')
        self.importance_text.pack(padx=15, pady=15, fill='both', expand=True)

    #text = "📊 視覺化分析圖表",
    def setup_analysis_page(self, parent):
        chart_frame = tk.LabelFrame(parent, text="📊 視覺化分析圖表",
                                    font=('Microsoft JhengHei', 14, 'bold'),
                                    bg='white', fg='#2c3e50', bd=3, relief='groove')
        chart_frame.pack(fill='both', expand=True, padx=15, pady=15)

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.patch.set_facecolor('white')

        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=15)

    # title="選擇發條車實驗資料CSV檔案"
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="選擇發條車實驗資料CSV檔案",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data = pd.read_csv(file_path, encoding='utf-8')
                required_cols = ['Set'] + self.feature_names + ['Distance', 'LateralDeviation']
                missing_cols = [col for col in required_cols if col not in self.data.columns]

                if missing_cols:
                    messagebox.showerror("資料格式錯誤", f"缺少以下欄位: {', '.join(missing_cols)}")
                    return

                self.data_status.config(text=f"✅ 已載入 {len(self.data)} 筆實驗資料", fg='#27ae60')
                self.create_parameter_controls()

                stats_info = f"實驗資料統計:\n"
                stats_info += f"• 實驗次數: {len(self.data)} 次\n"
                stats_info += f"• 距離範圍: {self.data['Distance'].min():.1f} - {self.data['Distance'].max():.1f} cm\n"
                stats_info += f"• 側向偏移範圍: {self.data['LateralDeviation'].min():.1f} - {self.data['LateralDeviation'].max():.1f} cm\n"
                stats_info += f"• 平均距離: {self.data['Distance'].mean():.2f} cm\n"
                stats_info += f"• 平均偏移: {self.data['LateralDeviation'].mean():.2f} cm"

                messagebox.showinfo("資料載入成功", f"成功載入發條車實驗資料！\n\n{stats_info}")
            except Exception as e:
                messagebox.showerror("載入錯誤", f"載入檔案時發生錯誤:\n{str(e)}")
    #控制參數介面該有的按鍵(第一頁左下)
    def create_parameter_controls(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.param_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        scrollbar.pack(side="right", fill="y", padx=(0, 5))

        self.param_vars = {}
        self.no_gear_vars = []

        param_groups = {
            "🔧 傳動系統設定": ['GearRatio', 'TransmissionSide', 'GearsNum'],
            "🛞 輪系配置設定": ['WheelNum', 'DrivePosition'],
            "📏 輪子尺寸設定": ['MainWheelNum', 'NonmainWheelNum', 'WheelThickness'],
            "⚖️ 車體設計設定": ['LoadPlate', 'Bearings', 'Weight'],
            "🌀 發條拉長設定": ['WindingLength']
        }

        row = 0
        for group_name, features in param_groups.items():
            group_label = tk.Label(scrollable_frame, text=group_name,
                                   font=('Microsoft JhengHei', 13, 'bold'),
                                   bg='#3498db', fg='white', pady=8)
            group_label.grid(row=row, column=0, columnspan=4, sticky='ew', pady=(10, 0))
            row += 1

            for feature in features:
                if self.data is not None and feature in self.data.columns:
                    self.create_parameter_widget(scrollable_frame, feature, row)
                    row += 1

            separator = tk.Frame(scrollable_frame, height=10, bg='white')
            separator.grid(row=row, column=0, columnspan=4, sticky='ew')
            row += 1

        scrollable_frame.columnconfigure(1, weight=1)
        scrollable_frame.columnconfigure(2, weight=1)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # 這附近到最底可能需要縮排才能過…
    #這些按鍵要用點的或拖的
    def create_parameter_widget(self, parent, feature, row):
        unique_values = sorted(self.data[feature].unique())
        param_name = self.get_param_display_name(feature)
        importance_text = ""

        if self.feature_importance is not None and feature in self.feature_names:
            idx = self.feature_names.index(feature)
            importance = self.feature_importance[idx]
            importance_text = f" (重要性: {importance:.3f})"

        name_label = tk.Label(parent, text=f"• {param_name}{importance_text}",
                              font=('Microsoft JhengHei', 12, 'bold'),
                              bg='white', fg='#2c3e50')
        name_label.grid(row=row, column=0, sticky='w', padx=(10, 15), pady=8)

        if feature in ['MainWheelNum', 'NonmainWheelNum']:
            self.create_wheel_diameter_control(parent, feature, row)
        elif feature == 'WheelThickness':
            self.create_wheel_thickness_control(parent, feature, row)
        elif feature == 'Weight':
            self.create_weight_control(parent, feature, row)
        elif feature == 'WindingLength':
            self.create_winding_length_control(parent, feature, row)
        else:
            self.create_radio_control(parent, feature, unique_values, row)
    #參數名稱變中文
    def get_param_display_name(self, feature):
        name_map = {
            'GearRatio': '齒輪比例',
            'TransmissionSide': '傳動方式',
            'GearsNum': '齒輪數量',
            'WheelNum': '輪子數量',
            'MainWheelNum': '驅動輪直徑',
            'NonmainWheelNum': '非驅動輪直徑',
            'DrivePosition': '驅動位置',
            'WheelThickness': '輪子厚度',
            'LoadPlate': '負載板',
            'Bearings': '軸承配置',
            'Weight': '載重重量',
            'WindingLength': '發條拉長'
        }
        return name_map.get(feature, feature)
    #輪直徑控制條
    def create_wheel_diameter_control(self, parent, feature, row):
        var = tk.DoubleVar(value=40.0)
        self.param_vars[feature] = var
        control_frame = tk.Frame(parent, bg='white')
        control_frame.grid(row=row, column=1, columnspan=2, sticky='ew', padx=15, pady=5)

        scale = tk.Scale(control_frame, from_=35, to=45, resolution=0.5, orient='horizontal',
                         variable=var, length=250, font=('Microsoft JhengHei', 10),
                         bg='white', fg='#2c3e50', highlightbackground='white',
                         command=lambda x: self.update_prediction())
        scale.pack(side='left')

        unit_label = tk.Label(control_frame, text="mm", font=('Microsoft JhengHei', 11),
                              bg='white', fg='#7f8c8d')
        unit_label.pack(side='left', padx=(5, 0))
    # 輪厚度控制條
    def create_wheel_thickness_control(self, parent, feature, row):
        var = tk.DoubleVar(value=5.25)
        self.param_vars[feature] = var
        control_frame = tk.Frame(parent, bg='white')
        control_frame.grid(row=row, column=1, columnspan=2, sticky='ew', padx=15, pady=5)

        scale = tk.Scale(control_frame, from_=3.5, to=7.0, resolution=0.25, orient='horizontal',
                         variable=var, length=250, font=('Microsoft JhengHei', 10),
                         bg='white', fg='#2c3e50', highlightbackground='white',
                         command=lambda x: self.update_prediction())
        scale.pack(side='left')

        unit_label = tk.Label(control_frame, text="mm", font=('Microsoft JhengHei', 11),
                              bg='white', fg='#7f8c8d')
        unit_label.pack(side='left', padx=(5, 0))
    #重量控制條
    def create_weight_control(self, parent, feature, row):
        var = tk.IntVar(value=75)
        self.param_vars[feature] = var
        control_frame = tk.Frame(parent, bg='white')
        control_frame.grid(row=row, column=1, columnspan=2, sticky='ew', padx=15, pady=5)

        scale = tk.Scale(control_frame, from_=0, to=150, resolution=5, orient='horizontal',
                         variable=var, length=250, font=('Microsoft JhengHei', 10),
                         bg='white', fg='#2c3e50', highlightbackground='white',
                         command=lambda x: self.update_prediction())
        scale.pack(side='left')

        unit_label = tk.Label(control_frame, text="g", font=('Microsoft JhengHei', 11),
                              bg='white', fg='#7f8c8d')
        unit_label.pack(side='left', padx=(5, 0))
    #發條拉的長度控制條
    def create_winding_length_control(self, parent, feature, row):
        var = tk.DoubleVar(value=10.0)
        self.param_vars[feature] = var
        control_frame = tk.Frame(parent, bg='white')
        control_frame.grid(row=row, column=1, columnspan=2, sticky='ew', padx=15, pady=5)

        scale = tk.Scale(control_frame, from_=0, to=20, resolution=0.5, orient='horizontal',
                         variable=var, length=250, font=('Microsoft JhengHei', 10),
                         bg='white', fg='#2c3e50', highlightbackground='white',
                         command=lambda x: self.update_prediction())
        scale.pack(side='left')

        unit_label = tk.Label(control_frame, text="cm", font=('Microsoft JhengHei', 11),
                              bg='white', fg='#7f8c8d')
        unit_label.pack(side='left', padx=(5, 0))
    #參數單選按鈕
    def create_radio_control(self, parent, feature, unique_values, row):
        var = tk.IntVar(value=unique_values[0])
        self.param_vars[feature] = var
        button_frame = tk.Frame(parent, bg='white')
        button_frame.grid(row=row, column=1, columnspan=2, sticky='w', padx=15, pady=5)

        for i, value in enumerate(unique_values):
            option_text = self.encoding_maps.get(feature, {}).get(value, str(value))
            rb = tk.Radiobutton(button_frame, text=option_text, variable=var, value=value,
                                bg='white', fg='#2c3e50', font=('Microsoft JhengHei', 11),
                                selectcolor='#3498db', cursor='hand2',
                                command=lambda: self.handle_gear_selection(feature, var.get()))
            rb.pack(side='left', padx=(0, 20))

            if '無齒輪' in option_text:
                self.no_gear_vars.append((feature, var, value))
    #無齒輪要一起改
    def handle_gear_selection(self, selected_feature, selected_value):
        if (selected_feature in ['GearRatio', 'TransmissionSide', 'GearsNum'] and
                selected_value == 2):
            for feature, var, no_gear_value in self.no_gear_vars:
                if feature in ['GearRatio', 'TransmissionSide', 'GearsNum']:
                    var.set(2)
        self.update_prediction()
    #若重新點按鍵就要重新預測結果(第157行ComboboxSelected)
    def on_target_change(self, event):
        self.target_var = self.target_var_tk.get()
        self.update_prediction()
    #訓練模型
    def train_model(self):
        #確認已經匯入資料
        if self.data is None:
            messagebox.showerror("錯誤", "請先載入實驗資料！")
            return

        try:
            # 權重處理
            weights = np.ones(len(self.data))
            if 'WindingLength' in self.data.columns:
                set_col = self.data['Set']
                weights[set_col == 27] *= 5  # set27加權

            # 訓練距離模型
            X = self.data[self.feature_names]
            y = self.data['Distance']
            X_train, self.X_test, y_train, self.y_test, w_train, _ = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )

            self.current_model_type = self.model_var.get()
            if self.current_model_type == "XGBoost":
                self.model = xgb.XGBRegressor(
                    learning_rate=0.1, max_depth=7, n_estimators=50, random_state=42
                )
            elif self.current_model_type == "Random Forest":
                self.model = RandomForestRegressor(
                    max_depth=10, n_estimators=100, random_state=42
                )
            else:
                self.model = GradientBoostingRegressor(
                    learning_rate=0.1, n_estimators=100, random_state=42
                )

            self.model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            self.model_status.config(text=f"✅ 距離模型已訓練完成")

            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_

            # 訓練側向偏移模型
            y_lateral = self.data['LateralDeviation']
            X_train_l, self.X_test_lateral, y_train_l, self.y_test_lateral, w_train_l, _ = train_test_split(
                X, y_lateral, weights, test_size=0.2, random_state=42
            )

            self.current_model_type_lateral = self.model_var_lateral.get()
            if self.current_model_type_lateral == "XGBoost":
                self.model_lateral = xgb.XGBRegressor(
                    learning_rate=0.1, max_depth=7, n_estimators=50, random_state=42
                )
            elif self.current_model_type_lateral == "Random Forest":
                self.model_lateral = RandomForestRegressor(
                    max_depth=10, n_estimators=100, random_state=42
                )
            else:
                self.model_lateral = GradientBoostingRegressor(
                    learning_rate=0.1, n_estimators=100, random_state=42
                )

            self.model_lateral.fit(X_train_l, y_train_l, sample_weight=w_train_l)
            y_pred_lateral = self.model_lateral.predict(self.X_test_lateral)
            mse_l = mean_squared_error(self.y_test_lateral, y_pred_lateral)
            r2_l = r2_score(self.y_test_lateral, y_pred_lateral)

            if hasattr(self.model_lateral, 'feature_importances_'):
                self.feature_importance_lateral = self.model_lateral.feature_importances_

            self.display_model_performance(r2, mse, y_pred, r2_l, mse_l, y_pred_lateral)
            self.update_charts(y_pred)
            self.create_parameter_controls()
            self.update_prediction()

            messagebox.showinfo("模型訓練完成",
                                f"距離模型訓練完成！\n模型準確度: {r2:.1%}\n"
                                f"偏移模型訓練完成！\n模型準確度: {r2_l:.1%}\n"
                                f"現在可以調整設計參數進行模擬預測。")
        except Exception as e:
            messagebox.showerror("訓練錯誤", f"訓練模型時發生錯誤:\n{str(e)}")
    #效能報告(第二頁上半部文字)
    def display_model_performance(self, r2, mse, y_pred, r2_l, mse_l, y_pred_lateral):
        performance_text = f"🤖 距離模型效能評估報告\n"
        performance_text += "=" * 60 + "\n\n"
        performance_text += f"R² Score: {r2:.4f}\nMSE: {mse:.4f}\n"
        performance_text += f"RMSE: {np.sqrt(mse):.4f}\nMAE: {np.mean(np.abs(self.y_test - y_pred)):.4f}\n\n"

        performance_text += f"🤖 側向偏移模型效能評估報告\n"
        performance_text += "=" * 60 + "\n\n"
        performance_text += f"R² Score: {r2_l:.4f}\nMSE: {mse_l:.4f}\n"
        performance_text += f"RMSE: {np.sqrt(mse_l):.4f}\nMAE: {np.mean(np.abs(self.y_test_lateral - y_pred_lateral)):.4f}\n"

        self.performance_text.delete(1.0, tk.END)
        self.performance_text.insert(1.0, performance_text)
        self.display_feature_importance()
    #重要性評估(第二頁下半)
    def display_feature_importance(self):
        if self.feature_importance is None:
            return

        importance_text = f"🔍 距離模型特徵重要性\n"
        importance_text += "=" * 60 + "\n"
        feature_importance_pairs = list(zip(self.feature_names, self.feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(feature_importance_pairs, 1):
            param_name = self.get_param_display_name(feature)
            importance_text += f"{i:2d}. {param_name:<12} 重要性: {importance:.4f}\n"

        if self.feature_importance_lateral is not None:
            importance_text += "\n🔍 側向偏移模型特徵重要性\n"
            importance_text += "=" * 60 + "\n"
            feature_importance_pairs_l = list(zip(self.feature_names, self.feature_importance_lateral))
            feature_importance_pairs_l.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, importance) in enumerate(feature_importance_pairs_l, 1):
                param_name = self.get_param_display_name(feature)
                importance_text += f"{i:2d}. {param_name:<12} 重要性: {importance:.4f}\n"

        self.importance_text.delete(1.0, tk.END)
        self.importance_text.insert(1.0, importance_text)
    #即時更新預測結果
    def update_prediction(self, *args):
        if self.model is None or self.model_lateral is None:
            print("模型未載入，無法進行預測")
            return

        try:
            params = []
            for feature in self.feature_names:
                if feature in self.param_vars:
                    value = self.param_vars[feature].get()
                    if feature == 'MainWheelNum':
                        params.append(0 if value >= 40 else 1)
                    elif feature == 'NonmainWheelNum':
                        params.append(0 if value >= 40 else 1)
                    elif feature == 'WheelThickness':
                        params.append(0 if value <= 5.25 else 1)
                    elif feature == 'Weight':
                        if value <= 25:
                            params.append(0)
                        elif value <= 75:
                            params.append(1)
                        elif value <= 125:
                            params.append(2)
                        else:
                            params.append(3)
                    elif feature == 'WindingLength':
                        params.append(float(value))
                    else:
                        params.append(int(value))
                else:
                    params.append(0)
            #多下面這行
            X = pd.DataFrame([params], columns=self.feature_names)
            pred_distance = self.model.predict(X)[0]
            pred_distance = max(0, pred_distance)
            pred_lateral = self.model_lateral.predict(X)[0]  # 與 distance 使用相同 X，保持一致

            self.update_car_visualization(pred_distance, pred_lateral)

        except Exception as e:
            print(f"預測時發生錯誤: {str(e)}")
    #更新圖表視覺化(第四頁)
    def update_charts(self, y_pred):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        plt.style.use('seaborn-v0_8-whitegrid')
        self.ax1.scatter(self.y_test, y_pred, alpha=0.7, color='#3498db', s=60, edgecolors='white', linewidth=0.5)
        self.ax1.plot([self.y_test.min(), self.y_test.max()],
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2, alpha=0.8)
        self.ax1.set_xlabel('Actual Distance (cm)', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Predicted Distance (cm)', fontsize=12, fontweight='bold')
        self.ax1.set_title('Prediction Accuracy Analysis', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)

        r2 = r2_score(self.y_test, y_pred)
        self.ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=self.ax1.transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=12, fontweight='bold')

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            indices = np.argsort(importance)
            simplified_names = []

            for i in indices:
                name = self.feature_names[i]
                name_map = {
                    'GearRatio': 'Gear Ratio',
                    'TransmissionSide': 'Transmission',
                    'GearsNum': 'Gear Number',
                    'WheelNum': 'Wheel Number',
                    'MainWheelNum': 'Drive Wheel',
                    'NonmainWheelNum': 'Non-drive Wheel',
                    'DrivePosition': 'Drive Position',
                    'WheelThickness': 'Wheel Thickness',
                    'LoadPlate': 'Load Plate',
                    'Bearings': 'Bearings',
                    'Weight': 'Weight',
                    'WindingLength': 'Winding Length'
                }
                simplified_names.append(name_map.get(name, name))

            colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
            bars = self.ax2.barh(range(len(importance)), importance[indices], color=colors[indices])
            self.ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            self.ax2.set_ylabel('Design Parameters', fontsize=12, fontweight='bold')
            self.ax2.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
            self.ax2.set_yticks(range(len(importance)))
            self.ax2.set_yticklabels(simplified_names, fontsize=10)
            self.ax2.grid(True, alpha=0.3, axis='x')

        residuals = self.y_test - y_pred
        self.ax3.scatter(y_pred, residuals, alpha=0.7, color='#e74c3c', s=60, edgecolors='white', linewidth=0.5)
        self.ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
        self.ax3.set_xlabel('Predicted Distance (cm)', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Residuals (cm)', fontsize=12, fontweight='bold')
        self.ax3.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.hist(self.y_test, bins=15, alpha=0.7, color='#2ecc71', edgecolor='white', linewidth=1)
        self.ax4.axvline(self.y_test.mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {self.y_test.mean():.2f} cm')
        self.ax4.set_xlabel('Distance (cm)', fontsize=12, fontweight='bold')
        self.ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        self.ax4.set_title('Distance Distribution', fontsize=14, fontweight='bold')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self.canvas.draw()
    #新增測試區域
    def update_car_visualization(self, distance, lateral):
        print("更新距離：", distance)
        print("目前 self.distance_var =", self.distance_var)
        self.distance_var.set(f"{distance:.2f}")
        self.lateral_var.set(f"{lateral:.2f}")
        self.sim_distance_var.set(f"{distance:.2f}")
        self.sim_lateral_var.set(f"{lateral:.2f}")

def main():
    root = tk.Tk()
    app = ClockworkCarSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()



