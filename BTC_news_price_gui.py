# BTC_news_price_gui_optimized.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, pipeline
import pandas as pd
from datetime import datetime, timezone
import os
import warnings
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter

# 强制中文支持（解决图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 全局配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "news_price_model.pth")
KLINE_CSV = os.path.join(SCRIPT_DIR, "btc_4h_2018.csv")
COUNT_PATH = os.path.join(SCRIPT_DIR, "learn_count.txt")
HISTORY_CSV = os.path.join(SCRIPT_DIR, "fed_news_history.csv")  # 历史记录文件

PRICE_MIN = 3000.0
PRICE_MAX = 20000.0
USE_NORMALIZATION = True

# 情绪分析pipeline：换成更正面友好的 twitter-roberta 模型
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


class NewsToPriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("ProsusAI/finbert")
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.fc = nn.Linear(768, 1)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.bert(**inputs)
        pooled = outputs.pooler_output
        price = self.fc(pooled)
        price = torch.nn.functional.softplus(price)
        return price.squeeze(-1)


def load_kline(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamps'])
        df = df.set_index('timestamps').sort_index()
        return df
    except Exception as e:
        print(f"加载K线错误: {e}")
        raise


def get_price_at_time(df_kline, dt):
    if df_kline is None:
        raise ValueError("K线数据未加载")
    if dt in df_kline.index:
        return df_kline.at[dt, 'close']
    pos = df_kline.index.get_indexer([dt], method='nearest')[0]
    if pos == -1:
        return df_kline['close'].iloc[0] if dt < df_kline.index[0] else df_kline['close'].iloc[-1]
    return df_kline['close'].iloc[pos]


def normalize_price(p):
    if not USE_NORMALIZATION:
        return p
    return (p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)


def denormalize_price(p_norm):
    if not USE_NORMALIZATION:
        return p_norm
    return p_norm * (PRICE_MAX - PRICE_MIN) + PRICE_MIN


def analyze_sentiment(text):
    """使用 twitter-roberta 模型分析情绪，返回 -10 到 +10 分数"""
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()  # positive / neutral / negative
    score = result['score']
    if 'positive' in label:
        return round(score * 10, 2)
    elif 'negative' in label:
        return round(-score * 10, 2)
    else:
        return 0.0  # neutral


class BTCNewsPriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("新闻 → BTC 价格预测系统（情绪对比优化版）")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.df_kline = None
        self.is_training = False
        self.learn_count = 0
        self.loss_history = []
        self.avg_loss = 0.0
        self.news_records = {}  # key: (date_str, title), value: (sentiment, true_price, pred_price)

        self.load_learn_count()
        self.load_news_records()

        self.create_widgets()
        self.load_model_and_data()
        self.update_status()
        self.plot_emotion_price_comparison()

    def load_learn_count(self):
        if os.path.exists(COUNT_PATH):
            try:
                with open(COUNT_PATH, 'r') as f:
                    self.learn_count = int(f.read().strip())
            except:
                self.learn_count = 0
        else:
            self.learn_count = 0

    def save_learn_count(self):
        try:
            with open(COUNT_PATH, 'w') as f:
                f.write(str(self.learn_count))
        except:
            pass

    def load_news_records(self):
        self.news_records = {}
        if os.path.exists(HISTORY_CSV):
            try:
                df = pd.read_csv(HISTORY_CSV)
                for _, row in df.iterrows():
                    key = (row['date'], row['title'])
                    if key not in self.news_records:
                        self.news_records[key] = (
                            row['sentiment'],
                            row['true_price'],
                            row['pred_price']
                        )
                print(f"已加载 {len(self.news_records)} 条唯一历史新闻记录")
            except Exception as e:
                print(f"加载历史记录失败: {e}")

    def save_news_record(self, date, title, sentiment, true_price, pred_price):
        key = (date.strftime('%Y-%m-%d'), title)
        if key not in self.news_records:
            self.news_records[key] = (sentiment, true_price, pred_price)
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'title': title,
                'sentiment': sentiment,
                'true_price': true_price,
                'pred_price': pred_price
            }
            if os.path.exists(HISTORY_CSV):
                df = pd.read_csv(HISTORY_CSV)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_csv(HISTORY_CSV, index=False)

    def load_model_and_data(self):
        self.model = NewsToPriceModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-6)

        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
                self.status_label.configure(
                    text=f"模型已加载 ({os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB)",
                    foreground="green"
                )
            except Exception as e:
                self.status_label.configure(text=f"模型加载失败: {e}", foreground="red")
        else:
            self.status_label.configure(text="未找到模型，使用全新初始化", foreground="orange")

        try:
            self.df_kline = load_kline(KLINE_CSV)
            self.data_status.configure(
                text=f"K线数据加载成功 ({len(self.df_kline)} 条记录)",
                foreground="green"
            )
        except Exception as e:
            self.data_status.configure(text=f"K线加载失败: {e}", foreground="red")
            messagebox.showerror("K线加载失败", str(e))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.LabelFrame(main_frame, text="当前状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="初始化中...", font=("Helvetica", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)

        self.learn_count_label = ttk.Label(status_frame, text=f"已学习新闻: {self.learn_count} 篇", font=("Helvetica", 10))
        self.learn_count_label.pack(side=tk.RIGHT, padx=20)

        self.avg_loss_label = ttk.Label(status_frame, text="平均 Loss: —", font=("Helvetica", 10))
        self.avg_loss_label.pack(side=tk.RIGHT, padx=20)

        input_frame = ttk.LabelFrame(main_frame, text="输入新闻 & 情绪分析", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="标题:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.title_entry = ttk.Entry(input_frame, width=80)
        self.title_entry.grid(row=0, column=1, sticky=tk.EW, pady=5)

        ttk.Label(input_frame, text="日期 (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.date_entry = ttk.Entry(input_frame, width=30)
        self.date_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(input_frame, text="内容 (可选):").grid(row=2, column=0, sticky=tk.NW, pady=5)
        self.content_text = scrolledtext.ScrolledText(input_frame, width=80, height=8, wrap=tk.WORD)
        self.content_text.grid(row=2, column=1, sticky=tk.EW, pady=5)

        # 情绪分析按钮
        ttk.Button(input_frame, text="分析情绪 (-10~+10)", command=self.analyze_sentiment_only).grid(row=3, column=0, columnspan=2, pady=10)

        input_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.train_btn = ttk.Button(btn_frame, text="学习此条新闻", command=self.start_learn_thread)
        self.train_btn.pack(side=tk.LEFT, padx=10)

        self.predict_btn = ttk.Button(btn_frame, text="仅预测此条新闻", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(btn_frame, text="清空输入", command=self.clear_input).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=10)

        output_frame = ttk.LabelFrame(main_frame, text="结果 & 学习进度", padding=10)
        output_frame.pack(fill=tk.X, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.output_text.pack(fill=tk.X, pady=5)

        chart_frame = ttk.LabelFrame(main_frame, text="新闻情绪 vs BTC价格历史对比（-10~+10情绪 / 价格归一化）", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)

        self.data_status = ttk.Label(info_frame, text="K线数据加载中...", foreground="gray")
        self.data_status.pack(side=tk.LEFT)

    def analyze_sentiment_only(self):
        text, _, title = self.validate_input()
        if not text:
            return

        sentiment = analyze_sentiment(text)
        msg = f"情绪分析结果（-10负面 ~ +10正面）:\n"
        msg += f"标题: {title}\n"
        msg += f"情绪分数: {sentiment:.2f}\n"
        msg += "—" * 50 + "\n"
        self.append_output(msg, "blue" if sentiment > 0 else "red")

    def append_output(self, text, color="black"):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.tag_add(color, "end-1l", "end")
        self.output_text.tag_config("green", foreground="green")
        self.output_text.tag_config("red", foreground="red")
        self.output_text.tag_config("blue", foreground="blue")
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def clear_input(self):
        self.title_entry.delete(0, tk.END)
        self.date_entry.delete(0, tk.END)
        self.content_text.delete("1.0", tk.END)

    def validate_input(self):
        title = self.title_entry.get().strip()
        date_str = self.date_entry.get().strip()
        content = self.content_text.get("1.0", tk.END).strip()

        if not title:
            messagebox.showwarning("输入错误", "标题不能为空！")
            return None, None, None

        if not date_str:
            messagebox.showwarning("输入错误", "日期不能为空！")
            return None, None, None

        try:
            dt_naive = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt_naive.replace(tzinfo=timezone.utc)
        except ValueError:
            messagebox.showwarning("输入错误", "日期格式错误！请使用 YYYY-MM-DD")
            return None, None, None

        text = title + (" " + content if content else "")
        return text, dt, title

    def start_learn_thread(self):
        if self.is_training:
            return
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.learn_one, daemon=True).start()

    def learn_one(self):
        if self.df_kline is None:
            messagebox.showerror("数据未加载", "K线数据尚未加载成功，无法进行学习")
            self.is_training = False
            self.train_btn.config(state=tk.NORMAL)
            return

        text, dt, title = self.validate_input()
        if not text:
            self.is_training = False
            self.train_btn.config(state=tk.NORMAL)
            return

        try:
            sentiment = analyze_sentiment(text)

            true_price_raw = get_price_at_time(self.df_kline, dt)
            true_price_norm = normalize_price(true_price_raw)
            true_price = torch.tensor([true_price_norm], dtype=torch.float32)

            self.model.train()
            pred_norm = self.model([text])
            loss = self.criterion(pred_norm, true_price)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_raw = denormalize_price(pred_norm.item())
            loss_val = loss.item()

            self.learn_count += 1
            self.save_learn_count()

            self.loss_history.append(loss_val)
            if len(self.loss_history) > 10:
                self.loss_history.pop(0)
            self.avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0

            self.save_news_record(dt, title, sentiment, true_price_raw, pred_raw)

            msg = f"本次学习完成！\n"
            msg += f"标题: {title}\n"
            msg += f"日期: {dt.date()}\n"
            msg += f"情绪分数 (-10~+10): {sentiment:.2f}\n"
            msg += f"真实价格: {true_price_raw:,.2f} USD\n"
            msg += f"预测价格: {pred_raw:,.2f} USD\n"
            msg += f"本次 Loss: {loss_val:.6f}\n"
            msg += f"已学习新闻总数: {self.learn_count} 篇\n"
            msg += f"最近10次平均 Loss: {self.avg_loss:.6f}\n"
            msg += "—" * 50 + "\n"

            self.root.after(0, lambda: self.append_output(msg, "green" if sentiment > 0 else "red"))
            self.root.after(0, self.update_status)
            self.root.after(0, self.plot_emotion_price_comparison)

            torch.save(self.model.state_dict(), MODEL_PATH)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("学习出错", str(e)))
            self.root.after(0, lambda: self.append_output(f"处理出错: {e}\n", "red"))

        finally:
            self.is_training = False
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

    def predict(self):
        text, _, _ = self.validate_input()
        if not text:
            return

        try:
            sentiment = analyze_sentiment(text)

            self.model.eval()
            with torch.no_grad():
                pred_norm = self.model([text])
                pred_price = denormalize_price(pred_norm.item())

            msg = f"预测结果（无训练）:\n"
            msg += f"情绪分数 (-10~+10): {sentiment:.2f}\n"
            msg += f"预测 BTC 价格: {pred_price:,.2f} USD\n"
            msg += "—" * 50 + "\n"

            self.append_output(msg, "blue" if sentiment > 0 else "red")

        except Exception as e:
            self.append_output(f"预测出错: {e}\n", "red")

    def plot_emotion_price_comparison(self):
        if not self.news_records:
            return

        sorted_records = sorted(self.news_records.items(), key=lambda x: datetime.strptime(x[0][0], '%Y-%m-%d'))

        dates = []
        sentiments = []
        true_prices_norm = []

        for (date_str, title), (sentiment, true_price, _) in sorted_records:
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            sentiments.append(sentiment)
            norm_price = (true_price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) * 20 - 10
            true_prices_norm.append(norm_price)

        self.ax.clear()
        self.ax.plot(dates, sentiments, 'b-', label='新闻情绪分数 (-10~+10)')
        self.ax.plot(dates, true_prices_norm, 'orange', label='BTC真实价格 (归一化-10~+10)')
        self.ax.set_xlabel('日期')
        self.ax.set_ylabel('分数 / 归一化价格')
        self.ax.set_title('新闻情绪 vs BTC价格历史对比')
        self.ax.legend()
        self.ax.grid(True)
        self.figure.autofmt_xdate(rotation=45)
        self.ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        self.canvas.draw()

    def update_status(self):
        self.learn_count_label.config(text=f"已学习新闻: {self.learn_count} 篇")
        self.avg_loss_label.config(text=f"平均 Loss: {self.avg_loss:.6f}" if self.avg_loss > 0 else "平均 Loss: —")

        if self.learn_count > 0:
            recent_losses = ", ".join([f"{l:.4f}" for l in self.loss_history[-5:]])
            self.status_label.config(
                text=f"模型已训练 {self.learn_count} 次 | 最近 Loss: {recent_losses}",
                foreground="green"
            )
        else:
            self.status_label.config(text="尚未学习任何新闻", foreground="gray")


if __name__ == "__main__":
    root = tk.Tk()
    app = BTCNewsPriceGUI(root)
    root.mainloop()