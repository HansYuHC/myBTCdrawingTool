# news_to_price_manual_improved.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from datetime import datetime, timezone
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 减少 bert 警告

# ===================== 全局配置 =====================
MODEL_PATH = "news_price_model.pth"
KLINE_CSV = "btc_4h_2018.csv"  # 你的K线文件路径

# 价格归一化参数（根据2018-2019 BTC大致范围设置，可自行调整）
PRICE_MIN = 3000.0
PRICE_MAX = 20000.0
# 如果你想关闭归一化，把下面 USE_NORMALIZATION 设为 False
USE_NORMALIZATION = True


# 1. 模型定义（新闻 → 价格）
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
        # 强制非负：softplus 比 relu 平滑，更适合连续值
        price = torch.nn.functional.softplus(price)
        return price.squeeze(-1)


# 2. 加载 K线
def load_kline(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True)
    df = df.set_index('timestamps').sort_index()
    return df


# 3. 获取最近价格（已优化）
def get_price_at_time(df_kline, dt):
    if dt in df_kline.index:
        return df_kline.at[dt, 'close']

    pos = df_kline.index.get_indexer([dt], method='nearest')[0]
    if pos == -1:
        print("警告：时间超出K线范围，使用边界值")
        return df_kline['close'].iloc[0] if dt < df_kline.index[0] else df_kline['close'].iloc[-1]

    return df_kline['close'].iloc[pos]


# 4. 价格归一化 / 反归一化
def normalize_price(p):
    if not USE_NORMALIZATION:
        return p
    return (p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)


def denormalize_price(p_norm):
    if not USE_NORMALIZATION:
        return p_norm
    return p_norm * (PRICE_MAX - PRICE_MIN) + PRICE_MIN


# 5. 主程序
def main():
    print("欢迎使用新闻 → BTC 价格预测系统！（改进版）")
    print("1. 学习（喂新闻 + 日期）")
    print("2. 预测")
    choice = input("请选择 (1/2): ").strip()

    # 模型初始化
    model = NewsToPriceModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)  # 稍大一点，更快收敛
    criterion = nn.MSELoss()

    # 尝试加载已有模型
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print(f"\n成功加载已有模型: {MODEL_PATH}")
            print("继续之前的学习进度～\n")
        except Exception as e:
            print(f"加载模型失败: {e}\n使用全新模型...")
    else:
        print("\n未找到已有模型，使用全新初始化...\n")

    if choice == "1":
        learn_from_news(model, optimizer, criterion)
    elif choice == "2":
        predict_price(model)
    else:
        print("无效选择")

    # 最后再保存一次（以防万一）
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n模型已保存至: {MODEL_PATH}")


# 6. 学习模式
def learn_from_news(model, optimizer, criterion):
    df_kline = load_kline(KLINE_CSV)
    model.train()

    print("\n学习模式：日期只需输入到天（YYYY-MM-DD），时间自动设为 00:00:00")
    print("输入 'quit' 随时退出\n")

    while True:
        print("─" * 60)
        print("请输入下一条新闻（输入 'quit' 退出）：")

        title = input("标题: ").strip()
        if title.lower() == 'quit':
            break
        if not title:
            print("标题不能为空，请重新输入～\n")
            continue

        date_str = input("发布日期 (YYYY-MM-DD): ").strip()
        try:
            # 只取日期部分，时间强制设为 00:00:00 UTC
            dt_naive = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt_naive.replace(tzinfo=timezone.utc)
        except ValueError:
            print("日期格式错误！请使用 YYYY-MM-DD （例如 2019-01-11）\n")
            continue

        content = input("内容（可选，直接回车跳过）: ").strip()
        # 如果有内容，前面加空格避免标题和内容连在一起
        text = title + (" " + content if content else "")

        try:
            true_price_raw = get_price_at_time(df_kline, dt)
            true_price_norm = normalize_price(true_price_raw)
            true_price = torch.tensor([true_price_norm], dtype=torch.float32)

            pred_norm = model([text])
            loss = criterion(pred_norm, true_price)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_raw = denormalize_price(pred_norm.item())

            print("\n" + "═" * 50)
            print("本次学习完成！")
            print(f"新闻标题: {title}")
            print(f"日期: {date_str}")
            print(f"真实价格: {true_price_raw:,.2f} USD")
            print(f"当前预测: {pred_raw:,.2f} USD")
            print(f"Loss (归一化): {loss.item():.6f}")
            print("模型参数已更新并保存")
            print("═" * 50 + "\n")

            # 保存
            torch.save(model.state_dict(), MODEL_PATH)

        except Exception as e:
            print(f"处理出错: {e}")
            print("本次未保存，继续输入下一条吧～\n")
            continue


# 7. 预测模式
def predict_price(model):
    model.eval()
    print("\n预测模式：输入新闻即可得到价格预测，输入 'quit' 退出")

    while True:
        title = input("\n标题: ").strip()
        if title.lower() == 'quit':
            break
        if not title:
            print("标题不能为空")
            continue

        content = input("内容（可选，回车跳过）: ").strip() or ""
        text = title + " " + content

        try:
            with torch.no_grad():
                pred_norm = model([text])
                pred_price = denormalize_price(pred_norm.item())
            print(f"预测BTC价格: {pred_price:.2f} USD\n")
        except Exception as e:
            print(f"预测出错: {e}")


if __name__ == "__main__":
    main()