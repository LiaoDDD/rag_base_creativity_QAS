import os
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

# 1. 檢查 OPENAI_API_KEY -------------------------------------------------

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "請先在環境變數中設定 OPENAI_API_KEY，例如：\n"
        '  export OPENAI_API_KEY="sk-xxxxx"\n'
    )

# 2. 匯入 TruLens 的 OpenAI Provider --------------------------------------

from trulens.providers.openai import OpenAI as TruOpenAI

# 評估時用的模型（只當評分員，不是回答問題的 LLM）
# 你說要用 gpt-4.1，就直接指定這個：
provider = TruOpenAI(model_engine="gpt-4.1")


# 3. 小工具：統一從 TruLens 回傳結果中取出 score -------------------------

def extract_score(result):
    """
    從 TruLens OpenAI feedback 函式的回傳結果中抽出分數。

    可能的型別：
    - float / int
    - (score, meta) 形式的 tuple，例如 (1.0, {"reason": "..."} )
    - dict，裡面有 "score" 或 "result"
    - 帶有 .score 屬性的物件
    """

    if result is None:
        return None

    # 1) 直接是數字
    if isinstance(result, (int, float)):
        return float(result)

    # 2) 常見情況：with_cot_reasons 會回傳 (score, {"reason": ...})
    if isinstance(result, tuple) and len(result) >= 1:
        first = result[0]
        if isinstance(first, (int, float)):
            return float(first)

    # 3) dict 形式
    if isinstance(result, dict):
        if "score" in result:
            return float(result["score"])
        if "result" in result:
            return float(result["result"])

    # 4) 物件有 .score 屬性
    if hasattr(result, "score"):
        return float(result.score)

    # 5) 最後手段：嘗試直接轉 float
    try:
        return float(result)
    except Exception:
        raise TypeError(f"無法從結果中擷取分數: {result!r}")


# 4. 定義資料結構：一筆 QA 測試資料 --------------------------------------

@dataclass
class QACase:
    scenario_id: str         # 例如 "S1"
    trait: str               # "想像力" / "冒險性" / ...
    mode: str                # "Baseline" 或 "RAG"
    question: str            # 教師提問
    answer: str              # LLM 回答（Baseline 或 RAG）
    context_chunks: List[str]  # 當作 RAG 的知識脈絡，用來算 groundedness / context relevance


# 5. 建立四個情境的資料（目前用簡化版文字，可自行換成完整回答） --------

cases: List[QACase] = []

# ---------- 情境一：想像力 ----------

question_1 = (
    "學生小廖在作文課上展現出豐富的想像力，例如描述未來城市飄浮在空中、"
    "建築像變形金剛一樣可以移動和交談。這反映了他具備哪方面的創造力特質？"
)

baseline_answer_1 = (
    "答案：想像力（Imagination）是指個體能夠創造出新穎、獨特的思維或意念的能力。"
    "在小廖的作文中，他展現出了豐富的想像力，例如將城市描述為「飄浮在空中」，"
    "並且讓建築物具有移動和交談的能力。這些元素都體現了他對未來世界的創造性思考和構想。"
)

rag_answer_1 = (
    "結論：小廖在作文中展現出豐富的想像力，這反映了他具備「想像力」的創造力特質，信心高。\n"
    "理由與依據：根據威廉斯創造力傾向量表中的想像力構面，"
    "想像力包括能視覺化並建立心像、樂於幻想尚未發生的事情等。"
    "小廖將城市寫成飄浮在空中、建築物能像變形金剛一樣移動和交談，"
    "符合量表中「突破現實限制、描繪超越現實構想」的指標。"
)

context_1 = [
    "威廉斯創造力傾向量表：想像力（Imagination）指個體能在腦中形成生動心像，"
    "樂於幻想尚未發生的情境，並能補完故事或情節。",
    "範例題項：當我閱讀故事時，會在腦中看到畫面，並想像不同的結局。"
]

cases.append(
    QACase(
        scenario_id="S1",
        trait="想像力",
        mode="Baseline",
        question=question_1,
        answer=baseline_answer_1,
        context_chunks=context_1,
    )
)

cases.append(
    QACase(
        scenario_id="S1",
        trait="想像力",
        mode="RAG",
        question=question_1,
        answer=rag_answer_1,
        context_chunks=context_1,
    )
)

# ---------- 情境二：冒險性 ----------

question_2 = (
    "學生小明在科學實驗課上總是自告奮勇第一個上臺嘗試從未做過的實驗，"
    "儘管有失敗風險也毫不畏懼。這種態度反映了哪一種創造力傾向？"
)

baseline_answer_2 = "答案：冒險傾向。"

rag_answer_2 = (
    "結論：小明最可能展現出「冒險性」的創造力特質，信心高。\n"
    "理由與依據：威廉斯創造力傾向量表中冒險性題項指出，"
    "喜歡嘗試新的事情，即使不一定成功也無所謂，"
    "以及願意在不確定結果的情況下先行嘗試。"
    "小明主動嘗試未做過的實驗、不怕失敗，與此特質高度相符。"
)

context_2 = [
    "威廉斯創造力傾向量表：冒險性（Risk-taking）指願意嘗試新事物，"
    "即使可能失敗也不退縮，樂於面對不確定性與挑戰。",
    "範例題項：我喜歡嘗試新的事情，只是為了看看會發生什麼結果。"
]

cases.append(
    QACase(
        scenario_id="S2",
        trait="冒險性",
        mode="Baseline",
        question=question_2,
        answer=baseline_answer_2,
        context_chunks=context_2,
    )
)

cases.append(
    QACase(
        scenario_id="S2",
        trait="冒險性",
        mode="RAG",
        question=question_2,
        answer=rag_answer_2,
        context_chunks=context_2,
    )
)

# ---------- 情境三：好奇性 ----------

question_3 = (
    "學生小英經常在課後拉著老師問各種『為什麼』，"
    "例如「為什麼天會下雨？」「為什麼植物會向陽生長？」等。"
    "這代表她具備哪種創造力人格特質？"
)

baseline_answer_3 = (
    "答案：好奇心（Curiosity）是創造力的重要基石。"
    "小英對周圍事物的好奇心和求知欲，體現了她對學習和探索的熱情與興趣。"
)

rag_answer_3 = (
    "結論：小英最可能具備的是好奇性（Curiosity），信心高。\n"
    "理由與依據：威廉斯創造力傾向量表中，好奇性包含富有追根究柢的精神，"
    "經常主動問「為什麼」，並願意深入思索事物的奧妙。"
    "小英課後頻繁追問各種『為什麼』，正是這項特質的典型表現。"
)

context_3 = [
    "威廉斯創造力傾向量表：好奇性（Curiosity）指經常主動發問，"
    "願意追問事情的原因與原理，對未知現象保持高度興趣。",
    "範例題項：我常常問「為什麼」，直到我弄懂為止。"
]

cases.append(
    QACase(
        scenario_id="S3",
        trait="好奇性",
        mode="Baseline",
        question=question_3,
        answer=baseline_answer_3,
        context_chunks=context_3,
    )
)

cases.append(
    QACase(
        scenario_id="S3",
        trait="好奇性",
        mode="RAG",
        question=question_3,
        answer=rag_answer_3,
        context_chunks=context_3,
    )
)

# ---------- 情境四：挑戰性 ----------

question_4 = (
    "學生小安在解數學題時，完成基本要求後會主動詢問老師有無更困難的題目可以嘗試，"
    "把困難當作樂趣，不怕挫折。這種態度反映了哪種創造力特質較為突出？"
)

baseline_answer_4 = (
    "答案：好奇心、冒險精神和堅持不懈的態度。"
    "小安對困難的接受和追求，體現了他對學習的熱情和挑戰自我的意願。"
)

rag_answer_4 = (
    "結論：小安的創造力特質中，挑戰性較為突出，信心高。\n"
    "理由與依據：在威廉斯創造力傾向量表中，挑戰性包含喜歡面對困難、"
    "願意尋找更具挑戰性的任務，並在沒有標準答案時仍持續嘗試。"
    "小安主動要求更難的題目，把困難當樂趣，與此特質高度吻合。"
)

context_4 = [
    "威廉斯創造力傾向量表：挑戰性（Challenge）指樂於面對高難度問題，"
    "不畏挫折，願意不斷嘗試不同的方法解決問題。",
    "範例題項：我喜歡解決困難的問題，即使沒有標準答案也沒關係。"
]

cases.append(
    QACase(
        scenario_id="S4",
        trait="挑戰性",
        mode="Baseline",
        question=question_4,
        answer=baseline_answer_4,
        context_chunks=context_4,
    )
)

cases.append(
    QACase(
        scenario_id="S4",
        trait="挑戰性",
        mode="RAG",
        question=question_4,
        answer=rag_answer_4,
        context_chunks=context_4,
    )
)


# 6. 定義三個評估函式（直接呼叫 provider） -------------------------------
#    ❗這裡是這次「修正的關鍵」：全部改成用「位置參數」呼叫，
#      避免不同版本的 keyword 名稱不一致導致錯誤。


def eval_answer_relevance(question: str, answer: str) -> float:
    """
    Answer Relevance：問題與回答的相關性。
    使用 TruLens OpenAI Provider 的 relevance_with_cot_reasons。
    """
    # 使用位置參數 (question, answer)
    result = provider.relevance_with_cot_reasons(question, answer)
    return extract_score(result)


def eval_context_relevance(question: str, context_chunks: List[str]) -> float:
    """
    Context Relevance：問題與「每一個 context chunk」的相關性平均。
    """
    if not context_chunks:
        return None

    scores = []
    for ctx in context_chunks:
        # 使用位置參數 (question, context)
        r = provider.context_relevance_with_cot_reasons(question, ctx)
        scores.append(extract_score(r))

    return sum(scores) / len(scores)


def eval_groundedness(answer: str, context_chunks: List[str]) -> float:
    """
    Groundedness：回答在多大程度上被 context 支持。
    使用 groundedness_measure_with_cot_reasons。
    注意：TruLens LLMProvider 的預期簽名是 (source, statement)，
    所以這裡用位置參數傳入 (context_chunks, answer)，
    不再使用 context= 這種 keyword argument。
    """
    if not context_chunks:
        return None

    # ✅ 關鍵修正：不要寫 context= / statement=，改用位置參數
    result = provider.groundedness_measure_with_cot_reasons(
        context_chunks,  # source
        answer           # statement
    )
    return extract_score(result)


# 7. 對所有情境與兩種模式進行評估 ----------------------------------------

rows: List[Dict[str, Any]] = []

print("開始評估四個情境（Baseline vs RAG）...\n")

for case in cases:
    print(f"評估 {case.scenario_id} - {case.trait} - {case.mode} ...")

    ans_rel = eval_answer_relevance(case.question, case.answer)
    ctx_rel = eval_context_relevance(case.question, case.context_chunks)
    grounded = eval_groundedness(case.answer, case.context_chunks)

    rows.append(
        {
            "scenario_id": case.scenario_id,
            "trait": case.trait,
            "mode": case.mode,
            "answer_relevance": ans_rel,
            "context_relevance": ctx_rel,
            "groundedness": grounded,
        }
    )

print("\n=== 評估完成，整理結果中 ===\n")

df = pd.DataFrame(rows)

# 8. 輸出詳細表格與平均分數 ----------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("============== 逐題評估結果（可直接貼到論文附錄） ==============\n")
print(df.to_string(index=False))

print("\n============== 依模式統計的平均分數（可用於主文比較） ==============\n")
mean_by_mode = df.groupby("mode")[["answer_relevance", "context_relevance", "groundedness"]].mean()
print(mean_by_mode.to_string())

# 匯出 CSV，方便之後用 Excel / SPSS / R 再分析
output_csv = "creativity_trulens_scores.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n已將逐題結果輸出為 CSV 檔：{output_csv}")
print("你可以在論文中說明：三項指標皆由 TruLens OpenAI Provider (gpt-4.1) 自動評估所得。")





