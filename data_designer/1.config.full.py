###
# 1

from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    InferenceParameters,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    ModelConfig,
    NeMoDataDesignerClient,
    SamplerColumnConfig,
    SamplerType,
    Score,
)

from pydantic import BaseModel, Field
from datasets import load_dataset
import pandas as pd
import random


###
# 2

NEMO_MICROSERVICES_BASE_URL = "http://10.19.60.80:8080"

data_designer_client = NeMoDataDesignerClient(base_url=NEMO_MICROSERVICES_BASE_URL)

print(data_designer_client)

###
# 3

MODEL_PROVIDER = "nvidiabuild"
MODEL_ID = "openai/gpt-oss-120b"
MODEL_ALIAS = "gpt-oss-120b"
SYSTEM_PROMPT = ""
JUDGE_MODEL_ALIAS = "quality-judge"

model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.9,
            top_p=0.95,
            max_tokens=2048,
            max_parallel_requests=8,
            timeout=1200
        ),
    ),
    ModelConfig(
        alias=JUDGE_MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            max_parallel_requests=4,
            timeout=1500,
        ),
    ),
]

###
# 4

import ipdb; ipdb.set_trace()

import json
import pandas as pd

personas_dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train")

# DataFrameに変換
df = personas_dataset.to_pandas()
print(df)
# for DEBUG NOTE TODO
import ipdb; ipdb.set_trace()

### df = df.head(10000)

'''
ipdb> personas_dataset
Dataset({
    features: ['uuid', 'professional_persona', 'sports_persona', 'arts_persona', 'travel_persona', 'culinary_persona', 'persona', 'cultural_background', 'skills_and_expertise', 'skills_and_expertise_list', 'hobbies_and_interests', 'hobbies_and_interests_list', 'career_goals_and_ambitions', 'sex', 'age', 'marital_status', 'education_level', 'occupation', 'region', 'area', 'prefecture', 'country'],
    num_rows: 1000000
}) # 1 million rows
'''
'''
0  uuid 09ae83ddc11745f6b995113fe7ed4a77
professional_persona 野本 花代子は、介護サービスの品質向上を推進する事業リーダーとして、構造化されたプロセス設計とリスクマネジメントの徹底に重点を置き、予測可能な成果を保証しつつ、現場のストレス要因を緩和するための継続的なヒューマン・ファクター分析を実施している。
sports_persona 野本 花代子は、シニア向け健康体操と季節花観賞を組み合わせたウォーキング・エクササイズを主催し、地域コミュニティの外向性を刺激しつつ、時間帯や天候に合わせたプログラム設計で、身体的活力と社会的交流を両立させている。
arts_persona 野本 花代子は、季節の花を用いた伝統的な和句と、爽やかなシニア向け音楽鑑賞会を融合させ、地域の美学と自律的な創作活動を促す文化的実践として、感情表出と精神的安定を支える芸術的表現ギャラリーと交流の場を創造している。
travel_persona 野本 花代子は、関東の歴史的散策路と郷土料理の食文化体験をテーマに、シニア仲間との月例旅行プランを企画し、事前に看護リスク評価と安全マップ作成を行うことで、探求心と安心感を同時に満たす旅行ルートを提供している。
culinary_persona 野本 花代子は、季節の保存食研究と節約料理の実践を通じて、関東郷土料理「すきやき」や「蓬蒸し」を低コストで再現し、具材の栄養価と保存期間をExcelで管理しながら、シニアの食事安全性と味覚享受のバランスを追求している。
persona 野本 花代子は、構造的予測力と節約志向を持つシニア介護リーダーで、科学的評価と地域文化実践を統合し、共感的安全感を高める行動を取る。
cultural_background 高度成長世代として、戦後の急速な経済拡大を体感し、節約志向と地域への帰属意識を大切にしている。東京都内の関東地域に根ざした生活経験が、介護福祉の現場での実践的な価値観に影響を与えている。
skills_and_expertise 介護福祉の現場でのケアプラン作成と利用者の身体機能維持に関する専門知識、大学理系卒業に基づく科学的思考と統計的分析力、Excelやメールを活用した事務管理とデータ集計、実務で培ったリスクマネジメントと予防介護のアプローチ。
skills_and_expertise_list ['ケアプラン作成','身体機能評価','Excelデータ集計','メール事務管理','リスクマネジメント','予防介護実践']
hobbies_and_interests 地域の歴史散策と季節の花の観賞、節約料理と保存食の研究、シニア向けの健康体操や爽やかな音楽鑑賞、関東の郷土料理と食文化の学習に興味を持ち、仲間と情報共有する活動に積極的に参加している。
hobbies_and_interests_list ['歴史散策','季節の花観賞','節約料理','保存食研究','シニア健康体操','音楽鑑賞','郷土料理学習']
career_goals_and_ambitions 地域の高齢者の生活の質向上を支える介護サービスの品質向上を推進し、後進の介護福祉士への指導と現場改善のリーダーシップを担うことで、関東地域における持続可能な介護体制構築に寄与したい。
sex 女
age 72
marital_status 離別 (子供あり)
education_level 大学卒 理系
occupation 介護福祉業 中堅
region 関東地方
area 東日本
prefecture 東京都
21  country 日本
'''

import ipdb; ipdb.set_trace()

'''
种子目标：共 2000 个种子

弱项 A：地理（250 个），工具（100 个），公共（200 个），其他（150 个）= 共 400 个

弱项 B（弱项强化）：金融（400 个），安全（350 个），词汇（350 个）= 共 1100 个

典型值：剩余 500 个种子

偏差抑制：每个职业最多 10 个，每个都道府县最多 12 个

SEED_TARGET: 2000 total seeds
WeakA: geo(250), tools(100), public(200), other(150) = 400 total
WeakB (weakness reinforcement): finance(400), safety(350), vocab(350) = 1100 total
Typical: Remaining 500 seeds
Bias suppression: Max 10 per occupation, max 12 per prefecture
'''

###
# 5

import re
import numpy as np
import pandas as pd
from typing import List, Optional

np.random.seed(42)


SEED_TARGET = 2000

# weakB（弱点補強）: 合計1100
WEAKB_TARGETS = {"finance": 400, "safety": 350, "vocab": 350}
N_WEAK_A = 400        # weakA全体
N_TYPICAL = SEED_TARGET - sum(WEAKB_TARGETS.values()) - N_WEAK_A  # = 500

# weakA 内をサブクォータで固定（ここが最重要）
N_GEO = 250
N_TOOLS = 100
N_PUBLIC = 200
N_WEAK_A_OTHER = N_WEAK_A - N_GEO - N_TOOLS - N_PUBLIC  # 150

# 偏り抑制（cap）
CAP_PER_OCCUPATION = 10
CAP_PER_PREFECTURE = 12

# D(公共)を“宗教/儀礼”に吸わせないための優先度（public_bonus）
PUBLIC_BONUS_KW = ["図書館","病院","役所","窓口","待合室","列","並ぶ","順番","受付","会計",
                   "駅","改札","切符","ゴミ","分別","ルール","禁止","優先席","エスカレーター","エレベーター"]
RELIGION_PENALTY_KW = ["寺","神社","教会","礼拝堂","ミサ","修道院","仏壇","お経","線香","数珠","墓","お守り"]

# 再現性: hash(cat)禁止 → 固定seed
CAT_RS = {"finance": 201, "safety": 202, "vocab": 203}
WA_RS  = {"geo": 111, "tools": 112, "other": 113}
TY_RS  = 301

# neutral を増やしすぎない（上限）
NEUTRAL_CAP = 50


###
# 6
import ipdb; ipdb.set_trace()
TEXT_COLS_ALL = [
    "occupation","hobbies_and_interests","marital_status","education_level","prefecture","region","area",
    "professional_persona","cultural_background","travel_persona","culinary_persona","persona",
]
CORE_COLS = ["occupation","hobbies_and_interests","marital_status","education_level","prefecture","region","area"]
EXTRA_COLS = ["uuid","age","age_band","skills_and_expertise_list",
              "travel_persona","hobbies_and_interests","area"]

for col in sorted(set(TEXT_COLS_ALL + CORE_COLS + EXTRA_COLS)): # 16 columns
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)
    else:
        df[col] = ""

def build_text(row: pd.Series, cols: List[str]) -> str:
    parts = [row.get(c, "") for c in cols]
    parts = [p for p in parts if str(p).strip()]
    return " / ".join(parts)

df["_all_text"]  = df.apply(lambda r: build_text(r, TEXT_COLS_ALL), axis=1) # 所有列，拼成一列 NOTE 这个非常耗费时间
df["_core_text"] = df.apply(lambda r: build_text(r, CORE_COLS), axis=1) # 只是重要的列，拼成一列
df["_core_len"]  = df["_core_text"].str.len() # 重要的core text的拼接之后的长度


###
# 7
import ipdb; ipdb.set_trace()
KEY_COLS = ["prefecture","region","area","occupation","education_level","marital_status"]

def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.strip())

df["_attr_key"] = df.apply(lambda r: "|".join([_norm(r.get(c, "")) for c in KEY_COLS]), axis=1)
df = df[df["_attr_key"] != "|||||"].copy() # 删除“完全空”的 key
# use this "key" to judge: 两行数据是不是“同一类人”

###
# 8
import ipdb; ipdb.set_trace()
NEG_KW_JC = [
    "陰謀","反ワク","極右","極左","ヘイト","差別","排外","テロ","過激",
    "万引き","窃盗","詐欺","横領","覚醒剤","麻薬",
    "絶対","許せない","嫌悪","憎い",
]
df["has_neg_jc"] = df["_core_text"].str.contains("|".join(map(re.escape, NEG_KW_JC)), regex=True, na=False)
df_jc = df[~df["has_neg_jc"]].copy()
# [1000000 rows x 27 columns] --> [999984 rows x 28 columns]

###
# 9
import ipdb; ipdb.set_trace()

geo_kw = ["電車","地下鉄","新幹線","駅","改札","切符","定期券","ICカード","Suica","PASMO","バス","バス停","タクシー",
          "高速道路","駐車場","乗り換え","時刻表","終電","始発","渋滞","踏切","信号","ホーム","路線","乗車券",
          "徒歩","歩く","歩道","横断歩道","交差点","信号待ち","右折","左折","移動","目的地","経路","ルート",
          "地図","ナビ","最寄り","乗る","降りる","下車","乗車","入口","出口","改札口","ホームドア","階段","乗り場"]

life_kw = ["料理","掃除","片付け","DIY","買い物","健康","睡眠","育児","子育て","弁当","洗濯","家事",
           "ゴミ出し","分別","資源ごみ","可燃","不燃","粗大ごみ","節約","整理整頓","収納",
           "病院","薬局","会計","レジ",
           "戸棚","食器棚","本棚","引き出し","クローゼット","タンス","棚","机","椅子",
           "掃除機","洗濯機","電子レンジ","炊飯器"]

tools_kw = ["包丁","まな板","鍋","フライパン","菜箸","おたま","ほうき","ちりとり","雑巾","スポンジ","洗剤","漂白剤",
            "掃除機","ドライバー","ペンチ","金づち","かなづち","ノコギリ","シャベル","釘","ハサミ",
            "ペン","鉛筆","シャーペン","消しゴム","ノート","付箋","ホッチキス","クリップ","定規","マーカー","カッター",
            "ガムテープ","セロテープ","ラップ","アルミホイル","トング","計量スプーン","計量カップ",
            "延長コード","充電器","電池","ドライヤー","アイロン"]

public_kw = ["公共","ルール","順番","列","並ぶ","割り込み","優先席","禁煙","喫煙","騒音","静かに","ゴミ","ポイ捨て","迷惑",
             "図書館","映画館","病院","待合室","エスカレーター","エレベーター","コンビニ","スーパー","店内"]

culture_kw = ["礼儀","作法","冠婚葬祭","お辞儀","正月","お盆","祭り","年中行事","着物","茶道",
              "挨拶","敬語","名刺","手土産","お礼","お詫び","謝罪","断る","遠慮","失礼","配慮","気遣い",
              "神社","寺","仏教","神道"]

finance_kw = ["支払い","会計","料金","値段","割引","クーポン","精算","返金","領収書","レシート","釣銭","おつり",
              "銀行","ATM","口座","振込","送金","引き落とし","請求","請求書","利用明細","明細","手数料","ポイント",
              "クレジットカード","デビット","電子マネー","キャッシュレス","税","納付","控除","保険","ローン","分割払い","家計","予算"]

safety_kw = ["危険","危ない","事故","転倒","火","火事","火傷","やけど","ガス","ガス漏れ","一酸化炭素","換気",
             "感電","漏電","ガソリン","発火","刃物","ナイフ","包丁",
             "飲酒運転","信号無視","スピード","雪道","凍結","ノーマルタイヤ","ヘルメット","シートベルト",
             "漂白剤","殺虫剤","洗剤","薬","誤飲","食中毒","熱中症","凍傷",
             "避難","非常口","消火器","火災報知器","警報","防災"]

vocab_kw = ["用途","目的","分類","種類","意味","定義","言い換え","言い回し","表現",
            "類義語","対義語","反対語","ことわざ","慣用句","比喩","例え",
            "適切","不適切","一般的","代表的","日常的"]

JC_CAT_KWS = {
    "finance": finance_kw,
    "safety":  safety_kw,
    "vocab":   vocab_kw,
    "public":  public_kw,
    "tools":   tools_kw,
    "life":    life_kw,
    "geo":     geo_kw,
    "culture": culture_kw,
}

def count_hits(text: str, kws: List[str]) -> int:
    if not text:
        return 0
    return sum(1 for k in kws if k in text)

ALL_JC_KW = [kw for kws in JC_CAT_KWS.values() for kw in kws] # put all keywords together! 279 keys

###
# 10

import ipdb; ipdb.set_trace()
# 基本は core_text
for cat, kws in JC_CAT_KWS.items():
    df_jc[f"score_{cat}"] = df_jc["_core_text"].map(lambda t, _kws=kws: count_hits(t, _kws))

# geo/tools のみ補助テキストで再計算（限定列のみ足す）
df_jc["_geo_text"] = df_jc["_core_text"] + " / " + df_jc["travel_persona"] + " / " + df_jc["area"]
df_jc["_tools_text"] = df_jc["_core_text"] + " / " + df_jc["hobbies_and_interests"] + " / " + df_jc["skills_and_expertise_list"]

df_jc["score_geo"] = df_jc["_geo_text"].map(lambda t: count_hits(t, geo_kw))
df_jc["score_tools"] = df_jc["_tools_text"].map(lambda t: count_hits(t, tools_kw))

df_jc["_kw_hits"] = df_jc["_core_text"].map(lambda t: count_hits(t, ALL_JC_KW))

MAX_SCORE = {"finance":14,"safety":16,"vocab":12,"public":10,"tools":10,"life":10,"geo":10,"culture":10}
mask_ok = pd.Series(True, index=df_jc.index)
for c, mx in MAX_SCORE.items():
    mask_ok &= (df_jc[f"score_{c}"] <= mx)
df_jc = df_jc[mask_ok].copy()

def assign_jc_category(row: pd.Series) -> Optional[str]:
    scores = {c: row[f"score_{c}"] for c in JC_CAT_KWS}
    maxv = max(scores.values()) if scores else 0
    if maxv <= 0:
        return None
    tied = [c for c, v in scores.items() if v == maxv]
    priority = ["finance","safety","vocab","public","tools","life","geo","culture"]
    for p in priority:
        if p in tied:
            return p
    return tied[0]

df_jc["jc_category"] = df_jc.apply(assign_jc_category, axis=1)
# 用关键词规则给 persona 自动打标签 + 清洗异常数据



###
# 11

import ipdb; ipdb.set_trace()
DEFINITION_KW = ["とは","という","意味","定義","何という","何と言う","どれを指す"]
df_jc["_has_definition"] = df_jc["_core_text"].str.contains("|".join(map(re.escape, DEFINITION_KW)), regex=True, na=False)

NEUTRAL_MAX_LEN_CORE = 260
NEUTRAL_MAX_HITS = 0
df_jc["is_neutral"] = (
    (df_jc["_core_len"] > 0)
    & (df_jc["_core_len"] <= NEUTRAL_MAX_LEN_CORE)
    & (df_jc["_kw_hits"] <= NEUTRAL_MAX_HITS)
    & (~df_jc["_has_definition"]) # 不是“定义句”，更像自然语言
)

neutral_pool = df_jc[df_jc["is_neutral"]].drop_duplicates(subset=["_attr_key"]).copy()
neutral_pool["jc_category"] = "neutral"
neutral_pool = neutral_pool.sort_values("_core_len").head(NEUTRAL_CAP)  # ★上限

jc_pool = df_jc[df_jc["jc_category"].notna()].drop_duplicates(subset=["_attr_key"]).copy()

score_cols = [f"score_{c}" for c in JC_CAT_KWS]
jc_pool["max_score_any"] = jc_pool[score_cols].max(axis=1) # 这个样本“最强属于哪个类别”

typical_pool = (
    pd.concat([neutral_pool, jc_pool[jc_pool["max_score_any"] <= 2]], axis=0)
    .drop_duplicates(subset=["_attr_key"])
    .copy()
)

weakB_pool = jc_pool[jc_pool["jc_category"].isin(["finance","safety","vocab"])].copy() # “关键类别”（重点训练）
# weakAはここで分割して使う（geo/tools/other）
geo_pool   = jc_pool[jc_pool["jc_category"]=="geo"].copy()
tools_pool = jc_pool[jc_pool["jc_category"]=="tools"].copy()
public_pool = jc_pool[jc_pool["jc_category"]=="public"].copy()  # ★追加
other_pool  = jc_pool[jc_pool["jc_category"].isin(["culture","life"])].copy()  # ★publicを除外


###
# 12
import ipdb; ipdb.set_trace()
def sample_with_caps(pool: pd.DataFrame, n: int, random_state: int = 0) -> pd.DataFrame:
    if n <= 0 or len(pool) == 0:
        return pool.iloc[0:0].copy()

    pool2 = pool.sample(frac=1.0, random_state=random_state).copy()

    out = []
    occ_cnt, pref_cnt = {}, {}

    # 1st pass: capあり
    for _, r in pool2.iterrows():
        occ = str(r.get("occupation",""))
        pref = str(r.get("prefecture",""))
        if occ and occ_cnt.get(occ, 0) >= CAP_PER_OCCUPATION:
            continue
        if pref and pref_cnt.get(pref, 0) >= CAP_PER_PREFECTURE:
            continue
        out.append(r)
        occ_cnt[occ] = occ_cnt.get(occ, 0) + 1
        pref_cnt[pref] = pref_cnt.get(pref, 0) + 1
        if len(out) >= n:
            return pd.DataFrame(out)

    # 2nd pass: cap緩和（残りを素直に埋める）
    used_keys = set([r["_attr_key"] for r in out if "_attr_key" in r])
    for _, r in pool2.iterrows():
        k = r.get("_attr_key", None)
        if k is not None and k in used_keys:
            continue
        out.append(r)
        if len(out) >= n:
            break

    return pd.DataFrame(out)


###
# 13
import ipdb; ipdb.set_trace()

seed_parts = []
used = set()

# ---- weakB（固定内訳）----
for cat, n in WEAKB_TARGETS.items():
    g = weakB_pool[(weakB_pool["jc_category"] == cat) & (~weakB_pool["_attr_key"].isin(used))].copy()
    s = sample_with_caps(g, min(n, len(g)), random_state=CAT_RS[cat])
    seed_parts.append(s)
    used |= set(s["_attr_key"].tolist())

# ---- weakA（サブクォータ）----
g = geo_pool[~geo_pool["_attr_key"].isin(used)].copy()
s = sample_with_caps(g, min(N_GEO, len(g)), random_state=WA_RS["geo"])
seed_parts.append(s)
used |= set(s["_attr_key"].tolist())

g = tools_pool[~tools_pool["_attr_key"].isin(used)].copy()
s = sample_with_caps(g, min(N_TOOLS, len(g)), random_state=WA_RS["tools"])
seed_parts.append(s)
used |= set(s["_attr_key"].tolist())

g = public_pool[~public_pool["_attr_key"].isin(used)].copy()
s = sample_with_caps(g, min(N_PUBLIC, len(g)), random_state=WA_RS.get("public", 114))
seed_parts.append(s)
used |= set(s["_attr_key"].tolist())

g = other_pool[~other_pool["_attr_key"].isin(used)].copy()
# public優先・宗教抑制を効かせる
def _count_kw(text: str, kws) -> int:
    if not text:
        return 0
    return sum(1 for k in kws if k in text)

g["_public_bonus"] = g["_core_text"].map(lambda t: _count_kw(t, PUBLIC_BONUS_KW))
g["_religion_pen"] = g["_core_text"].map(lambda t: _count_kw(t, RELIGION_PENALTY_KW))
g = g.sort_values(["_public_bonus","_religion_pen","_core_len"], ascending=[False, True, True]).copy()
s = sample_with_caps(g, min(N_WEAK_A_OTHER, len(g)), random_state=WA_RS["other"])
seed_parts.append(s)
used |= set(s["_attr_key"].tolist())

# ---- typical（残り）----
remain = SEED_TARGET - sum(len(x) for x in seed_parts)
g = typical_pool[~typical_pool["_attr_key"].isin(used)].sort_values("_core_len", ascending=True).copy()
s = sample_with_caps(g, min(remain, len(g)), random_state=TY_RS)
seed_parts.append(s)

seed_jc = pd.concat(seed_parts, axis=0).drop_duplicates(subset=["_attr_key"]).copy()

# サイズを必ず合わせる（最後に埋め戻し）
remain = SEED_TARGET - len(seed_jc)
if remain > 0:
    pool = df_jc[~df_jc["_attr_key"].isin(set(seed_jc["_attr_key"]))].drop_duplicates(subset=["_attr_key"]).copy()
    if len(pool) == 0:
        raise ValueError("fill pool empty; cannot reach SEED_TARGET")
    seed_jc = pd.concat([seed_jc, pool.head(remain)], axis=0).drop_duplicates(subset=["_attr_key"]).copy()

# 最終調整
if len(seed_jc) > SEED_TARGET:
    seed_jc = seed_jc.sample(n=SEED_TARGET, random_state=123)
elif len(seed_jc) < SEED_TARGET:
    raise ValueError(f"seed pool too small: seed_jc={len(seed_jc)} < {SEED_TARGET}")


###
# 14
import ipdb; ipdb.set_trace()
# ============================================================
# 9) jc_theme（categoryから決め打ち）
# ============================================================
CAT2THEME = {
    "finance": "C_支払い・お金",
    "safety":  "E_安全・危険",
    "vocab":   "B_道具・用途",
    "tools":   "B_道具・用途",
    "public":  "D_公共施設・マナー手順",
    "culture": "D_公共施設・マナー手順",
    "geo":     "A_交通・移動",
    "life":    "F_生活・家事",
    "neutral": "N_ニュートラル",
}
seed_jc["jc_theme"] = seed_jc["jc_category"].map(CAT2THEME).fillna("N_ニュートラル")

print("[seed_jc] size:", len(seed_jc))
print(seed_jc["jc_category"].value_counts(dropna=False))
print(seed_jc["jc_theme"].value_counts(dropna=False))

### 
# 15
# ============================================================
# 10) seed_out（prompt整合）
# ============================================================
import ipdb; ipdb.set_trace()
required_cols = ["occupation","prefecture","region","marital_status","age_band","jc_theme","skills_and_expertise_list"]
for c in required_cols:
    if c not in seed_jc.columns:
        seed_jc[c] = ""

# age/skills を使わない方針なら、ここで空にする（推奨）
# seed_jc["age_band"] = ""
# seed_jc["skills_and_expertise_list"] = ""

seed_jc["skills_and_expertise_list"] = seed_jc["skills_and_expertise_list"].fillna("").astype(str)

seed_out_cols = [
    "uuid","occupation","prefecture","region","marital_status",
    "age_band","skills_and_expertise_list","jc_theme","jc_category","_attr_key"
]
seed_out_cols = [c for c in seed_out_cols if c in seed_jc.columns]
seed_out = seed_jc[seed_out_cols].copy()

seed_out.head(10)


###
# 16
import hashlib

import ipdb; ipdb.set_trace()

THEME_TO_TOPIC_WEIGHTS = {
    "A_交通・移動": {"交通": 0.60, "公共の場": 0.25, "日常生活": 0.15},
    "B_道具・用途": {"日常生活": 0.35, "買い物": 0.20, "学校": 0.20, "食事": 0.15, "職場": 0.10},
    "C_支払い・お金": {"買い物": 0.40, "交通": 0.25, "公共の場": 0.20, "日常生活": 0.15},
    "D_公共施設・マナー手順": {"公共の場": 0.45, "交通": 0.20, "学校": 0.20, "日常生活": 0.15},
    "E_安全・危険": {"交通": 0.30, "公共の場": 0.30, "日常生活": 0.25, "学校": 0.15},
    "F_生活・家事": {"日常生活": 0.70, "食事": 0.30},
    "N_ニュートラル": {"日常生活": 1.00},
}

def stable_u01(s: str) -> float:
    """uuid等から0-1の安定乱数を作る（毎回同じ）"""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return x / 0xFFFFFFFF

def pick_weighted(weights: dict, u: float) -> str:
    acc = 0.0
    last = None
    for k, w in weights.items():
        acc += float(w)
        last = k
        if u <= acc + 1e-12:
            return k
    return last

key_col = "uuid" if ("uuid" in df.columns and df["uuid"].astype(str).str.strip().ne("").any()) else "_attr_key"
if key_col not in df.columns:
    raise ValueError("seedに uuid か _attr_key が必要です")

def assign_topic(row):
    theme = str(row.get("jc_theme", "")).strip()
    weights = THEME_TO_TOPIC_WEIGHTS.get(theme)
    if not weights:
        return "日常生活"  # 想定外は安全側
    u = stable_u01(str(row[key_col]))
    return pick_weighted(weights, u)

seed_jc["topic_category"] = seed_jc.apply(assign_topic, axis=1)


###
# 17
import ipdb; ipdb.set_trace()

import re
# ============ 3) テキスト正規化（未終了クォートの温床つぶし） ============
# 改行の正規化：CRLF/CR → LF
def normalize_newlines(s: str) -> str:
    return str(s).replace('\r\n', '\n').replace('\r', '\n')

# 制御文字の除去（タブと改行は保持）
_ctrl_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
def strip_ctrl(s: str) -> str:
    return _ctrl_re.sub('', s)

# 孤立サロゲートや非BMPは基本OKだが、壊れた文字（decode失敗由来）があれば落とす
def safe_unicode(s: str) -> str:
    # そのまま返す（必要なら独自の置換を追加）
    return s

# 内部の ASCII ダブルクォートは csv では "" に二重化されるが、
# 念のためここでも安全側に置換（過剰二重化はしないように一旦 raw を記録）
def escape_quotes_for_csv(s: str) -> str:
    # ここでは何もしなくても csv.writer が二重化するが、
    # 途中で他ツール経由する可能性があるなら下の一行を有効化
    # s = s.replace('"', '""')
    return s

def clean_cell(s: str) -> str:
    s = normalize_newlines(s)
    s = strip_ctrl(s)
    s = safe_unicode(s)
    s = escape_quotes_for_csv(s)
    return s

for c in df.columns:
    df[c] = df[c].map(clean_cell)

# ============ 4) 未終了クォート“になり得る”怪しい行の検出 ============
# ポイント：書き出し前に「各行をカンマで連結したと仮定した文字列」の
# ダブルクォート出現数（"）が“奇数”なら危険（完全検出ではないが安価な一次スクリーニング）
def is_potential_unterminated_quote(row_values):
    # ここでは「生CSV文字列」ではなく“連結仮想文字列”で簡易チェック
    s = ",".join(str(v) if v is not None else "" for v in row_values)
    # すでに "" に二重化していない生の " が含まれている場合に引っ掛けたい
    # ただし上の escape_quotes_for_csv を無効化しているのでここは緩めに総数チェック
    return s.count('"') % 2 == 1  # 奇数なら未終了の可能性

###
#18
import ipdb; ipdb.set_trace()

suspect_mask = seed_jc.apply(lambda r: is_potential_unterminated_quote(r.tolist()), axis=1)
suspects = seed_jc[~suspect_mask].reset_index(drop=True)
suspects = suspects.drop(
     columns=["is_neutral", "_has_definition",
             "has_neg_jc"],
)
suspects.to_csv("filtered_personas_balanced_clean_2000_jcommonsense.csv", index=False)


###
# 19
# jcommonsenseqa用の構造
import ipdb; ipdb.set_trace()
class JCommonsenseQAData(BaseModel):
    question: str = Field(
        ...,
        description="日常生活や常識に関する日本語の質問。文脈や状況を明確に含める。"
    )
    choice0: str = Field(..., description="選択肢0: 自然で妥当な選択肢")
    choice1: str = Field(..., description="選択肢1: 自然で妥当な選択肢")
    choice2: str = Field(..., description="選択肢2: 自然で妥当な選択肢")
    choice3: str = Field(..., description="選択肢3: 自然で妥当な選択肢")
    choice4: str = Field(..., description="選択肢4: 自然で妥当な選択肢")
    answer_index: int = Field(
        ...,
        ge=0,
        le=4,
        description="正解の選択肢のインデックス(0-4)"
    )
    reasoning: str = Field(
        ...,
        description="なぜその選択肢が正解なのかの説明"
    )

###
# 20
import ipdb; ipdb.set_trace()
# Seedデータ使用版の設定ビルダー
# 重要: seed_datasetパラメータでDataFrameを渡す
config_builder_with_seed_jcommonsenseqa = DataDesignerConfigBuilder(
    model_configs=model_configs,
)


###
# 21
import ipdb; ipdb.set_trace()

seed_dataset_reference_jcommonsense = data_designer_client.upload_seed_dataset(
    dataset="filtered_personas_balanced_clean_2000_jcommonsense.csv",
    repo_id="data-designer/filtered_personas_clean_balanced_2000_jcommonsense_add_jc_theme",
    datastore_settings={"endpoint": "http://10.19.60.80:3000/v1/hf"},
)

###
# 22
import ipdb; ipdb.set_trace()

# Pass the reference to the config builder for use during generation.
config_builder_with_seed_jcommonsenseqa.with_seed_dataset(
    seed_dataset_reference_jcommonsense,
)

###
# # jcommonsenseqaデータ生成
# 23
import ipdb; ipdb.set_trace()

# seedデータの列（age, gender, occupation等）を直接参照
config_builder_with_seed_jcommonsenseqa.add_column(
    LLMStructuredColumnConfig(
        name="jcqa_data",
        model_alias=MODEL_ALIAS,
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            "以下のペルソナに基づいて、日本語の常識推論問題を生成してください。\n"
            "ただし、答えが個人の好み・性格・価値観で変わる内容は禁止し、\n"
            "日本で一般的な知識・慣習・手順により答えが1つに定まる問題だけを作成してください。\n\n"
            "【ペルソナ情報（最小）】\n"
            "職業: {{ occupation }}\n"
            "居住地（都道府県）: {{ prefecture }}\n"
            "{% if region is defined %}生活圏（地方）: {{ region }}\n{% endif %}"
            "{% if marital_status is defined %}家族: {{ marital_status }}\n{% endif %}"
            "\n"
            "【状況カテゴリ指定】\n"
            "状況カテゴリは『{{ topic_category }}』としてください。\n\n"
            "【テーマ指定】\n"
            "この問題のテーマは『{{ jc_theme }}』としてください。\n"
            "このテーマと状況カテゴリの両方に整合する日本の一般常識（用途・場所・手順・呼称）で答えが一意に決まる問題にしてください。\n"
            "ただし、両者が矛盾する場合は『{{ jc_theme }}』に整合する内容を優先してください。\n\n"
            "【問題作成の条件】\n"
            "- 質問は具体的な状況を1〜2文\n"
            "- 選択肢は5つ。全て一見もっともらしいが、正解は1つだけ\n"
            "- 正解以外の4つは同カテゴリで意味が近く紛らわしい語を使う\n"
            "- 心理/感情/道徳で割れる内容は禁止（例：嬉しい/悲しい/正しい/許せない/思いやり 等）\n\n"
            "【出力形式（必ずこの形式を守ってください）】\n"
            "question: <質問文>\n"
            "choices:\n"
            "0. <選択肢>\n"
            "1. <選択肢>\n"
            "2. <選択肢>\n"
            "3. <選択肢>\n"
            "4. <選択肢>\n"
            "answer: <0〜4の数字を1文字で出力>\n"
        ),
        output_format=JCommonsenseQAData,
    )
)

print("jcommonsenseqa生成カラムを追加しました")


###
# 24
# 品質評価用のルーブリック定義
import ipdb; ipdb.set_trace()
QuestionClarityRubric = Score(
    name="question_clarity",
    description="質問や状況の明確性の評価",
    options={
        "明確": "質問や状況が具体的で明確に記述されており、意図が容易に理解できる。",
        "やや不明確": "質問や状況は理解できるが、一部曖昧な表現や不足している情報がある。",
        "不明確": "質問や状況の意図が不明確で、追加の説明が必要。",
    },
)

DifficultyRubric = Score(
    name="difficulty",
    description="問題の難易度の評価",
    options={
        "易しい": "日本人の多くが直感的に答えられるレベルの問題。背景知識がほとんど不要。",
        "普通": "一般的な日本人であれば少し考えれば答えられるレベルの問題。基本的な知識や状況理解が必要。",
        "難しい": "ある程度の知識や深い理解、丁寧な読み取りが必要となる問題。人によっては間違えやすい。",
    },
)

# Seedあり版に品質評価を追加
config_builder_with_seed_jcommonsenseqa.add_column(
    LLMJudgeColumnConfig(
        name="quality_metrics",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=(
            "以下の生成されたデータの品質を評価してください:\n\n"
            # jcqa_dataが存在する場合
            "【jcommonsenseqa問題】\n"
            "質問: {{ jcqa_data.question }}\n"
            "選択肢:\n"
            "0. {{ jcqa_data.choice0 }}\n"
            "1. {{ jcqa_data.choice1 }}\n"
            "2. {{ jcqa_data.choice2 }}\n"
            "3. {{ jcqa_data.choice3 }}\n"
            "4. {{ jcqa_data.choice4 }}\n"
            "正解: {{ jcqa_data.answer_index }}\n"
            "推論: {{ jcqa_data.reasoning }}\n"
            "上記の内容について、日本の文化的適合性、内容の明確性、推論・説明の質を評価してください。"
        ),
        scores=[
            QuestionClarityRubric,
            DifficultyRubric,
        ],
    )
)


# スコアを個別のカラムに抽出
for builder in [config_builder_with_seed_jcommonsenseqa]:
    builder.add_column(
        ExpressionColumnConfig(
            name="clarity_score",
            expr="{{ quality_metrics.question_clarity.score if quality_metrics else 'N/A' }}",
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="difficulty",
            expr="{{ quality_metrics.difficulty.score if quality_metrics else 'N/A' }}",
        )
    )

print("品質評価カラムを追加しました")


###
# 25
import ipdb; ipdb.set_trace()
# Seedあり版のプレビュー
print("\n" + "="*0)
print("Seedデータあり版のプレビューを生成中...")
print("="*10)

preview_with_seed_jcommonsenseqa = data_designer_client.preview(
    config_builder_with_seed_jcommonsenseqa,
    num_records=1,
) # NOTE TODO

print("\nプレビュー生成完了!")
preview_with_seed_jcommonsenseqa.display_sample_record()


###
# 26
# プレビューの分析を表示
print("\nプレビューデータの分析:")
preview_with_seed_jcommonsenseqa.analysis.to_report()


###
# 27

# プレビューデータをDataFrameとして確認
preview_df = preview_with_seed_jcommonsenseqa.dataset
print("\nプレビューデータの最初の数件:")
print(preview_df.head())


###
# 28
import ipdb; ipdb.set_trace()
# Seedあり版の本番生成
NUM_RECORDS = 100 #8000  # 必要に応じて調整

print("\n" + "="*80)
print(f"Seedデータあり版 {NUM_RECORDS}件のデータを生成中...")
print("="*80)

job_with_seed = data_designer_client.create(
    config_builder_with_seed_jcommonsenseqa,
    num_records=NUM_RECORDS,
) # NOTE TODO

print("ジョブを実行中... 完了を待機しています")

# ジョブ完了を待機
results_with_seed = job_with_seed.wait_until_done()
print("\nSeedあり版の生成完了!")


### 
# 29
# Seedあり版の分析
print("\n" + "="*80)
print("Seedデータあり版の分析")
print("="*80)

analysis_with_seed = job_with_seed.load_analysis()
analysis_with_seed.to_report()


###
# 30
import ipdb; ipdb.set_trace()
# データの読み込み
df_with_seed = job_with_seed.load_dataset()

print(f"Seedあり版のデータ数: {len(df_with_seed)}")

# 品質スコアの集計
def count_scores(df, name):
    print(f"\n{name}の品質スコア分布:")

    for metric in ['clarity_score', 'difficulty']:
        if metric in df.columns:
            counts = df[metric].value_counts()
            print(f"\n{metric}:")
            print(counts)

    scores = {
        'clarity': df['clarity_score'].value_counts().to_dict() if 'clarity_score' in df.columns else {},
        'difficulty': df['difficulty'].value_counts().to_dict() if 'difficulty' in df.columns else {},
    }
    return scores

scores_with_seed = count_scores(df_with_seed, "Seedあり版")


###
# 31
OUTPUT_DIR = f"jcommonsenseqa_{NUM_RECORDS}_filter_jcommonsenseqa_seed_2000_temperature_0_9"
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seedあり版の保存
job_with_seed.download_artifacts(
    output_path=OUTPUT_DIR,
    artifacts_folder_name="with_seed_data",
)

df_with_seed = job_with_seed.load_dataset()

# DataFrameをJSONLで保存（LoRAチューニング用）
df_with_seed.to_json(
    f"{OUTPUT_DIR}/with_seed_data.jsonl",
    orient='records',
    lines=True,
    force_ascii=False
)

print(f"\nデータを '{OUTPUT_DIR}' ディレクトリに保存しました。")
print("\n保存されたファイル:")
for filename in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(filepath):
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {filename} ({size:.2f} KB)")

