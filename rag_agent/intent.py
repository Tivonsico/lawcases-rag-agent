"""
意图识别模块
1. jieba 中文分词 + 法律领域词典 → 提取有语义的关键词
2. 分类器：关键词多 = 复杂意图 → LLM 改写
              关键词少 = 简单意图 → 直接下一步
3. LLM 同义词扩展
"""
import re
import json
from typing import List, Optional

import jieba
import jieba.analyse

# ── 初始化 jieba，加载法律领域自定义词典 ──
LEGAL_TERMS = [
    # 产品/侵权类
    "产品责任", "产品质量", "产品缺陷", "产品侵权", "产品召回",
    "产品责任侵权", "产品质量侵权", "产品责任纠纷",
    "生产者责任", "销售者责任", "制造者责任",
    # 合同类
    "合同纠纷", "合同违约", "合同效力", "合同解除",
    "买卖合同", "借款合同", "租赁合同", "劳动合同",
    # 侵权类
    "侵权责任", "人身损害", "精神损害", "损害赔偿",
    "交通事故", "医疗损害", "医疗纠纷", "工伤认定",
    "工伤赔偿", "工伤鉴定", "工伤事故",
    # 公司/商事
    "股权转让", "股东纠纷", "公司清算", "破产清算",
    "知识产权", "商标侵权", "专利侵权", "著作权",
    # 婚姻/家庭
    "离婚纠纷", "财产分割", "子女抚养", "继承纠纷",
    "遗产分配", "抚养权", "探视权",
    # 房产/土地
    "房产纠纷", "房屋买卖", "房屋租赁", "拆迁补偿",
    "土地使用权", "建设工程",
    # 劳动/人事
    "劳动仲裁", "劳动争议", "劳动关系", "劳务派遣",
    "工伤认定", "社会保险", "经济补偿", "违法解除",
    # 刑事
    "刑事犯罪", "刑事辩护", "取保候审", "故意伤害",
    "危险驾驶", "合同诈骗", "职务侵占", "贪污受贿",
    # 诉讼/程序
    "管辖权", "诉讼时效", "证据保全", "财产保全",
    "强制执行", "执行异议", "再审申请", "上诉",
    # 行政
    "行政处罚", "行政复议", "行政诉讼", "行政许可",
    # 指导性案例相关
    "指导性", "指导性文件", "指导性案例", "最高人民法院",
    "最高人民检察院", "裁判要旨", "裁判规则",
    # 通用法律
    "法律咨询", "法律顾问", "法律援助", "法律服务",
    "涉案", "违法行为", "法律责任", "法律适用",
    "当事人", "利害关系", "连带责任", "补充责任",
    "免责", "免责条款", "格式条款", "诚实信用",
    "公平原则", "自愿原则", "公序良俗",
    "不当得利", "无因管理", "不可抗力", "情势变更",
    "缔约过失", "违约责任", "侵权责任", "举证责任",
    # 案号/编号相关
    "号",
]
for term in LEGAL_TERMS:
    jieba.add_word(term, freq=200, tag="n")

# 确保这些词不被拆分
for w in ["指导性文件", "指导性案例", "产品责任侵权", "产品责任", "故意伤害罪", "量刑标准"]:
    jieba.suggest_freq(w, tune=True)

# ── 停用词 ──
STOP_WORDS = set(
    "的了在是有一不把我对人就与他这那也还个而和"
    "及但或因为所以虽然如果可以怎么什么吗啊哦嗯"
    "呢吧呀哈嘿哟呵啦嘛呗咯哇呐呕咦诶啵喽嘛唦"
)
NOISE_WORDS = {"你好", "请问", "谢谢", "感谢", "我想", "我要"}


def clean_query(text: str) -> str:
    """清洗输入：仅去掉无意义单字语气词和冗余空格，保留语义结构"""
    # 去标点（但保留空格分隔）
    text = re.sub(r'''[，。！？、；："'（）【】《》｡．.,!?+~·…&#@%^*()\[\]{}/\\|<>-]''', " ", text)
    # 合并连续空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str) -> List[str]:
    """
    用 jieba TextRank 提取关键词
    - 基于词共现图排序，天然过滤高频无意义词
    - 保留法律词典中的术语
    """
    if not text.strip():
        return []

    # TextRank 提取（allowPOS 不限制词性，让图算法自己排序）
    keywords = jieba.analyse.textrank(text, topK=8, withWeight=False)

    # 过滤停用词和噪音词
    result = []
    for w in keywords:
        w = w.strip()
        if not w or w in NOISE_WORDS:
            continue
        if all(c in STOP_WORDS for c in w):
            continue
        if len(w) == 1 and w in STOP_WORDS | {"的了吗是不"}:
            continue
        result.append(w)

    # 如果 TextRank 什么都没留下，直接返回分词后的有效词
    if not result:
        tokens = [t for t in jieba.lcut(text) if t.strip() and t not in NOISE_WORDS
                  and not all(c in STOP_WORDS for c in t)]
        return tokens[:5]
    return result[:8]


def classify_intent(keywords: List[str]) -> str:
    if len(keywords) <= 3:
        return "simple"
    return "complex"




class IntentProcessor:
    """意图处理器：jieba TextRank 提取关键词 + 精确案号匹配"""

    def __init__(self, llm_call_func=None):
        """
        llm_call_func: 已废弃，保留参数兼容旧调用方
        """
        self._cache = {}

    def process(self, user_query: str) -> dict:
        """
        完整意图处理流程（纯 jieba，不调 LLM）
        返回: {
            "original": 原始输入,
            "clean": 清洗后文本,
            "keywords": 核心关键词列表,
            "intent_type": "simple" / "complex",
            "expanded_keywords": 同 keywords,
            "normalized_query": 用于 embedding_normalized 的查询,
            "exact_terms": 用于 exact 通道的案号匹配,
        }
        """
        # 缓存命中
        cached = self._cache.get(user_query)
        if cached is not None:
            return cached

        # 1. 清洗
        clean = clean_query(user_query)
        keywords = extract_keywords(clean)
        exact_terms = self._extract_exact_terms(clean)

        if not keywords:
            ret = {
                "original": user_query, "clean": clean,
                "keywords": [], "intent_type": "simple",
                "expanded_keywords": [], "normalized_query": clean,
                "exact_terms": exact_terms,
            }
            self._cache[user_query] = ret
            return ret

        ret = {
            "original": user_query, "clean": clean,
            "keywords": keywords,
            "intent_type": classify_intent(keywords),
            "expanded_keywords": list(keywords),
            "normalized_query": " ".join(keywords),
            "exact_terms": exact_terms,
        }
        self._cache[user_query] = ret
        return ret

    @staticmethod
    def _extract_exact_terms(text: str) -> List[str]:
        patterns = [r"指导性?案例\s*\d+\s*号", r"检例第\s*\d+\s*号", r"FBM-[A-Za-z0-9-]+"]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        return list(dict.fromkeys(re.sub(r"\s+", "", item) for item in found))


if __name__ == "__main__":
    # 测试
    queries = [
        "我想了解一下五险一金的缴纳比例是多少啊？",
        "你好，请问养老保险怎么交，医疗保险怎么报销，还有失业保险怎么领，公积金怎么提取？",
        "车险怎么买最划算",
    ]
    ip = IntentProcessor()
    for q in queries:
        result = ip.process(q)
        print(f"\n输入: {q}")
        print(f"清洗: {result['clean']}")
        print(f"关键词: {result['keywords']}")
        print(f"意图类型: {result['intent_type']}")
        print(f"扩展后: {result['expanded_keywords']}")
