"""Theme bucket labels for stratified universe prescreen (name-based)."""
from __future__ import annotations


def refined_industry_label(name: object) -> str:
    """Infer a coarse theme bucket from instrument name (ETF/stock)."""
    text = str(name).upper()
    if "通信" in text:
        return "communication"
    if "人工智能" in text or "AI" in text:
        return "ai"
    if "科创" in text and "半导体" in text:
        return "star_semiconductor"
    if "半导体" in text or "芯片" in text or "集成电路" in text:
        return "semiconductor"
    if "算力" in text or "曙光" in text or "寒武纪" in text or "海光" in text:
        return "ai_compute"
    if "油气" in text or "石油" in text or "能源" in text:
        return "energy"
    if "化工" in text or "石化" in text:
        return "chemical"
    if "红利" in text or "股息" in text:
        return "dividend"
    if "医药" in text or "医疗" in text or "创新药" in text or "生物医药" in text:
        return "healthcare"
    if "证券" in text or "券商" in text:
        return "brokerage"
    if "军工" in text or "国防" in text:
        return "defense"
    if "有色" in text or "矿业" in text or "黄金" in text or "金ETF" in text:
        return "metals"
    if "金融" in text or "银行" in text:
        return "financial"
    if "房地产" in text or "地产" in text:
        return "real_estate"
    if "消费" in text:
        return "consumer"
    if "游戏" in text:
        return "game"
    if "卫星" in text:
        return "satellite"
    if "法国" in text or "沙特" in text or "日经" in text or "纳指" in text or "标普" in text:
        return "overseas"
    if "科创" in text:
        return "star_market"
    if "中创" in text or "治理" in text or "180" in text or "400" in text:
        return "broad_market"
    return "other"
