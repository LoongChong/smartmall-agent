"""模拟业务数据。

**明确声明：这是模拟数据，不连接任何真实客户、订单或支付系统。**
数据是确定性的（固定种子、固定时间基准），保证测试可复现。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..models import Logistics, LogisticsEvent, Order, OrderStatus, Product

#: 所有相对时间以此为基准，保证测试结果不随运行日期漂移。
#: 真实部署时业务数据来自数据库，不会用这个常量。
NOW = datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC)


PRODUCTS: dict[str, Product] = {
    "P1001": Product(
        product_id="P1001",
        name="星云 Pro 14 笔记本电脑",
        category="笔记本电脑",
        price=5999.0,
        stock=42,
        attributes={
            "屏幕": "14 英寸 2.8K 120Hz",
            "处理器": "星核 X9 八核",
            "内存": "16GB LPDDR5",
            "存储": "512GB PCIe 4.0",
            "重量": "1.29kg",
            "保修": "整机 2 年",
        },
        return_days=7,
    ),
    "P2002": Product(
        product_id="P2002",
        name="清风 S3 无线降噪耳机",
        category="耳机",
        price=799.0,
        stock=0,  # 故意设为缺货，用于测试缺货场景
        attributes={
            "蓝牙": "5.3",
            "降噪": "最高 48dB",
            "续航": "单次 8 小时 / 总 32 小时",
            "防水": "IPX5",
            "保修": "整机 1 年",
        },
        return_days=7,
    ),
    "P3003": Product(
        product_id="P3003",
        name="磐石 定制刻字保温杯",
        category="生活用品",
        price=129.0,
        stock=180,
        attributes={"容量": "500ml", "材质": "316 不锈钢", "定制": "支持激光刻字"},
        return_days=0,  # 定制商品不支持无理由退货
    ),
}


ORDERS: dict[str, Order] = {
    # U001 的订单：已签收 3 天，在 7 天无理由窗口内
    "SO20261001001": Order(
        order_id="SO20261001001",
        user_id="U001",
        product_id="P1001",
        product_name="星云 Pro 14 笔记本电脑",
        quantity=1,
        amount=5999.0,
        status=OrderStatus.DELIVERED,
        created_at=NOW - timedelta(days=8),
        delivered_at=NOW - timedelta(days=3),
        tracking_no="SF1234567890",
    ),
    # U001 的订单：已签收 20 天，超出 7 天无理由
    "SO20260901002": Order(
        order_id="SO20260901002",
        user_id="U001",
        product_id="P2002",
        product_name="清风 S3 无线降噪耳机",
        quantity=2,
        amount=1598.0,
        status=OrderStatus.DELIVERED,
        created_at=NOW - timedelta(days=28),
        delivered_at=NOW - timedelta(days=20),
        tracking_no="YT9876543210",
    ),
    # U001 的订单：运输中，尚未签收
    "SO20261010003": Order(
        order_id="SO20261010003",
        user_id="U001",
        product_id="P3003",
        product_name="磐石 定制刻字保温杯",
        quantity=1,
        amount=129.0,
        status=OrderStatus.SHIPPED,
        created_at=NOW - timedelta(days=2),
        delivered_at=None,
        tracking_no="JD5555666677",
    ),
    # U002 的订单：用于越权访问测试
    "SO20261005004": Order(
        order_id="SO20261005004",
        user_id="U002",
        product_id="P1001",
        product_name="星云 Pro 14 笔记本电脑",
        quantity=1,
        amount=5999.0,
        status=OrderStatus.PAID,
        created_at=NOW - timedelta(days=5),
        delivered_at=None,
        tracking_no=None,
    ),
}


LOGISTICS: dict[str, Logistics] = {
    "SF1234567890": Logistics(
        tracking_no="SF1234567890",
        carrier="顺丰速运",
        status="已签收",
        events=[
            LogisticsEvent(time=NOW - timedelta(days=6), location="深圳转运中心", description="快件已揽收"),
            LogisticsEvent(time=NOW - timedelta(days=4), location="杭州转运中心", description="快件到达目的城市"),
            LogisticsEvent(
                time=NOW - timedelta(days=3), location="杭州西湖营业点", description="快件已签收，签收人：本人"
            ),
        ],
    ),
    "YT9876543210": Logistics(
        tracking_no="YT9876543210",
        carrier="圆通速递",
        status="已签收",
        events=[
            LogisticsEvent(time=NOW - timedelta(days=24), location="广州分拨中心", description="快件已揽收"),
            LogisticsEvent(
                time=NOW - timedelta(days=20), location="杭州余杭营业点", description="快件已签收，签收人：前台代收"
            ),
        ],
    ),
    "JD5555666677": Logistics(
        tracking_no="JD5555666677",
        carrier="京东物流",
        status="运输中",
        events=[
            LogisticsEvent(time=NOW - timedelta(days=2), location="上海仓", description="商品已出库"),
            LogisticsEvent(time=NOW - timedelta(hours=10), location="嘉兴分拣中心", description="快件运输中"),
        ],
    ),
}
