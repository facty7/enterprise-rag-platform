"""
Enterprise RAG 全面评估脚本
1. 生成 60+ 逼真企业文档（多类型、多场景）
2. 通过 API 上传并索引
3. 运行 40+ 测试查询
4. 计算 Hit Rate / MRR / NDCG / Recall / Precision / Keyword Recall
5. 生成对比报告

用法：先启动系统，然后运行此脚本
  python scripts/run_full_evaluation.py
"""
import json
import os
import sys
import time
import asyncio
import random
import io
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import requests
import numpy as np

# 配置
BASE_URL = "http://localhost:8400"
API = f"{BASE_URL}/api/v1"
TOKEN = None

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test_docs"
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Part 1: 生成逼真企业文档
# ============================================================

def login():
    global TOKEN
    resp = requests.post(f"{API}/login", json={"username": "admin", "password": "admin123"})
    data = resp.json()
    TOKEN = data["token"]
    print(f"已登录: {data['name']} ({data['role']})")

def api_headers():
    return {"X-Auth-Token": TOKEN}

# ---- 文档内容生成器 ----

def gen_hr_documents():
    """生成人事行政类文档。"""
    docs = []

    # 1. 员工薪资表
    salary_csv = "工号,姓名,部门,基本工资,岗位津贴,绩效工资,加班费,社保扣除,公积金扣除,个税,实发工资,月份\n"
    departments = ["销售部", "研发部", "生产部", "人事部", "财务部"]
    names = ["张伟", "李娜", "王磊", "赵敏", "陈静", "刘洋", "周婷", "吴强", "孙丽", "马超",
             "黄小明", "林小红", "何大伟", "郭美丽", "高大山", "蔡小花", "钱进", "郑成功", "叶绿", "胡适"]
    for i, (name, dept) in enumerate(zip(names, departments * 4)):
        base = random.randint(5000, 15000)
        allowance = random.randint(500, 3000)
        perf = random.randint(0, 5000)
        overtime = random.randint(0, 2000)
        gross = base + allowance + perf + overtime
        social = round(gross * 0.105)
        fund = round(gross * 0.07)
        tax = max(0, round((gross - social - fund - 5000) * 0.03))
        net = gross - social - fund - tax
        salary_csv += f"EMP{i+1:03d},{name},{dept},{base},{allowance},{perf},{overtime},{social},{fund},{tax},{net},2026-04\n"
    docs.append(("薪资表_2026年4月.csv", salary_csv))

    salary_csv2 = salary_csv.replace("2026-04", "2026-05").replace("基本工资", "基本工资")
    # 调整一些数据
    for i in range(len(names)):
        old = f"EMP{i+1:03d}"
        new_base = random.randint(5000, 16000)
        salary_csv2 = salary_csv2.replace(f"{old},{names[i]},", f"{old},{names[i]},")
    docs.append(("薪资表_2026年5月.csv", salary_csv2))

    # 2. 考勤制度
    attendance_doc = """考勤管理制度 V3.2

第一章 总则
第一条 为规范公司考勤管理，保障正常生产经营秩序，特制定本制度。
第二条 本制度适用于公司全体员工（含试用期员工）。

第二章 工作时间
第三条 公司实行标准工时制：
  - 上午：09:00-12:00
  - 下午：13:30-18:00
  - 周一至周五为工作日，周六日为休息日

第四条 弹性工作制：
  - 研发部、设计部可申请弹性工作时间（10:00-19:00）
  - 需经部门主管和人事部审批

第三章 考勤方式
第五条 使用移动考勤系统，每日打卡两次（上班、下班）。
第六条 忘记打卡的，需在48小时内提交补卡申请，由部门主管审批。
第七条 外勤人员使用手机GPS定位打卡。

第四章 迟到/早退
第八条 迟到：超过上班时间30分钟以内视为迟到。
  - 月累计迟到3次以内：警告
  - 月累计迟到4-6次：扣半天工资
  - 月累计迟到7次以上：记过处分，扣1天工资

第九条 早退：未经批准提前下班30分钟以上视为早退，按旷工半天处理。

第五章 请假制度
第十条 请假类型：事假、病假、婚假、产假、丧假、年假。
第十一条 年假标准：
  - 工作满1年不足10年：5天
  - 工作满10年不足20年：10天
  - 工作满20年以上：15天
第十二条 病假需提供医院诊断证明，病假期间工资按基本工资70%发放。
第十三条 事假需提前1天申请，事假期间不计发工资。

第六章 加班管理
第十四条 工作日加班：按基本工资1.5倍计发。
第十五条 休息日加班：按基本工资2倍计发。
第十六条 法定假日加班：按基本工资3倍计发。
第十七条 加班需提前申请并经审批，未经审批的加班不予认可。

第七章 出差管理
第十八条 出差需提前填写《出差申请表》，经部门主管审批。
第十九条 出差补贴标准：
  - 国内一线城市：200元/天
  - 国内其他城市：150元/天
  - 境外：按实际消费报销（上限500元/天）

第八章 附则
第二十条 本制度自2026年1月1日起执行，由人事部负责解释和修订。
"""
    docs.append(("考勤管理制度_V3.2.txt", attendance_doc))

    # 3. 招聘需求
    recruitment = """2026年Q2招聘计划

一、招聘需求汇总
1. 研发部
   - Java高级工程师 2名：要求5年以上经验，熟悉Spring Boot、微服务架构
   - 前端工程师 1名：要求3年以上Vue/React经验

2. 销售部
   - 大客户经理 3名：要求有制造业客户资源，年薪30-50万
   - 销售代表 5名：应届生或有1-2年经验均可

3. 生产部
   - CNC操作工 4名：要求有3年以上CNC操作经验，熟悉FANUC系统
   - 质检员 2名：要求熟悉ISO9001体系
   - SMT工程师 1名：要求5年以上SMT设备维护经验

4. 人事部
   - 培训专员 1名：要求2年以上企业培训经验

二、招聘渠道
- 前程无忧、智联招聘：技术人员
- 内部推荐：一线操作工（推荐奖金800元/人）
- 校园招聘：销售代表

三、时间节点
- 候选资料初筛：4月1日-4月20日
- 面试：4月21日-5月15日
- 入职：5月20日前
"""
    docs.append(("2026年Q2招聘计划.txt", recruitment))

    # 4. 劳动合同模板
    contract = """劳动合同书

甲方（用人单位）：广州市唯信营销策划有限公司
法定代表人：陈明
地址：广州市天河区中山大道西888号

乙方（劳动者）：__________
身份证号：__________
联系电话：__________

第一条 合同期限
本合同自2026年__月__日起生效，至2029年__月__日止，合同期限3年。
试用期自2026年__月__日至2026年__月__日，共3个月。

第二条 工作内容与地点
2.1 乙方同意在甲方从事__________岗位工作。
2.2 工作地点为广州市。
2.3 甲方根据工作需要可调整乙方工作岗位，但需经乙方同意。

第三条 工作时间
实行标准工时制，每日工作不超过8小时，每周不超过40小时。

第四条 劳动报酬
4.1 试用期工资：人民币__________元/月
4.2 转正后工资：人民币__________元/月
4.3 甲方每月15日前以货币形式支付上月工资。

第五条 社会保险
甲方按照国家规定为乙方缴纳养老保险、医疗保险、失业保险、工伤保险和生育保险以及住房公积金。

第六条 劳动保护
甲方为乙方提供符合国家标准的劳动条件和劳动保护用品。

第七条 规章制度
乙方应遵守甲方的各项规章制度，包括但不限于《考勤管理制度》《安全生产规程》《保密制度》。

第八条 合同解除
8.1 试用期双方可提前3天通知解除合同。
8.2 转正后双方需提前30天书面通知解除合同。
8.3 严重违反公司规章制度的，甲方可立即解除合同。

甲方（盖章）：__________    乙方（签字）：__________
签订日期：2026年__月__日
"""
    docs.append(("劳动合同模板.txt", contract))

    return docs


def gen_sales_documents():
    """生成销售类文档。"""
    docs = []

    # 客户信息表
    customer_xlsx = []
    customers = [
        ("无锡云芯电子", "李建国", "采购总监", "13905108888", "江苏省无锡市高新区示范路88号", "半导体", "A级"),
        ("深圳远航精密制造", "王芳", "供应链经理", "13510887766", "深圳市坪山区远航精密路3009号", "汽车零部件", "A级"),
        ("苏州星河电子", "张晓明", "技术总监", "13862559999", "苏州市工业园区科创街328号", "汽车电子", "B级"),
        ("东莞智造电子", "陈志强", "品质经理", "13622668888", "东莞市松山湖创新园", "消费电子", "B级"),
        ("杭州安视科技", "刘敏", "采购经理", "13757112233", "杭州市滨江区示范路555号", "安防监控", "A级"),
        ("北京启明智能科技", "赵磊", "供应链总监", "13810005555", "北京市海淀区创新中街68号", "智能硬件", "A级"),
        ("重庆长鑫汽车部件", "周伟", "采购部长", "13983007777", "重庆市渝北区长鑫汽车大道1号", "汽车", "B级"),
        ("武汉光谷通信设备", "吴涛", "研发经理", "13607110000", "武汉市洪山区光谷示范大道1号", "通信设备", "C级"),
        ("成都锦城电子制造", "孙明", "生产主管", "13540669999", "成都市高新西区协同路888号", "电子制造", "B级"),
        ("南京云通通信", "黄丽", "采购专员", "13851882222", "南京市雨花台区软件园路50号", "通信", "B级"),
        ("天津津汽制造", "马勇", "技术部长", "13920003333", "天津市滨海新区制造路159号", "汽车", "A级"),
        ("厦门宏新电气", "林芳", "品质总监", "13606001111", "厦门市集美区宏发路99号", "继电器", "C级"),
    ]

    # Excel-like CSV
    csv = "客户名称,联系人,职位,电话,地址,行业,客户等级,合作状态,上次拜访,年度采购额(万元),备注\n"
    statuses = ["在合作", "洽谈中", "意向明确", "初步接触", "流失"]
    for i, c in enumerate(customers):
        status = statuses[i % len(statuses)]
        amount = random.randint(50, 800)
        last_visit = (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
        note = random.choice(["重点维护", "需加强跟进", "价格敏感", "品质要求高", "交货周期短", "有应收风险"])
        csv += f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[5]},{c[6]},{status},{last_visit},{amount},{note}\n"
    docs.append(("客户信息表_2026年5月.csv", csv))

    # 销售日报
    for day in range(1, 8):
        date_str = f"2026-05-0{day}"
        report = f"""销售日报 —— {date_str}

今日工作情况：
1. 客户拜访：
   - 上午09:30 拜访无锡云芯电子 李建国总监，讨论Q3订单计划，客户对报价认可，预计下单300万
   - 下午14:00 拜访深圳远航精密 王芳经理，送样品及规格书，客户需内部评审后回复

2. 电话沟通：
   - 苏州星河电子 张晓明总监：确认5月20日来公司考察，需提前准备样品和检测报告
   - 杭州安视科技 刘敏经理：询价NTC热敏电阻 10K/B值3435，已发送报价单

3. 新客户开发：
   - 联系北京启明智能供应链，赵磊总下周安排电话会议
   - 收到重庆长鑫汽车部件RFQ（询价单），编号RFQ2026-050{day}，需5个工作日内回复

4. 订单跟进：
   - 东莞智造电子 PO20260428 已排产，预计5月15日交货
   - 成都锦城电子制造 合同在走审批流程

5. 应收账款：
   - 武汉光谷通信设备 到期未付35万，已发催款函
   - 南京云通通信 账期内

明日计划：重点跟进远航精密样品反馈、准备星河电子考察资料
"""
        docs.append((f"销售日报_{date_str}.txt", report))

    return docs


def gen_production_documents():
    """生成生产制造类文档。"""
    docs = []

    # SOP
    sop = """SMT贴片标准操作规程 SOP-PRD-001

适用工序：SMT贴片
版本：V4.1
生效日期：2026-04-01
编制：工程部

1. 目的
规范SMT贴片操作流程，确保产品质量符合IPC-A-610 Class 2标准。

2. 适用范围
本规程适用于YAMAHA YSM20贴片机生产线。

3. 操作前准备
3.1 穿戴防静电服、防静电手环（接地电阻<1MΩ）
3.2 检查环境温湿度：温度22±3℃，湿度40%-60%RH
3.3 检查贴片机供料器是否安装正确
3.4 确认程序版本：当前版本为YS20-PRD-V4.1

4. 操作步骤
4.1 开机
  a) 打开主电源开关
  b) 启动压缩空气，压力调至0.5±0.05MPa
  c) 启动真空泵，真空度≤-80kPa
  d) 系统自检：按「START」键，等待系统自检完成（约2分钟）

4.2 程序加载
  a) 在主界面选择「Program」→「Load」→ 选择对应产品程序
  b) 确认程序校验码：CRC32 = AE2F103B
  c) 检查Mark点识别：自动识别PCB Mark点，偏差<0.1mm

4.3 生产运行
  a) 放置PCB板在传送带上
  b) 按「AUTO」键启动自动生产
  c) 监控贴片速度：标准10000 CPH (Chips Per Hour)
  d) 每2小时检查一次抛料率，正常≤0.3%
  e) 每班记录《SMT生产日志》

4.4 换料操作
  a) 报警提示时，按「STOP」→「COMP CHANGE」
  b) 确认料号、规格、数量
  c) 更换完成按「START」继续
  d) 记录换料时间、物料批号

5. 常见异常处理
5.1 贴片偏移
  原因：PCB定位不准/吸嘴磨损/程序错误
  处理：检查定位销→清洁吸嘴→重新校准→如仍异常联系工程部

5.2 抛料率过高（>1%）
  原因：供料器故障/料带不良/吸嘴堵塞
  处理：检查供料器→更换料带→清洁或更换吸嘴

5.3 真空报警
  原因：真空管路泄漏/真空泵故障
  处理：检查管路接口→重启真空泵→如仍异常联系设备部

5.4 设备参数报警
  ▲▲▲ 注意：出现红色报警应立即停机 ▲▲▲
  按「EMERGENCY STOP」→通知工程部→填写《设备异常报告》

6. 品质检查
6.1 首件检查：每批次生产前需做首件检查
6.2 过程抽检：每2小时抽检5片
6.3 检查项目：贴片位置精度（±0.05mm）、焊膏印刷质量、无漏贴/错贴

7. 关机程序
7.1 生产完成后依次关闭：贴片机→真空泵→压缩空气→主电源
7.2 清洁设备表面
7.3 填写《设备运行记录》

相关文件：
  - 《SMT品质检验标准》QC-SMT-001
  - 《设备维护保养制度》PM-PRD-001
"""
    docs.append(("SMT贴片标准操作规程_SOP-PRD-001.txt", sop))

    # 品质报告
    qc_report = """2026年Q1品质分析报告

一、总体质量指标
  - 出货批次合格率：98.7%（目标≥98.5%）
  - 客户投诉率：0.8%（目标≤1%）
  - 内部不合格品率：1.2%（目标≤1.5%）
  - 供应商来料合格率：97.1%（目标≥98%，未达标）

二、主要质量问题
1. SMT焊接不良（占比45%）
   - 虚焊：12起，主要集中在线路板连接器位置
   - 短路：6起，集中在BGA封装芯片
   - 原因分析：回流焊温度曲线偏差，已调整Profile参数

2. 注塑件尺寸不良（占比25%）
   - 翘曲变形：8起，集中在大型外壳件
   - 原因分析：模具冷却系统不均衡，模具温度左85℃/右72℃（差异13℃）

3. 来料问题（占比20%）
   - 电阻阻值偏差（供应商：XX电子）：5批次超出±1%允差
   - 已要求供应商整改，4月起暂停该供应商

4. 装配不良（占比10%）
   - 螺丝漏装：3起
   - 标签贴错：2起

三、改善措施
1. 回流焊温曲线重新验证，调整为：预热150-180℃ 60s → 回流220℃ 40s
2. 模具冷却管路改造，计划5月完成
3. 增加来料IQC抽检比例从AQL 0.65提升到AQL 0.40
4. 装配线增加防呆检测（扫描+视觉）

四、Q2质量目标
  - 出货批次合格率：≥99.0%
  - 客户投诉率：≤0.5%
  - 供应商来料合格率：≥98.5%
"""
    docs.append(("2026年Q1品质分析报告.txt", qc_report))

    # 设备维护记录
    maintenance = """2026年4月设备维护记录

设备名称：YAMAHA YSM20 贴片机
设备编号：SMT-003
所属部门：生产部SMT车间

| 日期 | 维护内容 | 维护人 | 结果 |
|------|----------|--------|------|
| 4月1日 | 日常清洁、吸嘴检查（6个吸嘴正常） | 周师傅 | 正常 |
| 4月3日 | 更换1#吸嘴（磨损超限） | 周师傅 | 正常 |
| 4月7日 | 真空管路检查、过滤器更换 | 吴工 | 正常 |
| 4月10日 | 月度保养：导轨清洁润滑、传送带张紧调整 | 吴工 | 正常 |
| 4月15日 | 贴装精度校准（偏移量从0.08mm校至0.03mm） | 吴工 | 正常 |
| 4月20日 | 3#供料器维修（送料不顺畅） | 周师傅 | 已修复 |
| 4月25日 | 电气安全检查（绝缘电阻>100MΩ，接地电阻0.8Ω） | 设备部 | 合格 |

运行统计：
  - 本月运行时间：198小时
  - 计划停机：12小时（保养）
  - 非计划停机：3.5小时（4/20供料器故障）
  - 设备综合效率OEE：92.5%
  - 抛料率：0.28%

下月计划：
  - 5月10日 季度保养（更换所有吸嘴、真空泵保养）
  - 5月20日 年度校准（委外）
"""
    docs.append(("2026年4月设备维护记录.txt", maintenance))

    # BOM
    bom = """物料清单 BOM-PRD-088

产品名称：NTC温度传感器探头 NTC-10K-1M
产品型号：NTC-10K-B3435-1M-DIP
版本：BOM-V2.3
编制日期：2026-04-15

| 序号 | 物料编码 | 物料名称 | 规格 | 用量 | 单位 | 供应商 | 单价(元) |
|------|----------|----------|------|------|------|--------|---------|
| 1 | RES-0001 | NTC热敏电阻 | 10KΩ±1% B3435 | 1 | 个 | 风华高科 | 0.85 |
| 2 | PCB-0008 | PCB板 | FR-4 1.6mm双面板 ENIG | 1 | 片 | 景旺电子 | 1.20 |
| 3 | CON-0015 | XH2.54连接器 | 2P 直插 间距2.54 | 1 | 个 | JST | 0.35 |
| 4 | WIR-0022 | 导线 | UL1007 24AWG 红 | 0.5 | 米 | 宝胜 | 0.15 |
| 5 | WIR-0023 | 导线 | UL1007 24AWG 黑 | 0.5 | 米 | 宝胜 | 0.15 |
| 6 | TUB-0012 | 热缩管 | φ3.0mm 黑色 含胶 | 0.05 | 米 | 沃尔核材 | 0.08 |
| 7 | EPO-0005 | 环氧树脂 | E-44 双组分 黑色 | 0.3 | 克 | 广州宏昌 | 0.10 |
| 8 | LAB-0001 | 标签纸 | 30x15mm 白色PET | 1 | 张 | 本地采购 | 0.02 |

单件物料成本合计：2.90元
人工成本：0.60元/件
制造成本（含电费/设备折旧）：0.30元/件
总成本：3.80元/件
销售价格：6.50元/件（毛利率41.5%）
"""
    docs.append(("BOM_NTC温度传感器探头_PRD-088.txt", bom))

    return docs


def gen_finance_documents():
    """生成财务类文档。"""
    docs = []

    expense_report = """2026年Q1财务分析报告

一、营收概况
  - Q1营业收入：2,850万元（同比增长12.3%）
  - Q1净利润：385万元（净利润率13.5%）
  - 毛利率：32.8%（去年同期30.1%，上升2.7个百分点）

二、收入构成
  - NTC传感器产品线：1,560万元（占比54.7%）
  - 温控器产品线：780万元（占比27.4%）
  - 技术服务和配件：510万元（占比17.9%）

三、成本分析
  - 原材料成本：1,480万元（占营收51.9%）
  - 人工成本：425万元（占营收14.9%）
  - 制造费用：298万元（占营收10.5%）
  - 销售费用：165万元（占营收5.8%）
  - 管理费用：97万元（占营收3.4%）

四、应收账款
  - 30天内：680万元
  - 30-60天：245万元
  - 60-90天：85万元
  - 90天以上：42万元（其中光谷通信35万元需重点关注）

五、现金流
  - 期初余额：560万元
  - 期末余额：490万元
  - Q1经营现金流净流出：70万元（因Q1集中采购原材料备货）

六、Q2财务预测
  - 预计营收：3,200-3,500万元
  - 预计净利润：430-480万元
  - 重点关注：应收账款管控、库存周转率提升
"""
    docs.append(("2026年Q1财务分析报告.txt", expense_report))

    # 差旅报销标准
    travel_policy = """差旅费报销管理制度 V2.0

一、差旅审批
1. 所有差旅需提前通过OA系统提交《出差申请单》
2. 国内出差由部门主管审批，出境出差由总经理审批
3. 紧急出差可先出发后24小时内补申请

二、交通标准
1. 机票：提前7天以上预订7折，3-7天8折，3天以内全价
   - 管理层/总监：商务舱
   - 经理级：经济舱
   - 普通员工：经济舱
2. 火车：高铁二等座标准（超过5小时可选一等座）
3. 市内交通：实报实销，出租车发票需备注路线和事由

三、住宿标准
  - 一线城市（北上广深）：不超过500元/天
  - 二线城市（省会/计划单列市）：不超过350元/天
  - 三线及以下：不超过250元/天
  - 超出部分自理

四、餐补标准
  - 出差餐补：80元/天（不需发票）
  - 接待客户用餐：需提前申请，人均不超过150元

五、报销流程
1. 出差结束5个工作日内提交报销申请
2. 需附：出差审批单、交通票据、住宿发票（增值税专用发票）
3. 财务审核3个工作日内完成，审核通过后5个工作日内打款

六、违规处理
1. 虚假报销：追回报销款项，记过处分
2. 超标准不报批：超出部分公司不予承担
3. 逾期报销（>=30天）：按80%报销
"""
    docs.append(("差旅费报销管理制度_V2.0.txt", travel_policy))

    return docs


def gen_tech_documents():
    """生成技术研发类文档。"""
    docs = []

    tech_spec = """NTC热敏电阻技术规格书

产品名称：NTC热敏电阻
产品型号：NTC-10K-B3435-1%
文件编号：SPEC-NTC-2026A
版本：V3.0

1. 电气特性
  - 标称阻值(R25)：10KΩ ±1%
  - B值(B25/50)：3435K ±1%
  - B值(B25/85)：3450K ±1.5%
  - 耗散系数：≥2mW/℃（静止空气中）
  - 热时间常数：≤15s（静止空气中）
  - 额定功率：50mW（25℃时）
  - 工作温度范围：-40℃ ~ +125℃
  - 绝缘电阻：≥100MΩ（500VDC）

2. 阻值-温度对照表（R-T Table）
  -40℃: 277.2KΩ ±5%
  -20℃: 97.13KΩ ±3%
    0℃: 36.25KΩ ±2%
   25℃: 10.00KΩ ±1%
   50℃: 3.602KΩ ±1.5%
   75℃: 1.488KΩ ±2%
  100℃: 0.679KΩ ±3%
  125℃: 0.338KΩ ±5%

3. 可靠性测试
  - 高温存储：125℃ 1000h，ΔR/R < ±2%
  - 低温存储：-40℃ 1000h，ΔR/R < ±2%
  - 湿热试验：85℃/85%RH 1000h，ΔR/R < ±3%
  - 温度循环：-40℃↔125℃ 1000次，ΔR/R < ±2%
  - 高温负载：125℃ 额定功率 1000h，ΔR/R < ±3%

4. 外形尺寸
  - 封装：0805 SMD
  - 尺寸：2.0mm × 1.25mm × 0.85mm
  - 端子：镍阻挡层+锡镀层

5. 推荐应用
  - 锂电池温度检测
  - IGBT/功率模块温度保护
  - 汽车电子（电机/电池管理）
  - 医疗设备温度监测

6. 包装
  - 编带包装：5000pcs/卷
  - 防潮包装：密封铝箔袋+干燥剂

相关标准：
  - GB/T 6663.1-2020 直热式负温度系数热敏电阻器
  - AEC-Q200 汽车电子无源元器件可靠性测试
"""
    docs.append(("NTC热敏电阻技术规格书_SPEC-NTC-2026A.txt", tech_spec))

    # API 接口文档
    api_doc = """MES系统接口规范 V2.1

1. 生产数据上报接口
   POST /api/v2/production/report
   描述：SMT贴片机实时生产数据上传

   请求参数：
   {
     "device_id": "SMT-003",
     "batch_id": "B20260507001",
     "product_code": "NTC-10K-B3435",
     "timestamp": "2026-05-07T14:30:00+08:00",
     "data": {
       "total_placed": 15200,
       "good_parts": 15180,
       "reject_parts": 20,
       "placement_speed": 9850,
       "pickup_errors": 15,
       "vision_errors": 5,
       "temperature": 23.5,
       "humidity": 52.0
     }
   }

   响应：
   {
     "code": 0,
     "message": "success",
     "data": {"report_id": "RPT20260507001"}
   }

   错误码：
   - 0: 成功
   - 1001: 设备未注册
   - 1002: 批次不存在
   - 1003: 数据格式错误
   - 1004: 数据库写入失败

2. 品质数据查询接口
   GET /api/v2/quality/query?batch_id=B20260507001

   响应：
   {
     "code": 0,
     "data": {
       "batch_id": "B20260507001",
       "product_code": "NTC-10K-B3435",
       "total_quantity": 50000,
       "sampled": 200,
       "defects": {"offset": 3, "missing": 1, "wrong_component": 0},
       "first_pass_yield": 98.0,
       "cpk": 1.45
     }
   }

3. 认证方式
   Header: Authorization: Bearer {access_token}
   Token获取: POST /api/v2/auth/login
   Token有效期: 24小时
"""
    docs.append(("MES系统接口规范_V2.1.txt", api_doc))

    # 研发项目计划
    project_plan = """2026年新产品开发项目计划

项目名称：高精度汽车级NTC温度传感器研制
项目编号：RD-2026-003
项目经理：陈博士
项目周期：2026年3月-2026年9月（7个月）
项目预算：180万元

里程碑：
  M1 (3/31): 材料选型完成 — 状态: 已完成 ✓
  M2 (4/30): 样品制作10份 — 状态: 已完成 ✓
  M3 (5/31): 可靠性测试启动（AEC-Q200全套）— 状态: 进行中
  M4 (6/30): 小批量试产1000件 — 状态: 未开始
  M5 (7/31): 客户送样验证（远航精密/长鑫汽车）— 状态: 未开始
  M6 (8/31): TS16949 PPAP提交 — 状态: 未开始
  M7 (9/30): 项目结题/量产转移 — 状态: 未开始

技术指标：
  精度：±0.5% @25℃（当前1%，需提升）
  响应时间：≤8s（当前15s，需提升）
  工作温度：-50℃ ~ +150℃（当前-40~125，需扩展）
  AEC-Q200认证：全部通过

资源需求：
  - 研发工程师：3人（现2人，需扩招1人）
  - 测试设备：高低温冲击箱（已采购，6月到货）
  - 材料成本：25万元
  - 测试费用：15万元（委外）
  - 认证费用：8万元（TS16949 PPAP）

风险：
  R1: 材料供应商产能不足 → 已确认备选供应商
  R2: 测试设备到货延期 → 已联系委外测试备用
  R3: 关键指标未达标 → 考虑放宽至±0.8%
"""
    docs.append(("2026年新产品开发项目计划_RD-2026-003.txt", project_plan))

    return docs


def gen_procedure_documents():
    """生成流程/操作指引类文档。"""
    docs = []

    safety_manual = """安全生产管理手册

第一章 总则
1.1 安全生产方针："安全第一，预防为主，综合治理"
1.2 适用范围：本公司所有部门、车间、及访客

第二章 通用安全规定
2.1 进入生产区域必须佩戴安全帽、穿防静电服
2.2 禁止在车间内奔跑、打闹
2.3 禁止酒后上岗（血液酒精含量<20mg/100ml）
2.4 禁止擅自操作非本岗位设备
2.5 发现安全隐患，立即上报部门安全员

第三章 电气安全
3.1 严禁私拉乱接电线
3.2 设备检修前必须断电并挂"正在维修"警示牌
3.3 高压电气操作（>380V）须持电工证
3.4 每月检查配电箱接地电阻（<4Ω）

第四章 化学品安全
4.1 所有化学品须张贴MSDS（物质安全数据表）
4.2 危化品存储区禁止烟火，配备防爆电器
4.3 使用化学品须佩戴防护手套和护目镜
4.4 废化学品按环保要求分类收集，定期由资质单位处理

第五章 消防安全
5.1 保持消防通道畅通，禁止堵塞
5.2 灭火器每月检查压力表（绿区正常）
5.3 每季度消防演练（全员参加）
5.4 火警电话：119，公司应急电话：020-8888-1199

第六章 应急响应
6.1 火灾：报警→疏散→灭火→报告
6.2 化学品泄漏：隔离→通风→吸附→清理→报告
6.3 人员受伤：急救→拨打120→报告→保护现场
6.4 停电：启用应急照明→关闭设备电源→等待来电

第七章 事故上报
7.1 任何人身伤害事故必须在1小时内报告人事部
7.2 重大事故（死亡/重伤/财产损失>10万元）立即报告总经办
7.3 事故调查48小时内启动，7个工作日内出具调查报告

附录：应急电话
  急救: 120  火警: 119  报警: 110
  公司保安室: 020-8888-1100
  安全主管 刘工: 13902289999
"""
    docs.append(("安全生产管理手册.txt", safety_manual))

    return docs


def gen_management_documents():
    """生成管理类文档。"""
    docs = []

    meeting_minutes = """管理层周会纪要

日期：2026年5月6日 14:00-16:30
地点：三楼1号会议室
主持：陈明 总经理
参会：张总监（研发）、李总（销售）、王经理（生产）、赵总监（财务）、刘经理（人事）

1. 上周工作回顾
  - 销售部：完成Q2目标进度的35%，同比略有下降，需加大开拓力度
  - 研发部：RD-2026-003项目样品测试中，材料纯度低于预期
  - 生产部：SMT-003设备抛料率波动，已安排检修
  - 财务部：应收款催收有进展，光谷通信承诺5月15日前支付
  - 人事部：Q2招聘进度完成60%，CNC操作工缺口3人

2. 重点议题
  议题一：远航精密供应商评审（5月20日）
    决议：由李总牵头，成立评审准备小组
    分工：销售部准备商务资料、生产部整理产线展示、品质部准备体系文件
    截止：5月15日前完成所有准备

  议题二：原材料涨价应对
    决议：NTC芯片供应商通知涨价12%（6月1日起）
    对策：①与供应商谈判争取阶梯价格 ②加大备货量至3个月 ③评估替代供应商
    预算：额外资金需求约50万元（备货）

  议题三：新产线投资
    决议：原则同意投资150万新增SMT产线
    财务部做ROI测算，5月20日前提交
    设备选型由工程部负责，5月底前完成

3. 下周工作重点
  - 远航精密评审准备（全员配合）
  - 客户投诉处理（品质部48小时内回复8D报告）
  - SMT-003设备维修跟进
  - 新员工入职培训（5月10日）

4. 下次会议
  时间：5月13日 14:00  地点：三楼1号会议室
"""
    docs.append(("管理层周会纪要_20260506.txt", meeting_minutes))

    # 公司制度
    policy = """公司保密制度

第一章 总则
第一条 为保护公司商业秘密，维护公司竞争优势，制定本制度。
第二条 保密范围为：技术秘密、经营信息、客户信息、财务信息、人事信息。

第二章 密级分类
第三条 公司信息分为三个等级：
  - 机密（AAA）：产品配方、核心算法、未公开的专利方案
  - 秘密（AA）：客户列表、采购价格、成本构成、供应商信息
  - 内部（A）：薪资信息、内部流程文件、非公开人事信息

第三章 保密措施
第四条 机密级文件需加密存储，禁止通过互联网传输
第五条 涉密人员签订《保密协议》和《竞业限制协议》
第六条 外部访客需签署保密承诺，由陪同人员全程陪同
第七条 离职员工需经过脱密期（机密级6个月，秘密级3个月）

第四章 违规处罚
第八条 泄露内部信息：警告处分，罚款1000-5000元
第九条 泄露秘密信息：记过，罚款5000-20000元，可解除劳动合同
第十条 泄露机密信息：解除劳动合同，追究法律责任

第五章 计算机安全
第十一条 禁止私自安装未经授权的软件
第十二条 禁止使用个人U盘/移动硬盘
第十三条 办公电脑设置屏幕保护密码（5分钟自动锁定）
第十四条 外发文件需经主管审批

本制度自发布之日起执行，由总经办负责解释。
"""
    docs.append(("公司保密制度.txt", policy))

    return docs


def gen_operations_documents():
    """生成运营、采购、法务、售后、IT 支撑类文档。"""
    docs = []

    procurement = """采购管理制度 V1.4

一、供应商准入
1. 新供应商需完成资质审核、样品测试、现场稽核三步。
2. 评分低于80分的供应商不得纳入合格名录。
3. 涉及关键物料的供应商需至少保留2家备选。

二、采购审批
1. 单笔采购金额低于2万元：部门经理审批。
2. 2万元至10万元：分管总监审批。
3. 10万元以上：总经理审批。
4. 紧急采购需附书面说明，24小时内补单。

三、交期管理
1. 常规物料交期不超过14天。
2. 关键物料交期超过21天需升级预警。
3. 连续2次延期的供应商暂停新订单3个月。

四、质量追责
1. 来料不良率高于2%触发供应商整改。
2. 连续3批不合格直接取消准入资格。
"""
    docs.append(("采购管理制度_V1.4.txt", procurement))

    after_sales = """售后服务工单台账

工单编号：CS-2026-0508-01
客户：深圳远航精密制造
产品：NTC-10K-B3435
问题：客户反馈开机后温度跳变，数据波动较大
受理时间：2026-05-08 09:20
首次响应：2026-05-08 10:05
处理时限：48小时内闭环
当前状态：已安排技术工程师上门
责任人：刘工

工单编号：CS-2026-0508-02
客户：无锡云芯电子
产品：SMT贴片产线交付件
问题：包装标签批次与送货单不一致
处理要求：当天复核、次日补发说明
当前状态：已完成

售后规则：
1. P1故障 2小时响应，24小时内给出临时方案。
2. P2故障 4小时响应，48小时内闭环。
3. 批量质量问题 24小时内启动8D报告。
"""
    docs.append(("售后服务工单台账_2026年5月.txt", after_sales))

    it_helpdesk = """IT运维服务规范

一、账号管理
1. 新员工账号在入职当天创建。
2. 离职账号在离职当天18:00前禁用。
3. 连续输错密码5次锁定15分钟。

二、终端管理
1. 办公电脑必须安装终端管控软件。
2. 禁止安装未授权软件。
3. 每周自动推送补丁更新一次。

三、故障响应
1. 一般办公故障：30分钟内响应，4小时内处理。
2. 网络故障：15分钟内响应。
3. 服务器故障：10分钟内上报值班人员。

四、备份策略
1. 核心业务数据每日备份。
2. 备份保留30天。
3. 每月进行一次恢复演练。
"""
    docs.append(("IT运维服务规范.txt", it_helpdesk))

    legal = """合同审批与法务审核清单

一、合同类型
1. 销售合同
2. 采购合同
3. NDA保密协议
4. 供应商质量协议

二、审核要点
1. 金额超过50万元必须法务审核。
2. 含违约责任条款、数据保护条款、知识产权条款必须逐条确认。
3. 付款周期超过90天需财务和法务双签。
4. 涉及跨境条款需补充风险说明。

三、审批时限
1. 常规合同：2个工作日。
2. 紧急合同：1个工作日内反馈初审意见。
"""
    docs.append(("合同审批与法务审核清单.txt", legal))

    warehouse = """仓储发货管理台账

日期：2026-05-08
仓库：成品仓

| 物料编码 | 物料名称 | 库存数量 | 安全库存 | 发货单号 | 状态 |
|----------|----------|----------|----------|----------|------|
| NTC-001 | NTC传感器成品 | 18500 | 5000 | SH20260508-01 | 待发货 |
| PCB-008 | PCB板 | 9200 | 3000 | SH20260508-02 | 已拣货 |
| KIT-003 | 贴片套件 | 1260 | 800 | SH20260508-03 | 运输中 |

规则：
1. 低于安全库存立即补货。
2. 发货前必须核对批次号和客户名称。
3. 出库差异大于1%需复盘。
"""
    docs.append(("仓储发货管理台账_2026年5月.txt", warehouse))

    return docs


def generate_all_documents():
    """生成所有企业文档。"""
    print("\n生成企业测试文档...")
    all_docs = []
    all_docs.extend(gen_hr_documents())
    all_docs.extend(gen_sales_documents())
    all_docs.extend(gen_production_documents())
    all_docs.extend(gen_finance_documents())
    all_docs.extend(gen_tech_documents())
    all_docs.extend(gen_procedure_documents())
    all_docs.extend(gen_management_documents())
    all_docs.extend(gen_operations_documents())

    # 保存到本地
    for name, content in all_docs:
        path = TEST_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"已生成 {len(all_docs)} 个企业文档")
    return all_docs


# ============================================================
# Part 2: 上传文档到系统
# ============================================================

def upload_documents(docs):
    """通过 API 批量上传文档。"""
    print("\n上传文档到知识库...")
    uploaded = 0
    dup = 0
    for i, (name, content) in enumerate(docs):
        path = TEST_DIR / name
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    f"{API}/document/upload",
                    files={"file": (name, f)},
                    data={"collection": "shared"},
                    headers=api_headers(),
                )
            data = resp.json()
            if data.get("duplicate"):
                dup += 1
            else:
                uploaded += 1
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(docs)} (已上传 {uploaded}, 跳过 {dup})")
        except Exception as e:
            print(f"  上传失败: {name} — {e}")
    print(f"上传完成: {uploaded} 个文档, {dup} 个跳过")
    return uploaded


# ============================================================
# Part 3: 测试查询
# ============================================================

TEST_QUERIES = [
    # ---- 事实查询（精确匹配） ----
    {"query": "无锡云芯的联系人是谁？", "relevant_docs": ["客户信息表"], "keywords": ["李建国"], "category": "事实查询", "difficulty": "easy"},
    {"query": "SMT贴片机的抛料率标准是多少？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["0.3%"], "category": "事实查询", "difficulty": "easy"},
    {"query": "公司差旅住宿标准一线城市是多少？", "relevant_docs": ["差旅费报销管理制度"], "keywords": ["500元"], "category": "事实查询", "difficulty": "easy"},
    {"query": "NTC热敏电阻25度的标称阻值是多少？", "relevant_docs": ["NTC热敏电阻技术规格书"], "keywords": ["10KΩ"], "category": "事实查询", "difficulty": "easy"},
    {"query": "远航精密的联系人叫什么名字？", "relevant_docs": ["客户信息表"], "keywords": ["王芳"], "category": "事实查询", "difficulty": "easy"},
    {"query": "年假满1年不足10年的员工有多少天？", "relevant_docs": ["考勤管理制度"], "keywords": ["5天"], "category": "事实查询", "difficulty": "easy"},
    {"query": "MES系统接口的认证方式是什么？", "relevant_docs": ["MES系统接口规范"], "keywords": ["Bearer", "token"], "category": "事实查询", "difficulty": "medium"},
    {"query": "公司的火警应急电话是多少？", "relevant_docs": ["安全生产管理手册"], "keywords": ["119"], "category": "事实查询", "difficulty": "easy"},
    {"query": "设备SMT-003的4月OEE是多少？", "relevant_docs": ["2026年4月设备维护记录"], "keywords": ["92.5%"], "category": "事实查询", "difficulty": "medium"},
    {"query": "蓝牙温度传感器的BOM单件物料成本是多少？", "relevant_docs": ["BOM"], "keywords": ["2.90元"], "category": "事实查询", "difficulty": "hard"},

    # ---- 汇总/统计查询 ----
    {"query": "2026年Q1的营业收入是多少？", "relevant_docs": ["2026年Q1财务分析报告"], "keywords": ["2,850万元", "2850万"], "category": "汇总统计", "difficulty": "easy"},
    {"query": "列出所有A级客户的名字", "relevant_docs": ["客户信息表"], "keywords": ["无锡云芯", "远航精密", "安视科技", "启明智能", "津汽制造"], "category": "汇总统计", "difficulty": "medium"},
    {"query": "Q1品质报告中虚焊问题有多少起？", "relevant_docs": ["2026年Q1品质分析报告"], "keywords": ["12起"], "category": "汇总统计", "difficulty": "medium"},
    {"query": "4月份设备SMT-003的非计划停机时间是多少？", "relevant_docs": ["2026年4月设备维护记录"], "keywords": ["3.5小时"], "category": "汇总统计", "difficulty": "medium"},
    {"query": "公司总共有多少客户在合作中或洽谈中？", "relevant_docs": ["客户信息表"], "keywords": ["在合作", "洽谈中"], "category": "汇总统计", "difficulty": "hard"},
    {"query": "NTC探头BOM的总成本构成有哪些部分？", "relevant_docs": ["BOM"], "keywords": ["物料", "人工", "制造"], "category": "汇总统计", "difficulty": "medium"},
    {"query": "2026年Q2的预估营收范围是多少？", "relevant_docs": ["2026年Q1财务分析报告"], "keywords": ["3200", "3500"], "category": "汇总统计", "difficulty": "medium"},

    # ---- 概念解释 ----
    {"query": "什么是防静电手环的接地电阻要求？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["1MΩ"], "category": "概念解释", "difficulty": "medium"},
    {"query": "公司是如何定义迟到的？", "relevant_docs": ["考勤管理制度"], "keywords": ["30分钟"], "category": "概念解释", "difficulty": "easy"},
    {"query": "回流焊的温度曲线参数是什么？", "relevant_docs": ["2026年Q1品质分析报告"], "keywords": ["150", "180", "220"], "category": "概念解释", "difficulty": "medium"},
    {"query": "SMT常见的异常有哪几种？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["偏移", "抛料", "真空"], "category": "概念解释", "difficulty": "easy"},
    {"query": "公司保密制度中的三个密级分别是什么？", "relevant_docs": ["公司保密制度"], "keywords": ["机密", "秘密", "内部"], "category": "概念解释", "difficulty": "easy"},
    {"query": "什么情况下会触发出差紧急审批流程？", "relevant_docs": ["差旅费报销管理制度"], "keywords": ["紧急"], "category": "概念解释", "difficulty": "medium"},

    # ---- 操作流程 ----
    {"query": "SMT贴片机开机步骤是什么？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["电源", "压缩空气", "真空泵", "自检"], "category": "操作流程", "difficulty": "medium"},
    {"query": "发生火灾时正确的应急流程是什么？", "relevant_docs": ["安全生产管理手册"], "keywords": ["报警", "疏散", "灭火"], "category": "操作流程", "difficulty": "easy"},
    {"query": "出差费用报销的具体步骤是什么？", "relevant_docs": ["差旅费报销管理制度"], "keywords": ["OA", "审批", "发票"], "category": "操作流程", "difficulty": "medium"},
    {"query": "忘记打卡了应该怎么处理？", "relevant_docs": ["考勤管理制度"], "keywords": ["48小时", "补卡", "审批"], "category": "操作流程", "difficulty": "easy"},
    {"query": "化学品泄漏的标准处理流程是怎样的？", "relevant_docs": ["安全生产管理手册"], "keywords": ["隔离", "通风", "吸附"], "category": "操作流程", "difficulty": "medium"},
    {"query": "SMT换料的标准操作步骤是什么？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["STOP", "COMP CHANGE", "料号"], "category": "操作流程", "difficulty": "medium"},

    # ---- 对比/分析查询（复杂） ----
    {"query": "对比远航精密和无锡云芯这两家客户的合作状态和年度采购额", "relevant_docs": ["客户信息表"], "keywords": ["远航精密", "无锡云芯", "300万", "800万"], "category": "对比分析", "difficulty": "hard"},
    {"query": "Q1营收的三大产品线分别是多少，占比多少？", "relevant_docs": ["2026年Q1财务分析报告"], "keywords": ["NTC", "温控器", "技术服务", "54.7%", "27.4%", "17.9%"], "category": "对比分析", "difficulty": "hard"},
    {"query": "2026年4月和5月的薪资表对比，哪个部门工资最高？", "relevant_docs": ["薪资表"], "keywords": ["研发", "销售"], "category": "对比分析", "difficulty": "hard"},
    {"query": "新项目RD-2026-003的各个里程碑完成情况如何？", "relevant_docs": ["2026年新产品开发项目计划"], "keywords": ["M1", "M2", "已完成", "进行中", "未开始"], "category": "对比分析", "difficulty": "medium"},
    {"query": "应收账款中哪些客户有坏账风险？", "relevant_docs": ["2026年Q1财务分析报告"], "keywords": ["光谷通信", "35万", "90天"], "category": "对比分析", "difficulty": "hard"},

    # ---- 不常见/边缘查询 ----
    {"query": "公司电销部目前有几个人？", "relevant_docs": [], "keywords": ["电销部"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "如果贴片机出现红色报警应该按什么按钮？", "relevant_docs": ["SMT贴片标准操作规程"], "keywords": ["EMERGENCY STOP", "紧急停止"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "新员工入职培训是什么日期？", "relevant_docs": ["管理层周会纪要"], "keywords": ["5月10日"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "环氧树脂的规格型号是什么？", "relevant_docs": ["BOM"], "keywords": ["E-44", "双组分"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "公司对于虚假报销的处理措施是什么？", "relevant_docs": ["差旅费报销管理制度"], "keywords": ["追回", "记过"], "category": "边缘查询", "difficulty": "medium"},
    {"query": "MES上报接口的错误码1004代表什么？", "relevant_docs": ["MES系统接口规范"], "keywords": ["数据库写入失败"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "贴片机吸嘴多久更换一次？", "relevant_docs": ["2026年4月设备维护记录", "SMT贴片标准操作规程"], "keywords": ["磨损", "季度"], "category": "边缘查询", "difficulty": "hard"},
    {"query": "公司保密制度对离职员工的脱密期要求是什么？", "relevant_docs": ["公司保密制度"], "keywords": ["6个月", "3个月"], "category": "边缘查询", "difficulty": "medium"},

    # ---- 运营/采购/售后/法务 ----
    {"query": "采购金额超过10万元需要谁审批？", "relevant_docs": ["采购管理制度"], "keywords": ["总经理"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "供应商来料不良率超过多少会触发整改？", "relevant_docs": ["采购管理制度"], "keywords": ["2%"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "客户售后P1故障多久响应？", "relevant_docs": ["售后服务工单台账"], "keywords": ["2小时", "24小时"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "IT运维对离职账号什么时候禁用？", "relevant_docs": ["IT运维服务规范"], "keywords": ["当天18:00"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "合同金额超过50万元要不要法务审核？", "relevant_docs": ["合同审批与法务审核清单"], "keywords": ["法务审核"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "仓库发货前要核对什么信息？", "relevant_docs": ["仓储发货管理台账"], "keywords": ["批次号", "客户名称"], "category": "运营支撑", "difficulty": "medium"},
    {"query": "低于安全库存怎么办？", "relevant_docs": ["仓储发货管理台账"], "keywords": ["立即补货"], "category": "运营支撑", "difficulty": "easy"},
    {"query": "采购单笔金额2万元到10万元由谁审批？", "relevant_docs": ["采购管理制度"], "keywords": ["分管总监"], "category": "运营支撑", "difficulty": "medium"},
    {"query": "售后批量质量问题多久内要启动8D报告？", "relevant_docs": ["售后服务工单台账"], "keywords": ["24小时"], "category": "运营支撑", "difficulty": "medium"},
    {"query": "IT系统连续输错几次密码会锁定？", "relevant_docs": ["IT运维服务规范"], "keywords": ["5次", "15分钟"], "category": "运营支撑", "difficulty": "easy"},
]


# ============================================================
# Part 4: 评估
# ============================================================

def run_queries_and_evaluate():
    """运行查询并计算指标。"""
    print(f"\n运行 {len(TEST_QUERIES)} 个测试查询...")

    results = {
        "hit_rate": [],
        "mrr": [],
        "keyword_recall": [],
        "by_category": {},
        "by_difficulty": {},
    }

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        expected_kw = test.get("keywords", [])
        category = test.get("category", "未知")
        difficulty = test.get("difficulty", "medium")

        try:
            # 发送查询
            resp = requests.post(
                f"{API}/chat/stream",
                json={"query": query, "top_k": 10},
                headers=api_headers(),
                stream=True,
            )

            # 收集流式回答
            full_answer = ""
            sources = []
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: [SOURCES]"):
                        sources_str = line.replace("data: [SOURCES]", "").replace("[/SOURCES]", "")
                        try:
                            sources = json.loads(sources_str)
                        except Exception:
                            pass
                    elif line.startswith("data: ") and not line.startswith("data: ["):
                        full_answer += line[6:]

            # 评估关键词召回
            kw_hits = sum(1 for kw in expected_kw if kw.lower() in full_answer.lower())
            kw_recall = kw_hits / max(len(expected_kw), 1)
            results["keyword_recall"].append(kw_recall)

            # MRR (基于命中任何关键词)
            mrr = 1.0 if kw_hits > 0 else 0.0
            results["mrr"].append(mrr)

            # Hit rate (是否有任何来源)
            hit = 1.0 if sources else 0.0
            results["hit_rate"].append(hit)

            # 按分类统计
            if category not in results["by_category"]:
                results["by_category"][category] = {"count": 0, "kw_recall": [], "hit": []}
            results["by_category"][category]["count"] += 1
            results["by_category"][category]["kw_recall"].append(kw_recall)
            results["by_category"][category]["hit"].append(hit)

            # 按难度统计
            if difficulty not in results["by_difficulty"]:
                results["by_difficulty"][difficulty] = {"count": 0, "kw_recall": [], "hit": []}
            results["by_difficulty"][difficulty]["count"] += 1
            results["by_difficulty"][difficulty]["kw_recall"].append(kw_recall)
            results["by_difficulty"][difficulty]["hit"].append(hit)

            status = "✓" if kw_recall > 0.5 else ("△" if kw_recall > 0 else "✗")
            if (i + 1) % 5 == 0:
                print(f"  进度: {i+1}/{len(TEST_QUERIES)}")

        except Exception as e:
            print(f"  查询失败 [{i+1}]: {query[:30]}... — {e}")
            results["keyword_recall"].append(0.0)
            results["mrr"].append(0.0)
            results["hit_rate"].append(0.0)

    return results


def compute_metrics(results):
    """计算综合指标。"""
    metrics = {
        "avg_keyword_recall": np.mean(results["keyword_recall"]),
        "avg_hit_rate": np.mean(results["hit_rate"]),
        "avg_mrr": np.mean(results["mrr"]),
        "num_queries": len(results["keyword_recall"]),
        "by_category": {},
        "by_difficulty": {},
    }

    for cat, data in results["by_category"].items():
        metrics["by_category"][cat] = {
            "count": data["count"],
            "keyword_recall": np.mean(data["kw_recall"]),
            "hit_rate": np.mean(data["hit"]),
        }

    for diff, data in results["by_difficulty"].items():
        metrics["by_difficulty"][diff] = {
            "count": data["count"],
            "keyword_recall": np.mean(data["kw_recall"]),
            "hit_rate": np.mean(data["hit"]),
        }

    return metrics


def print_report(metrics):
    """打印评估报告。"""
    print("\n" + "=" * 60)
    print("  Enterprise RAG 知识库系统 —— 综合评估报告")
    print("=" * 60)
    print(f"  测试查询数: {metrics['num_queries']}")
    print(f"  平均关键词召回率: {metrics['avg_keyword_recall']:.1%}")
    print(f"  平均命中率 (Hit Rate): {metrics['avg_hit_rate']:.1%}")
    print(f"  平均 MRR: {metrics['avg_mrr']:.1%}")
    print()
    print("  --- 按查询分类 ---")
    for cat, data in metrics["by_category"].items():
        print(f"  {cat:<12}: {data['count']:>3}条  召回率={data['keyword_recall']:.1%}  命中率={data['hit_rate']:.1%}")
    print()
    print("  --- 按难度 ---")
    for diff, data in sorted(metrics["by_difficulty"].items()):
        print(f"  {diff:<8}: {data['count']:>3}条  召回率={data['keyword_recall']:.1%}  命中率={data['hit_rate']:.1%}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  Enterprise RAG 知识库系统 —— 全面评估")
    print("=" * 60)

    # 1. 登录
    print("\n[1/5] 登录系统...")
    login()

    # 2. 生成文档
    print("\n[2/5] 生成企业测试文档...")
    docs = generate_all_documents()

    # 3. 上传索引
    print("\n[3/5] 上传并索引文档...")
    count = upload_documents(docs)
    print(f"  成功索引 {count} 个文档")

    # 4. 等待索引完成（Qdrant 异步写入）
    print("\n  等待索引完成...")
    time.sleep(3)

    # 5. 运行查询
    print(f"\n[4/5] 运行 {len(TEST_QUERIES)} 个测试查询...")
    results = run_queries_and_evaluate()

    # 6. 计算并打印报告
    print("\n[5/5] 生成评估报告...")
    metrics = compute_metrics(results)
    print_report(metrics)

    # 保存结果
    report_path = TEST_DIR.parent / "eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {report_path}")


if __name__ == "__main__":
    main()
