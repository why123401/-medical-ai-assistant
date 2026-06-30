#!/usr/bin/env python
"""Generate 200+ golden QA pairs for medical device evaluation.

Coverage matrix:
  Devices: 6 (Ventilator, CT, MRI, Monitor, Ultrasound, Autoclave)
  Categories: 4 (specification, fault_diagnosis, maintenance, safety)
  Question styles: 5+ (direct, scenario, comparison, troubleshooting, "what if")

Usage:
    python scripts/generate_golden_set.py

Output: data/eval/qa_golden_set.jsonl (200+ pairs)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Device knowledge base — ground truth for all QA generation
# ============================================================

DEVICES = {
    "MED-VENT-X200": {
        "name": "呼吸机",
        "category": "生命支持",
        "specs": {
            "氧浓度范围": "21% - 100%（步进 1%）",
            "潮气量范围": "5 - 800 ml（步进 1 ml）",
            "呼吸频率": "4 - 60 次/分",
            "吸气时间": "0.1 - 2.0 秒",
            "PEEP 范围": "0 - 25 cmH2O",
            "触发灵敏度": "0.1 - 20 L/min 或 0.01 - 2.0 L",
            "内置电池续航": "≥ 4 小时",
            "尺寸": "420mm × 320mm × 180mm",
            "重量": "6.5 kg",
            "适用人群": "成人/儿童/新生儿",
            "电源": "AC 100-240V, 50/60Hz",
        },
        "modes": ["A/C (SIMV)", "SIMV", "PSV", "CPAP", "PCV", "VCVC"],
        "faults": {
            "E101": {"name": "气源压力不足", "severity": "warning",
                     "causes": ["压缩空气气源压力低于 400 kPa", "气源管路过滤器堵塞", "压力传感器故障"],
                     "procedures": ["检查中央供气压力表", "更换气源管路过滤器", "校准压力传感器"]},
            "E102": {"name": "氧气供应中断", "severity": "critical",
                     "causes": ["中央供氧管路断开", "氧气钢瓶耗尽"],
                     "procedures": ["检查供氧管路连接", "更换氧气钢瓶", "切换至备用气源"]},
            "E103": {"name": "管路泄漏", "severity": "warning",
                     "causes": ["呼气阀滤芯破损", "气管导管气囊压力不足", "回路接头未拧紧"],
                     "procedures": ["检查所有管路接头", "测量气囊压力（应 25-30 cmH2O）", "更换呼气阀滤芯"]},
            "E104": {"name": "氧浓度过低", "severity": "critical",
                     "causes": ["空气/氧气比例阀故障", "管路泄漏", "氧传感器故障", "气源供氧压力不足", "氧浓度混合室堵塞"],
                     "procedures": ["检查管路连接", "更换备用管路", "检查中央供氧压力", "校准氧传感器", "更换氧传感器模组", "更换比例阀"]},
            "E105": {"name": "呼气阀卡滞", "severity": "warning",
                     "causes": ["呼气阀阀门积碳", "弹簧疲劳"],
                     "procedures": ["清洁呼气阀", "更换弹簧", "更换呼气阀模组"]},
            "E201": {"name": "患者回路压力传感器故障", "severity": "critical",
                     "causes": ["传感器漂移", "连接线松动", "主板 ADC 模块故障"],
                     "procedures": ["执行传感器零点校准", "检查传感器连接线", "更换传感器模组"]},
            "E202": {"name": "流量传感器校准失效", "severity": "warning",
                     "causes": ["传感器污染", "温度漂移"],
                     "procedures": ["清洁流量传感器", "执行温度补偿校准"]},
            "E301": {"name": "内置电池电量低", "severity": "warning",
                     "causes": ["电池老化", "未充电"],
                     "procedures": ["连接市电", "更换电池模组"]},
        },
        "maintenance": {
            "daily": ["开机自检确认报警功能正常", "检查管路无裂纹和松动", "清洗湿化器", "氧浓度检测", "电池测试（续航≥4小时）"],
            "weekly": ["更换一次性管路和过滤器", "清洁设备外壳和显示屏", "检查气源管路连接", "测试所有报警功能", "备份患者数据和配置参数"],
            "monthly": ["更换氧电池", "校准潮气量传感器", "检查呼气阀功能", "清洁空气过滤器", "执行全参数校准"],
            "yearly": ["全面功能测试", "更换所有一次性耗材", "气路密封性测试", "电气安全测试", "软件版本检查和升级"],
        },
        "scenes": ["ICU 重症监护", "手术室麻醉通气", "急诊抢救", "转运途中生命支持", "家庭长期机械通气"],
    },
    "MED-CT-3200": {
        "name": "CT 扫描仪",
        "category": "影像诊断",
        "specs": {
            "探测器排数": "128 排",
            "层厚": "0.5 mm - 5.0 mm（可调）",
            "扫描时间": "0.35 秒/圈",
            "球管热容量": "3.5 MHU",
            "球管散热率": "2.8 MU/min",
            "床面承重": "230 kg",
            "扫描野": "60 cm",
            "辐射剂量特点": "低剂量模式可达常规剂量的 30%",
            "电源": "三相 AC 380V, 50Hz, ≥ 30 kW",
            "环境温度要求": "18°C - 26°C",
            "环境湿度要求": "30% - 70%（无凝结）",
        },
        "modes": ["常规螺旋扫描", "心电门控扫描", "灌注成像", "双能成像", "MIP", "VR", "仿真内镜"],
        "faults": {
            "A101": {"name": "球管预热超时", "severity": "warning",
                     "causes": ["球管老化", "预热电路故障"],
                     "procedures": ["等待球管自然冷却后重试", "检查预热电路", "联系售后更换球管"]},
            "A102": {"name": "球管温度过高", "severity": "warning",
                     "causes": ["连续扫描次数过多", "散热风扇故障", "冷却风道堵塞"],
                     "procedures": ["暂停扫描等待降温", "检查散热风扇", "清理冷却风道"]},
            "A201": {"name": "高压发生器故障", "severity": "critical",
                     "causes": ["高压电缆老化", "绝缘击穿"],
                     "procedures": ["检查高压电缆连接", "更换高压电缆", "联系售后检修"]},
            "A202": {"name": "冷却系统水温异常", "severity": "critical",
                     "causes": ["水温过高或过低", "冷却泵故障"],
                     "procedures": ["检查冷却水温度（正常 18-22°C）", "重启冷却系统", "更换冷却泵"]},
            "A203": {"name": "冷却系统水流不足", "severity": "critical",
                     "causes": ["水管堵塞", "水泵故障"],
                     "procedures": ["检查水管通畅性", "更换过滤器", "更换水泵"]},
            "F201": {"name": "探测器温度异常", "severity": "critical",
                     "causes": ["冷却系统故障", "探测器模块老化", "环境温度过高", "温度传感器故障"],
                     "procedures": ["检查机房温度", "检查冷却系统水位和水温", "重启冷却系统", "更换温度传感器"]},
            "F202": {"name": "数据采集系统通信中断", "severity": "critical",
                     "causes": ["DAQS 连接松动", "DAQS 板卡故障", "探测器高压供电异常"],
                     "procedures": ["关机重启", "检查 DAQS 连接", "更换 DAQS 板卡"]},
        },
        "maintenance": {
            "daily": ["水校准", "球管日校准", "检查冷却系统", "清洁床面", "查看扫描日志"],
            "weekly": ["空气校准", "检查散热曲线", "清洁空调滤网", "检查高压电缆", "测试急停按钮"],
            "monthly": ["更换冷却过滤器", "检查球管寿命", "探测器校准", "床面精度检查", "备份配置"],
            "yearly": ["更换制冷剂和冷却水", "球管全面检测", "高压发生器校准", "辐射剂量检测", "机械部件润滑", "软件升级"],
        },
    },
    "MED-MRI-1.5T": {
        "name": "磁共振成像系统",
        "category": "影像诊断",
        "specs": {
            "场强": "1.5T",
            "磁体类型": "超导",
            "孔径": "70 cm",
            "梯度场强": "40 mT/m",
            "梯度切换率": "200 T/m/s",
            "接收线圈通道数": "32 通道",
            "最大病人重量": "227 kg",
            "电源": "三相 AC 380V",
        },
        "faults": {
            "M001": {"name": "超导磁体失超", "severity": "critical",
                     "causes": ["磁体温度异常升高", "外部磁场干扰"],
                     "procedures": ["立即疏散人员", "打开放空阀", "联系厂家更换液氦"]},
            "M002": {"name": "梯度线圈过热", "severity": "warning",
                     "causes": ["冷却水流量不足", "梯度放大器故障"],
                     "procedures": ["检查冷却水系统", "重启梯度放大器"]},
        },
        "maintenance": {
            "daily": ["检查液氦液位", "开机匀场", "运行 phantom 校准"],
            "weekly": ["检查冷却水质量", "清洁空调系统", "检查梯度线圈"],
            "monthly": ["更换冷却水", "全面 phantom 测试", "检查射频线圈"],
            "yearly": ["超导磁体检漏", "梯度系统校准", "射频系统检测", "安全认证年检"],
        },
    },
    "MED-MON-5000": {
        "name": "多参数病人监护仪",
        "category": "监测",
        "specs": {
            "监测参数": "心电、血氧、血压、呼吸、体温",
            "电池续航": "8 小时",
            "显示屏": "12.1 英寸 TFT LCD",
            "波形刷新率": "≥ 200 Hz",
            "NIBP 测量范围": "0 - 299 mmHg",
            "SpO2 测量范围": "0% - 100%",
            "重量": "2.8 kg",
        },
        "faults": {
            "E001": {"name": "导联线断路", "severity": "warning",
                     "causes": ["插头松动", "导线内部断裂"],
                     "procedures": ["检查插头连接", "更换导联线"]},
            "E002": {"name": "SpO2 信号弱", "severity": "warning",
                     "causes": ["探头位置不当", "末梢循环差", "指甲油干扰"],
                     "procedures": ["重新放置探头", "更换测量部位", "清除指甲油"]},
            "E003": {"name": "NIBP 测量失败", "severity": "warning",
                     "causes": ["袖带尺寸不合适", "管路漏气", "患者频繁移动"],
                     "procedures": ["更换合适尺寸袖带", "检查管路", "等待患者静止后重试"]},
        },
        "maintenance": {
            "daily": ["检查导联线", "清洁设备表面", "测试电池"],
            "weekly": ["校准血压模块", "检查 SpO2 探头", "更换袖带"],
            "monthly": ["校用心电模块", "校准 NIBP 模块", "校准 SpO2 模块", "更换电池（如续航<4小时）"],
            "yearly": ["全面电气安全检测", "所有参数精度校准", "显示屏亮度校准", "扬声器测试"],
        },
    },
    "MED-US-PRO": {
        "name": "超声诊断仪",
        "category": "影像诊断",
        "specs": {
            "频段": "2 - 15 MHz",
            "探头数": "最多 8 个",
            "成像模式": "B/M/D/彩色多普勒/弹性成像",
            "显示屏": "19 英寸液晶显示器",
            "最大穿透深度": "38 cm",
            "重量": "18 kg",
        },
        "faults": {
            "U001": {"name": "探头接触不良", "severity": "warning",
                     "causes": ["探头线缆断裂", "连接器氧化"],
                     "procedures": ["检查探头线缆", "清洁连接器", "更换探头"]},
            "U002": {"name": "图像噪波过多", "severity": "warning",
                     "causes": ["增益设置不当", "探头频率不匹配", "主板故障"],
                     "procedures": ["调整增益和深度", "更换合适频率探头", "联系售后检测主板"]},
        },
        "maintenance": {
            "daily": ["清洁探头和机身", "检查探头完整性", "运行自检程序"],
            "weekly": ["校准图像分辨率", "检查所有探头", "清洁散热风扇"],
            "monthly": ["全面图像质量测试", "校准多普勒频率", "检查机械臂"],
            "yearly": ["声学性能检测", "电气安全测试", "软件升级", "探头深度校准"],
        },
    },
    "MED-INF-500": {
        "name": "高压灭菌器",
        "category": "消毒",
        "specs": {
            "容量": "50 L",
            "灭菌温度": "121°C / 134°C 两档",
            "121°C 循环时间": "45 分钟",
            "134°C 循环时间": "18 分钟",
            "工作压力": "≤ 0.25 MPa",
            "电源": "AC 220V, 50Hz",
            "材质": "304 不锈钢",
        },
        "faults": {
            "E100": {"name": "升温失败", "severity": "critical",
                     "causes": ["加热管故障", "温度传感器故障", "电源异常"],
                     "procedures": ["检查加热管", "更换温度传感器", "检查电源"]},
            "E200": {"name": "门无法开启", "severity": "critical",
                     "causes": ["腔内仍有压力", "温度过高", "安全联锁激活"],
                     "procedures": ["确认压力和温度降至安全值", "使用紧急释放手柄", "联系售后"]},
            "E300": {"name": "压力异常", "severity": "warning",
                     "causes": ["安全阀故障", "压力表损坏", "密封圈老化"],
                     "procedures": ["检查安全阀", "更换压力表", "更换密封圈"]},
        },
        "maintenance": {
            "daily": ["清空灭菌腔", "检查密封圈", "添加灭菌用水", "运行空载循环"],
            "weekly": ["清洁排水过滤器", "检查安全阀", "校准温度", "更换密封圈（如使用>6个月）"],
            "monthly": ["全面清洁灭菌腔", "检查门铰链", "测试安全联锁", "生物指示剂测试"],
            "yearly": ["压力容器年检", "更换所有易损件", "安全阀强制检定", "电气系统全面检测"],
        },
    },
}

# ============================================================
# Question templates — each generates multiple variations
# ============================================================

TEMPLATES = [
    # --- Specification questions ---
    {
        "category": "specification",
        "patterns": [
            "{device_name} 的{param}是多少？",
            "{device_code} 的{param}参数是什么？",
            "{device_name}{param}的范围/规格是什么？",
            "{device_code} 支持的最大{param}是多少？",
            "{device_code} 的{param}技术指标？",
        ],
        "param_map": "specs",
    },
    {
        "category": "specification",
        "patterns": [
            "{device_name} 支持哪些通气/成像/监测模式？",
            "{device_code} 有哪些功能模式可选？",
            "{device_name} 的{mode_list}包括哪些？",
        ],
        "param_map": "modes",
    },
    {
        "category": "specification",
        "patterns": [
            "{device_name} 适用于哪些场景？",
            "{device_code} 的应用场景有哪些？",
            "{device_name} 可以在哪些场合使用？",
        ],
        "param_map": "scenes",
    },

    # --- Fault diagnosis questions ---
    {
        "category": "fault_diagnosis",
        "patterns": [
            "{device_name} 报警代码{fault_code}（{fault_name}）怎么处理？",
            "{device_code} 出现{fault_code}故障该怎么办？",
            "{device_name}{fault_code}报警的可能原因和处理步骤？",
            "{device_name}{fault_code}是什么意思？如何排除？",
        ],
        "param_map": "faults",
    },
    {
        "category": "fault_diagnosis",
        "patterns": [
            "{device_name} 的{fault_name}可能由什么原因引起？",
            "什么情况下{device_name}会报{fault_code}？",
            "{device_name}{fault_code}的故障原因？",
        ],
        "param_map": "faults",
    },

    # --- Maintenance questions ---
    {
        "category": "maintenance",
        "patterns": [
            "{device_name} 的{interval}维护需要做哪些工作？",
            "{device_code} 的{interval}保养清单？",
            "{device_name}{interval}维护项目有哪些？",
        ],
        "param_map": "maintenance",
    },

    # --- Scenario questions ---
    {
        "category": "fault_diagnosis",
        "patterns": [
            "{device_name} 在使用过程中出现{fault_name}，应该怎么处理？",
            "如果{device_code}报了{fault_code}，第一步该做什么？",
            "{device_name}{fault_code}报警时能否继续使用？",
        ],
        "param_map": "faults",
    },
]


def generate_qa_pairs() -> list[dict]:
    """Generate the full golden QA dataset."""
    pairs = []
    pair_id = 0

    for device_code, device in DEVICES.items():
        device_name = device["name"]

        # --- Specs ---
        for param, value in device["specs"].items():
            pair_id += 1
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {device_code} 的{param}是多少？",
                "expected_answer": f"{device_name}（{device_code}）的{param}为：{value}",
                "relevant_contexts": [f"{device_code}_specs#{param}"],
                "device_code": device_code,
                "category": "specification",
            })

        # --- Modes ---
        if "modes" in device:
            pair_id += 1
            mode_str = "、".join(device["modes"][:5])
            if len(device["modes"]) > 5:
                mode_str += f"等共{len(device['modes'])}种"
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {device_code} 支持哪些模式？",
                "expected_answer": f"{device_name}支持的模式包括：{mode_str}",
                "relevant_contexts": [f"{device_code}_specs#modes"],
                "device_code": device_code,
                "category": "specification",
            })

        # --- Scenes ---
        if "scenes" in device:
            pair_id += 1
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {device_code} 适用于哪些场景？",
                "expected_answer": f"{device_name}适用于：{ '、'.join(device['scenes'])}",
                "relevant_contexts": [f"{device_code}_specs#scenes"],
                "device_code": device_code,
                "category": "specification",
            })

        # --- Faults ---
        for fault_code, fault in device.get("faults", {}).items():
            pair_id += 1
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {device_code} 报警{fault_code}（{fault['name']}）怎么处理？",
                "expected_answer": f"{fault_code}表示{fault['name']}（严重程度:{fault['severity']}）。可能原因：{'、'.join(fault['causes'])}。处理步骤：{'；'.join(fault['procedures'])}",
                "relevant_contexts": [f"{device_code}_faults#{fault_code}"],
                "device_code": device_code,
                "category": "fault_diagnosis",
            })

            # Cause-focused question
            pair_id += 1
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {fault_code}故障的可能原因有哪些？",
                "expected_answer": f"{fault_code}（{fault['name']}）的可能原因包括：{'、'.join(fault['causes'])}",
                "relevant_contexts": [f"{device_code}_faults#{fault_code}"],
                "device_code": device_code,
                "category": "fault_diagnosis",
            })

        # --- Maintenance ---
        for interval, tasks in device.get("maintenance", {}).items():
            pair_id += 1
            pairs.append({
                "id": f"qa-{pair_id:04d}",
                "question": f"{device_name} {device_code} 的{interval}维护需要做哪些工作？",
                "expected_answer": f"{device_name}{interval}维护项目：{'；'.join(tasks)}",
                "relevant_contexts": [f"{device_code}_maintenance#{interval}"],
                "device_code": device_code,
                "category": "maintenance",
            })

    # --- Cross-device comparison questions ---
    pair_id += 1
    pairs.append({
        "id": f"qa-{pair_id:04d}",
        "question": "呼吸机和CT扫描仪的维护周期有什么区别？",
        "expected_answer": "呼吸机和维护以管路、氧电池等耗材更换为主，CT以水校准和球管散热为主。两者都有日/周/月/年的分级维护计划，但具体项目不同。",
        "relevant_contexts": ["maintenance_schedule.txt#呼吸机", "maintenance_schedule.txt#CT"],
        "device_code": "MULTIPLE",
        "category": "comparison",
    })

    # --- Additional scenario and edge-case questions to reach 200+ ---
    scenario_questions = [
        # Ventilator scenarios
        ("MED-VENT-X200", "呼吸机在转运病人途中突然报警 E104，现场没有备用管路怎么办？",
         "E104 是氧浓度过低报警。转运途中无备用管路时：1.立即切换至携带式简易呼吸囊（Ambu bag）手动通气 2.检查气管导管气囊压力 3.尽快到达 ICU 后更换管路和氧传感器"),
        ("MED-VENT-X200", "新生儿使用 MED-VENT-X200 时潮气量应该设多少？",
         "新生儿潮气量一般设为 5-10 ml/kg。MED-VENT-X200 最小潮气量为 5 ml，步进 1 ml，可以满足新生儿精确调节需求"),
        ("MED-VENT-X200", "MED-VENT-X200 断电后能坚持多久？",
         "内置电池可支持连续工作 ≥ 4 小时。断电后设备会自动切换到电池供电并报警提示"),

        # CT scenarios
        ("MED-CT-3200", "CT 扫描过程中突然出现 F201 报警，正在扫描的患者怎么办？",
         "F201 是探测器温度异常，系统会自动暂停扫描。1.先将患者移出扫描床 2.检查机房温度和冷却系统 3.重启冷却系统 4.温度恢复正常后重新扫描"),
        ("MED-CT-3200", "机房温度超过 26°C 对 CT 扫描仪有什么影响？",
         "机房温度超过 26°C 可能导致 F201 探测器温度异常报警，影响扫描质量甚至停机。应保持机房温度在 18-26°C"),
        ("MED-CT-3200", "MED-CT-3200 可以做心脏冠脉成像吗？",
         "可以。MED-CT-3200 支持心电门控扫描模式，结合 128 排探测器和 0.35 秒/圈的扫描速度，可以进行心脏冠脉 CTA 检查"),

        # MRI scenarios
        ("MED-MRI-1.5T", "磁共振失超（M001）有多危险？",
         "超导磁体失超是最严重的 MRI 事故。液氦瞬间气化可能导致窒息风险。处理：1.立即疏散所有人员 2.打开放空阀 3.确保通风系统运行 4.联系厂家更换液氦"),
        ("MED-MRI-1.5T", "1.5T 磁共振和 3.0T 有什么区别？",
         "1.5T 场强适中，成像速度较快，对金属植入物患者更安全（伪影较少）。3.0T 分辨率更高但扫描时间更长，有金属植入物的患者不能进入"),

        # Monitor scenarios
        ("MED-MON-5000", "监护仪 SpO2 读数一直偏低但患者看起来很正常怎么办？",
         "可能是探头问题。1.检查探头位置是否正确 2.更换测量部位（如左手换右手）3.清除指甲油 4.如仍异常更换 SpO2 探头"),
        ("MED-MON-5000", "监护仪电池续航不足 4 小时是什么原因？",
         "可能原因：1.电池老化（正常使用 2-3 年后衰减）2.屏幕亮度设置过高 3.开启了不必要的监测模块。建议更换电池模组"),

        # Ultrasound scenarios
        ("MED-US-PRO", "超声探头接触不良会出现什么问题？",
         "U001 探头接触不良会导致图像闪烁或无图像。处理：1.检查探头线缆是否完好 2.清洁连接器 3.重新插紧探头 4.必要时更换探头"),
        ("MED-US-PRO", "超声诊断仪可以做胎儿检查吗？",
         "可以。MED-US-PRO 支持 B/M/D/彩色多普勒模式，适合产科检查。2-15MHz 频段覆盖了经腹和经阴道探头的需求"),

        # Autoclave scenarios
        ("MED-INF-500", "高压灭菌器灭菌后器械还是湿的怎么回事？",
         "可能原因：1.干燥时间设置不足 2.灭菌袋包装过大 3.冷却阶段进风过快。处理：延长干燥时间，检查门密封圈，调整冷却风速"),
        ("MED-INF-500", "高压灭菌器可以灭菌液体（如生理盐水）吗？",
         "可以，但需要使用专门的液体灭菌程序，温度设为 121°C，时间适当延长，并且不能使用快速灭菌模式（134°C）"),

        # Maintenance deep-dive
        ("MED-VENT-X200", "呼吸机的氧电池多久更换一次？",
         "建议每月更换一次氧电池（O2 Sensor），或在以下情况提前更换：氧浓度校准失败、报警频繁、使用频率极高（ICU 全天候使用）"),
        ("MED-CT-3200", "CT 球管的寿命一般是多久？",
         "CT 球管寿命取决于使用频率和散热条件，一般为 2-5 年或 3000-10000 次扫描。MED-CT-3200 球管热容量 3.5MHU，散热率 2.8MU/min，属于中高端水平"),
        ("MED-INF-500", "高压灭菌器的密封圈多久更换一次？",
         "建议每 6 个月更换一次密封圈，或使用次数超过 1000 次后更换。如发现灭菌泄漏或压力无法维持，应立即更换"),

        # Safety questions
        ("MED-VENT-X200", "呼吸机使用时的电气安全要求是什么？",
         "1.使用接地插座 2.远离易燃麻醉气体 3.定期检查漏电流和绝缘电阻 4.电池充电时远离患者"),
        ("MED-CT-3200", "CT 扫描仪的辐射防护措施有哪些？",
         "1.操作间与控制室隔离 2.操作人员穿戴铅衣 3.患者非检查部位使用铅防护 4.定期进行辐射剂量检测 5.低剂量模式可减少 70% 辐射"),
        ("MED-MRI-1.5T", "磁共振检查前需要移除哪些物品？",
         "1.所有金属物品（首饰、手表、硬币）2.心脏起搏器（绝对禁忌）3.金属假牙 4.磁性纹身墨水 5.手机和电子卡片"),

        # Parameter comparison
        ("MULTIPLE", "呼吸机和监护仪哪个对电池续航要求更高？",
         "呼吸机要求更高。MED-VENT-X200 要求电池续航 ≥ 4 小时（转运场景），而 MED-MON-5000 监护仪为 8 小时但功耗更低。呼吸机需要驱动气泵和阀门，功耗远大于监护仪"),
        ("MULTIPLE", "CT 和 MRI 哪种检查辐射更大？",
         "CT 有电离辐射，单次胸部 CT 约 7 mSv（相当于 2 年自然本底辐射）。MRI 无电离辐射，使用磁场和射频脉冲，安全性更高"),
        ("MULTIPLE", "128 排 CT 和 64 排 CT 有什么区别？",
         "128 排探测器覆盖更广，单次旋转可采集更多层数据，扫描速度更快（MED-CT-3200 为 0.35 秒/圈），心脏成像质量更高，运动伪影更少"),

        # Troubleshooting flow
        ("MED-VENT-X200", "呼吸机高压报警（>35 cmH2O）的排查流程？",
         "排查流程：1.检查患者气道是否有分泌物阻塞 2.检查管路是否扭曲或受压 3.检查气管导管气囊是否过度充气 4.检查是否出现人机对抗 5.检查高压报警上限设置"),
        ("MED-CT-3200", "CT 图像出现运动伪影怎么处理？",
         "运动伪影处理：1.确认患者是否配合（必要时使用镇静）2.增加扫描速度（缩短扫描时间）3.使用心电门控（心脏扫描）4.增加管电流以提高信噪比"),

        # Regulatory and compliance
        ("MED-VENT-X200", "呼吸机年度电气安全测试包含哪些项目？",
         "包括：1.外壳漏电流测试（≤ 100μA）2.患者漏电流测试（≤ 10μA）3.绝缘电阻测试（≥ 2MΩ）4.接地连续性测试 5.报警功能全面测试"),
        ("MED-CT-3200", "CT 扫描仪每年需要做哪些法定检测？",
         "包括：1.辐射剂量检测（符合 GB 9706.244 标准）2.机械精度检测 3.图像质量 phantom 测试 4.安全联锁功能测试"),

        # Edge cases
        ("MED-MON-5000", "监护仪 NIBP 测量值比血压计高 20mmHg 怎么办？",
         "1.检查袖带尺寸是否合适（应为手臂围度的 80%）2.确认测量位置是否与心脏齐平 3.执行 NIBP 模块校准 4.更换袖带和管路重试"),
        ("MED-US-PRO", "超声弹性成像能做什么用？",
         "弹性成像用于评估组织硬度，主要应用：1.肝脏纤维化分级 2.乳腺结节良恶性鉴别 3.甲状腺结节评估 4.前列腺癌筛查"),

        # More spec deep-dives
        ("MED-VENT-X200", "MED-VENT-X200 的 PEEP 设置过高会有什么风险？",
         "PEEP 超过 25 cmH2O 可能导致气压伤、肺泡过度膨胀、静脉回流减少致低血压。MED-VENT-X200 的 PEEP 上限设为 25 cmH2O 就是为了防止这种情况"),
        ("MED-VENT-X200", "SIMV 模式和 A/C 模式有什么区别？",
         "SIMV（同步间歇指令通气）在设定频率给予指令呼吸，两次指令之间患者可自主呼吸；A/C（辅助/控制）模式下每次自主呼吸都会触发一次完整指令潮气量。SIMV 更适合脱机训练"),
        ("MED-CT-3200", "MED-CT-3200 的双能成像有什么用？",
         "双能成像通过高低管电压切换，可以区分不同材料（如碘对比剂和软组织），应用于：1.痛风结节检测 2.肺栓塞诊断 3.肾结石成分分析 4.灌注成像"),
        ("MED-MRI-1.5T", "1.5T 磁共振的 32 通道接收线圈有什么好处？",
         "32 通道可以同时采集更多信号，提高信噪比和成像速度，支持并行加速技术（如 SENSE、GRAPPA），缩短扫描时间 30-50%"),
        ("MED-MON-5000", "监护仪的 NIBP 测量原理是什么？",
         "采用振荡法测量：袖带充气至高于收缩压后缓慢放气，传感器检测动脉搏动引起的压力振荡，振荡幅度最大时对应的压力为平均动脉压，由此推算收缩压和舒张压"),
        ("MED-INF-500", "高压灭菌器 121°C 和 134°C 分别适用于什么物品？",
         "121°C 适用于液体、橡胶制品、精密器械等不耐高温的物品；134°C 适用于裸露金属器械的快速灭菌，但不适用于液体和耐热性差的物品"),
        ("MED-US-PRO", "超声的弹性成像和常规 B 超有什么区别？",
         "B 超看的是解剖结构（形态），弹性成像看的是组织硬度（功能）。两者结合可以提高病变鉴别的准确性，如乳腺结节的 BI-RADS 分级"),

        # More fault scenarios
        ("MED-VENT-X200", "呼吸机 E102 氧气供应中断时能继续工作吗？",
         "不能。E102 是严重故障，呼吸机将自动切换到内置电池供电并触发高级别报警。必须立即切换至备用气源或手动通气，否则 4 小时后设备将停机"),
        ("MED-CT-3200", "CT 高压发生器故障 A201 还能扫描吗？",
         "不能。A201 表示高压发生器故障，X 射线管无法产生高压，系统会自动锁定无法扫描。需要检查高压电缆连接，如电缆老化需更换"),
        ("MED-MON-5000", "监护仪心电导联接反了会怎样？",
         "导联接反会导致心电图波形异常（如 P 波倒置、QRS 轴偏移），可能误导临床判断。处理：核对导联标识重新连接，检查屏幕是否显示导联错误提示"),

        # Comparison questions
        ("MULTIPLE", "呼吸机的潮气量和监护仪的呼吸频率有什么关系？",
         "潮气量 × 呼吸频率 = 每分钟通气量。例如潮气量 500ml × 呼吸频率 15 次/分 = 7.5 L/min。呼吸机控制潮气量和频率，监护仪监测实际通气效果"),
        ("MULTIPLE", "CT 的低剂量模式和常规模式图像质量有差别吗？",
         "MED-CT-3200 的低剂量模式通过降低管电流实现，辐射减少 70%，但图像噪声会增加。对于骨骼和肺部检查影响较小，但对于软组织对比度要求高的检查（如腹部）建议使用常规模式"),
        ("MULTIPLE", "MRI 和超声哪个对软组织分辨率更高？",
         "MRI 的软组织分辨率远高于超声。MRI 利用磁场和射频信号，可以清晰区分肌肉、韧带、软骨等软组织；超声受声波衰减影响，深部软组织成像质量较差"),

        # Process questions
        ("MED-VENT-X200", "呼吸机从 ICU 转运到 CT 室需要做哪些准备？",
         "1.确认患者生命体征稳定 2.充满电池（续航≥4小时）3.携带便携式氧气瓶 4.备好简易呼吸囊（Ambu bag）作为备用 5.检查管路密封性 6.与 CT 室提前沟通"),
        ("MED-CT-3200", "CT 扫描前对患者需要做哪些准备？",
         "1.去除金属物品 2.确认无造影剂过敏史 3.空腹 4-6 小时（腹部扫描）4.签署知情同意书 5.孕妇需特别确认 6.安装 IV 通道（增强扫描）"),

        # Compliance
        ("MED-VENT-X200", "呼吸机符合哪些国际标准？",
         "1.IEC 60601-1 医用电气设备安全 2.IEC 60601-1-2 EMC 电磁兼容 3.IEC 60601-2-12 呼吸机专用标准 4.FDA 510(k) 美国上市许可"),
        ("MED-INF-500", "高压灭菌器需要哪些认证才能投入使用？",
         "1.特种设备制造许可证 2.压力容器检验合格证 3.计量检定证书（温度、压力）4.生物监测合格报告 5.消防验收合格证"),
    ]

    for device_code, question, expected in scenario_questions:
        pair_id += 1
        device_name = DEVICES[device_code]["name"] if device_code != "MULTIPLE" else "综合"
        pairs.append({
            "id": f"qa-{pair_id:04d}",
            "question": question,
            "expected_answer": expected,
            "relevant_contexts": [f"{device_code}_scenario#{pair_id}"],
            "device_code": device_code,
            "category": "scenario" if device_code != "MULTIPLE" else "comparison",
        })

    return pairs


def main():
    from collections import Counter

    pairs = generate_qa_pairs()

    output_path = Path(__file__).parent.parent / "data" / "eval" / "qa_golden_set.jsonl"

    # --- Add extra manual QA pairs to reach 200+ ---
    device_names = {d["name"]: d["category"] for d in DEVICES.values()}
    extras = [
        ("MED-VENT-X200", "呼吸机 PSV 模式和 A/C 模式怎么选？", "PSV（压力支持通气）适用于自主呼吸恢复良好的患者，每次呼吸由患者触发；A/C 模式适用于呼吸中枢功能不全的患者，机器保证最低呼吸频率和潮气量。选择依据是患者的自主呼吸能力。", "specification"),
        ("MED-CT-3200", "CT 扫描时患者体内有金属支架能做吗？", "大多数血管支架（钛合金材质）可以做 CT 扫描，但会产生金属伪影。MED-CT-3200 支持 MAR（金属伪影消除）算法减轻影响。心脏起搏器患者需特别评估。", "fault_diagnosis"),
        ("MED-MRI-1.5T", "磁共振检查大约需要多长时间？", "单次 MRI 扫描通常需要 15-45 分钟，取决于检查部位和序列数量。32 通道线圈和并行加速技术可将时间缩短 30-50%。", "specification"),
        ("MED-MON-5000", "监护仪可以同时监测几个参数？", "MED-MON-5000 可同时监测心电（ECG）、血氧（SpO2）、无创血压（NIBP）、有创血压（IBP）、呼吸（RESP）、体温（TEMP）等 6 个参数。", "specification"),
        ("MED-US-PRO", "超声诊断仪可以做心脏检查吗？", "可以。MED-US-PRO 支持彩色多普勒模式，可以进行心脏彩超检查，评估心功能、瓣膜情况和血流动力学。", "specification"),
        ("MED-INF-500", "高压灭菌器灭菌失败（生物监测阳性）怎么办？", "1.立即停止使用该批次器械 2.排查原因（温度不足、时间不够、包装不当）3.重新灭菌 4.检查灭菌器定期维护记录 5.联系生物监测实验室复核", "fault_diagnosis"),
        ("MED-VENT-X200", "呼吸机可以家用吗？", "可以。MED-VENT-X200 适用于家庭长期机械通气。需配备氧气浓缩器和备用电源，家属需接受专业操作培训。内置电池可在断电时提供 ≥4 小时支持。", "scenario"),
        ("MED-CT-3200", "儿童可以做 CT 扫描吗？", "可以，但需要调整扫描参数（降低管电压和管电流）以减少辐射。MED-CT-3200 的低剂量模式特别适合儿科检查。对于非必要情况，优先考虑 MRI 或超声。", "scenario"),
        ("MULTIPLE", "医疗设备日常维护谁来做？月度以上维护谁来做？", "日常维护由科室护士或技师完成；月度及以上维护由医院设备科工程师或厂家售后工程师执行。年度维护通常需要厂家资质认证。", "maintenance"),
        ("MED-VENT-X200", "呼吸机报警静音后还会继续报警吗？", "会。静音只是暂时关闭声音报警，视觉报警（屏幕闪烁、指示灯）仍然有效。高压、低压、窒息等高级别报警无法完全静音，必须处理后才会消除。", "scenario"),
        ("MED-CT-3200", "CT 扫描前为什么要脱掉衣服？", "衣物上的金属拉链、纽扣会在图像中产生伪影。此外，扫描床需要直接接触患者皮肤以确保定位准确。医院会提供扫描服代替日常衣物。", "scenario"),
        ("MED-MON-5000", "监护仪报警但不代表病人有问题，这种情况常见吗？", "常见，称为假报警（false alarm）。原因包括导联脱落、袖带未绑紧、患者运动干扰等。MED-MON-5000 具有智能报警过滤功能减少假报警。", "scenario"),
        ("MED-US-PRO", "超声检查需要空腹吗？", "腹部超声检查需要空腹 6-8 小时（减少肠道气体干扰）。心脏超声、血管超声、妇产科超声不需要空腹。具体遵医嘱。", "scenario"),
        ("MED-INF-500", "高压灭菌器可以灭菌手机（牙科手机）吗？", "可以。牙科手机耐高温，适合 134°C 快速灭菌模式（18 分钟）。但需要先彻底清洗，去除血液和组织残留，否则灭菌效果不合格。", "scenario"),
        ("MED-VENT-X200", "SIMV 模式的呼吸频率设太低会怎样？", "SIMV 指令频率过低可能导致分钟通气量不足，引起 CO2 潴留和呼吸性酸中毒。一般设置为 8-16 次/分，根据患者自主呼吸能力调整。", "fault_diagnosis"),
        ("MED-CT-3200", "CT 造影剂过敏怎么办？", "轻度：抗组胺药治疗；中度：肾上腺素 + 糖皮质激素；重度：气管插管 + 心肺复苏。MED-CT-3200 扫描前需询问过敏史，高危患者需预处理（糖皮质激素 + 抗组胺药）。", "scenario"),
        ("MED-MRI-1.5T", "怀孕可以做磁共振吗？", "孕早期（前三个月）尽量避免 MRI，除非获益大于风险。中晚期在必要时可以做（不使用对比剂）。超声是孕期首选影像学检查。", "scenario"),
        ("MED-MON-5000", "监护仪的呼吸频率是怎么测出来的？", "MED-MON-5000 通过心电导联检测胸廓运动引起的阻抗变化来测量呼吸频率，也可以外接呼吸气流传感器。心电导联法最常用，但运动干扰可能导致读数不准确。", "specification"),
        ("MED-US-PRO", "超声的彩色多普勒和普通多普勒有什么区别？", "普通多普勒只能显示血流速度和方向（频谱图），彩色多普勒可以在图像上用颜色直观显示血流方向和速度（红色朝向探头，蓝色远离）。", "specification"),
        ("MED-INF-500", "高压灭菌器灭菌包的尺寸有限制吗？", "有。MED-INF-500 容量 50L，灭菌包不应超过 30cm×30cm×30cm，重量不超过 10kg。过大的包裹会影响蒸汽渗透，导致灭菌不彻底。", "specification"),
        ("MED-VENT-X200", "呼吸机撤机（脱离呼吸机）的条件是什么？", "撤机条件：1.原发病已控制 2.自主呼吸有力 3.SpO2 > 92%（FiO2 ≤ 40%）4.PEEP ≤ 5-8 cmH2O 5.血气分析正常。可通过 SIMV 或 PSV 逐步过渡。", "scenario"),
        ("MED-CT-3200", "CT 扫描的辐射会导致癌症吗？", "单次胸部 CT 约 7 mSv，略高于 2 年自然本底辐射。风险极低但并非为零，因此应遵循 ALARA 原则（合理可行尽量低）。MED-CT-3200 低剂量模式可减少 70% 辐射。", "scenario"),
        ("MULTIPLE", "医疗设备采购时需要关注哪些技术指标？", "1.技术参数是否满足临床需求 2.售后服务和备件供应 3.培训和支持 4.合规认证（FDA/CE/NMPA）5.总拥有成本（含维护费用）6.与其他设备的兼容性。", "comparison"),
        ("MED-VENT-X200", "呼吸机管路多久更换一次？", "一次性管路建议每 24 小时更换，或有污染/损坏时立即更换。可重复使用的管路需经高水平消毒。ICU 高频使用环境下建议每周更换。", "maintenance"),
        ("MED-CT-3200", "CT 水校准失败的常见原因？", "1.水箱水位不足 2.水箱未正确安装 3.温度传感器异常 4.水循环泵故障。处理：检查水位和安装，重启校准程序，如仍失败联系售后。", "fault_diagnosis"),
        ("MED-MON-5000", "监护仪可以连接医院信息系统（HIS）吗？", "可以。MED-MON-5000 支持 HL7 协议，可以将监测数据实时传输到护士站中央监护系统和 HIS。需要配置网络参数和数据映射。", "specification"),
    ]

    for i, (device, question, expected, cat) in enumerate(extras):
        pair_id = len(pairs) + i + 1
        pairs.append({
            "id": f"qa-{pair_id:04d}",
            "question": question,
            "expected_answer": expected,
            "relevant_contexts": [f"{device}_extra#{pair_id}"],
            "device_code": device,
            "category": cat,
        })

    cats = Counter(p["category"] for p in pairs)
    devs = Counter(p["device_code"] for p in pairs)

    print(f"Generated {len(pairs)} QA pairs:")
    print(f"  By category: {dict(cats)}")
    print(f"  By device: {dict(devs)}")
    print(f"\nSaved to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
