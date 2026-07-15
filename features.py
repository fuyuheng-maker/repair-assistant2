import json
import os
import uuid
import re
import threading
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException, Depends, Query, Form

router = APIRouter()

# 全局变量占位，由 init_features 注入
_embed_model = None
_get_user_collection = None  # 函数，用于获取用户 collection
_client = None
_TEXT_MODEL = None
_get_current_user = None
_require_admin = None

PROCEDURE_TEMPLATES = {
    ("发动机", "日常检查"): [
        {"step": 1, "desc": "检查机油液位", "compliance": "冷车时油位应在上下标记之间", "tool": "机油尺"},
        {"step": 2, "desc": "检查冷却液液位", "compliance": "液面不低于 LOW 线", "tool": "目视"},
        {"step": 3, "desc": "检查皮带松紧度", "compliance": "按压10mm无异常，无裂纹", "tool": "手压"},
        {"step": 4, "desc": "检查进气系统", "compliance": "空滤无破损，进气管道无漏气", "tool": "目视"},
        {"step": 5, "desc": "检查排气系统", "compliance": "无漏气声，支架无松动", "tool": "目视/听诊"},
    ],
    ("发动机", "小修"): [
        {"step": 1, "desc": "更换机油机滤", "compliance": "扭矩25N·m，密封垫圈", "tool": "扭矩扳手"},
        {"step": 2, "desc": "清洗空滤", "compliance": "吹尘方向从内向外", "tool": "压缩空气"},
        {"step": 3, "desc": "检查火花塞", "compliance": "电极间隙0.8-0.9mm", "tool": "塞尺"},
        {"step": 4, "desc": "检查节气门", "compliance": "积碳清理干净，开度正常", "tool": "清洗剂"},
        {"step": 5, "desc": "检查冷却系统", "compliance": "散热器无堵塞，节温器工作正常", "tool": "压力表"},
    ],
    ("发动机", "中修"): [
        {"step": 1, "desc": "更换正时皮带/链条", "compliance": "按厂家规定扭矩拧紧", "tool": "正时工具组"},
        {"step": 2, "desc": "检查气门间隙", "compliance": "进气0.25-0.30mm，排气0.30-0.35mm", "tool": "塞尺"},
        {"step": 3, "desc": "更换水泵", "compliance": "密封良好，轴承无间隙", "tool": "扳手"},
        {"step": 4, "desc": "检查活塞环", "compliance": "开口间隙0.20-0.40mm", "tool": "塞尺"},
        {"step": 5, "desc": "清洗喷油嘴", "compliance": "雾化良好，流量一致", "tool": "超声波清洗机"},
    ],
    ("发动机", "大修"): [
        {"step": 1, "desc": "拆卸发动机总成", "compliance": "使用吊具，保持垂直", "tool": "吊车"},
        {"step": 2, "desc": "解体气缸盖", "compliance": "对角线顺序分2-3次拧松", "tool": "扭矩扳手"},
        {"step": 3, "desc": "测量气缸内径", "compliance": "圆度<0.05mm，圆柱度<0.10mm", "tool": "量缸表"},
        {"step": 4, "desc": "研磨气门", "compliance": "接触面宽度1.2-1.5mm", "tool": "气门研磨机"},
        {"step": 5, "desc": "组装发动机", "compliance": "按厂家规定顺序和扭矩拧紧", "tool": "扭矩扳手"},
        {"step": 6, "desc": "冷磨热试", "compliance": "怠速运转30分钟，无异常响声", "tool": "听诊器"},
    ],
    ("变速箱", "日常检查"): [
        {"step": 1, "desc": "检查变速箱油液位", "compliance": "油温60-80℃时检查，油位在标记之间", "tool": "油尺"},
        {"step": 2, "desc": "检查变速箱油质", "compliance": "油液无异味、无金属屑", "tool": "目视"},
        {"step": 3, "desc": "检查漏油情况", "compliance": "壳体密封处无渗漏", "tool": "目视"},
        {"step": 4, "desc": "检查换挡机构", "compliance": "换挡顺畅，无卡滞", "tool": "手动操作"},
    ],
    ("变速箱", "小修"): [
        {"step": 1, "desc": "更换变速箱油", "compliance": "使用厂家指定油品", "tool": "放油扳手"},
        {"step": 2, "desc": "更换变速箱滤芯", "compliance": "滤芯安装到位，密封圈完好", "tool": "扳手"},
        {"step": 3, "desc": "检查油封", "compliance": "无老化、无渗漏", "tool": "目视"},
        {"step": 4, "desc": "调整换挡拉线", "compliance": "拉线松紧适度，换挡位置准确", "tool": "调整扳手"},
    ],
    ("变速箱", "中修"): [
        {"step": 1, "desc": "拆卸变速箱上盖", "compliance": "按顺序拆卸，零件分类摆放", "tool": "套筒组"},
        {"step": 2, "desc": "检查同步器", "compliance": "同步环磨损不超限，弹簧张力正常", "tool": "塞尺"},
        {"step": 3, "desc": "检查齿轮磨损", "compliance": "齿面无点蚀、剥落", "tool": "目视"},
        {"step": 4, "desc": "更换轴承", "compliance": "轴承转动灵活，无间隙", "tool": "压床"},
        {"step": 5, "desc": "组装调试", "compliance": "各档位啮合良好，换挡顺畅", "tool": "手动操作"},
    ],
    ("变速箱", "大修"): [
        {"step": 1, "desc": "完全解体变速箱", "compliance": "按步骤拆卸，做好标记", "tool": "专用工具"},
        {"step": 2, "desc": "清洗所有零件", "compliance": "零件清洁无油污", "tool": "清洗剂"},
        {"step": 3, "desc": "检查行星齿轮组", "compliance": "齿轮间隙正常，轴承完好", "tool": "百分表"},
        {"step": 4, "desc": "更换摩擦片", "compliance": "新片浸泡油液后安装", "tool": "压具"},
        {"step": 5, "desc": "调整间隙", "compliance": "离合器间隙符合厂家标准", "tool": "塞尺"},
        {"step": 6, "desc": "总成装配", "compliance": "按顺序组装，加注规定油量", "tool": "扭矩扳手"},
    ],
    ("液压系统", "日常检查"): [
        {"step": 1, "desc": "检查液压油液位", "compliance": "液位在刻度范围内", "tool": "液位计"},
        {"step": 2, "desc": "检查液压油温度", "compliance": "正常工作温度40-60℃", "tool": "温度计"},
        {"step": 3, "desc": "检查系统泄漏", "compliance": "各接头、油缸无渗漏", "tool": "目视"},
        {"step": 4, "desc": "检查滤清器", "compliance": "滤芯无堵塞，指示灯未亮", "tool": "目视"},
        {"step": 5, "desc": "检查蓄能器压力", "compliance": "氮气压力符合规定值", "tool": "压力表"},
    ],
    ("液压系统", "小修"): [
        {"step": 1, "desc": "更换液压油", "compliance": "排放彻底，使用规定牌号", "tool": "放油阀"},
        {"step": 2, "desc": "更换液压滤芯", "compliance": "滤芯型号正确，安装到位", "tool": "扳手"},
        {"step": 3, "desc": "清洗液压油箱", "compliance": "油箱内部清洁无杂质", "tool": "清洗剂"},
        {"step": 4, "desc": "检查密封圈", "compliance": "更换老化密封圈", "tool": "密封圈套件"},
    ],
    ("液压系统", "中修"): [
        {"step": 1, "desc": "测试油泵性能", "compliance": "压力、流量符合规定", "tool": "压力测试仪"},
        {"step": 2, "desc": "检查控制阀", "compliance": "阀芯无卡滞，响应正常", "tool": "液压试验台"},
        {"step": 3, "desc": "检查液压缸", "compliance": "缸筒内壁无拉伤，密封良好", "tool": "内径量表"},
        {"step": 4, "desc": "检查油管", "compliance": "油管无老化、裂纹", "tool": "目视"},
        {"step": 5, "desc": "系统排气", "compliance": "系统无空气，运行平稳", "tool": "排气阀"},
    ],
    ("液压系统", "大修"): [
        {"step": 1, "desc": "拆卸液压系统", "compliance": "分步拆卸，做好标记", "tool": "套筒组"},
        {"step": 2, "desc": "检修液压泵", "compliance": "泵体无磨损，密封良好", "tool": "专用工具"},
        {"step": 3, "desc": "检修控制阀组", "compliance": "各阀动作灵敏，无内漏", "tool": "试验台"},
        {"step": 4, "desc": "更换液压缸密封", "compliance": "密封件安装正确，无扭曲", "tool": "安装工具"},
        {"step": 5, "desc": "更换全部油管", "compliance": "新管符合压力等级", "tool": "管接头"},
        {"step": 6, "desc": "系统调试", "compliance": "各动作平稳，无冲击", "tool": "调试设备"},
    ],
    ("电气系统", "日常检查"): [
        {"step": 1, "desc": "检查蓄电池电压", "compliance": "空载电压12.6V以上", "tool": "万用表"},
        {"step": 2, "desc": "检查线路连接", "compliance": "接头无松动、腐蚀", "tool": "目视"},
        {"step": 3, "desc": "检查保险丝", "compliance": "保险丝完好，规格正确", "tool": "试灯"},
        {"step": 4, "desc": "检查照明系统", "compliance": "各灯光工作正常", "tool": "目视"},
        {"step": 5, "desc": "检查仪表显示", "compliance": "各仪表指示准确", "tool": "目视"},
    ],
    ("电气系统", "小修"): [
        {"step": 1, "desc": "更换蓄电池", "compliance": "新电池规格匹配，接线正确", "tool": "扳手"},
        {"step": 2, "desc": "更换保险丝", "compliance": "使用相同规格保险丝", "tool": "保险丝夹"},
        {"step": 3, "desc": "修复线路接头", "compliance": "接头清洁，紧固可靠", "tool": "端子钳"},
        {"step": 4, "desc": "更换灯泡", "compliance": "灯泡规格正确，安装牢固", "tool": "手"},
    ],
    ("电气系统", "中修"): [
        {"step": 1, "desc": "检查发电机", "compliance": "输出电压13.5-14.5V", "tool": "万用表"},
        {"step": 2, "desc": "检查启动电机", "compliance": "启动有力，无异常响声", "tool": "电流表"},
        {"step": 3, "desc": "检查线路老化", "compliance": "更换老化线束", "tool": "剥线钳"},
        {"step": 4, "desc": "检查传感器", "compliance": "传感器信号准确", "tool": "诊断仪"},
        {"step": 5, "desc": "检查接地系统", "compliance": "接地电阻小于1Ω", "tool": "电阻表"},
    ],
    ("电气系统", "大修"): [
        {"step": 1, "desc": "全面检测电气系统", "compliance": "按电路图逐一排查", "tool": "万用表/示波器"},
        {"step": 2, "desc": "更换主线束", "compliance": "新线束与原车一致", "tool": "扎带"},
        {"step": 3, "desc": "检修 ECU", "compliance": "ECU工作正常，无故障码", "tool": "诊断仪"},
        {"step": 4, "desc": "检修控制模块", "compliance": "各模块通信正常", "tool": "CAN分析仪"},
        {"step": 5, "desc": "系统编程", "compliance": "软件版本最新", "tool": "编程设备"},
        {"step": 6, "desc": "功能测试", "compliance": "所有电气功能正常", "tool": "目视/测试"},
    ],
    ("制动系统", "日常检查"): [
        {"step": 1, "desc": "检查制动液液位", "compliance": "液位在MAX-MIN之间", "tool": "目视"},
        {"step": 2, "desc": "检查刹车片磨损", "compliance": "剩余厚度≥3mm", "tool": "卡尺"},
        {"step": 3, "desc": "检查刹车盘", "compliance": "盘面无裂纹、沟槽", "tool": "目视"},
        {"step": 4, "desc": "检查制动管路", "compliance": "管路无渗漏、老化", "tool": "目视"},
        {"step": 5, "desc": "测试制动性能", "compliance": "制动灵敏，无跑偏", "tool": "路试"},
    ],
    ("制动系统", "小修"): [
        {"step": 1, "desc": "更换刹车片", "compliance": "新片涂抹消音膏", "tool": "刹车分泵工具"},
        {"step": 2, "desc": "更换制动液", "compliance": "按顺序放气，无气泡", "tool": "放气阀"},
        {"step": 3, "desc": "打磨刹车盘", "compliance": "盘面平整，粗糙度合适", "tool": "刹车盘打磨机"},
        {"step": 4, "desc": "检查刹车分泵", "compliance": "活塞回位顺畅，无卡滞", "tool": "分泵活塞工具"},
    ],
    ("制动系统", "中修"): [
        {"step": 1, "desc": "更换刹车盘", "compliance": "新盘厚度符合规定", "tool": "扳手"},
        {"step": 2, "desc": "检修制动钳", "compliance": "导向销润滑良好，活塞无锈蚀", "tool": "润滑脂"},
        {"step": 3, "desc": "检查手刹系统", "compliance": "手刹行程合适，有效", "tool": "扳手"},
        {"step": 4, "desc": "检查ABS系统", "compliance": "ABS传感器工作正常", "tool": "诊断仪"},
        {"step": 5, "desc": "更换制动软管", "compliance": "新管无老化、鼓包", "tool": "扳手"},
    ],
    ("制动系统", "大修"): [
        {"step": 1, "desc": "拆卸制动系统", "compliance": "分步拆卸，做好标记", "tool": "套筒组"},
        {"step": 2, "desc": "检修制动总泵", "compliance": "活塞密封良好，回油顺畅", "tool": "专用工具"},
        {"step": 3, "desc": "检修真空助力器", "compliance": "助力效果正常，无漏气", "tool": "真空泵"},
        {"step": 4, "desc": "更换所有制动管路", "compliance": "新管符合规格", "tool": "管接头"},
        {"step": 5, "desc": "更换ABS模块", "compliance": "编码正确，通信正常", "tool": "诊断仪"},
        {"step": 6, "desc": "系统排气调试", "compliance": "制动踏板软硬适中", "tool": "放气阀"},
    ],
    ("冷却系统", "日常检查"): [
        {"step": 1, "desc": "检查冷却液液位", "compliance": "液位在MAX-MIN之间", "tool": "目视"},
        {"step": 2, "desc": "检查冷却风扇", "compliance": "风扇运转正常，无异响", "tool": "目视"},
        {"step": 3, "desc": "检查散热器", "compliance": "散热片无堵塞、破损", "tool": "目视"},
        {"step": 4, "desc": "检查水管", "compliance": "水管无老化、渗漏", "tool": "目视"},
        {"step": 5, "desc": "检查节温器", "compliance": "温度达到时正常开启", "tool": "温度计"},
    ],
    ("冷却系统", "小修"): [
        {"step": 1, "desc": "更换冷却液", "compliance": "按比例混合，排放彻底", "tool": "放水阀"},
        {"step": 2, "desc": "清洗散热器", "compliance": "散热片清洁无堵塞", "tool": "高压水枪"},
        {"step": 3, "desc": "更换水管", "compliance": "新管卡箍紧固", "tool": "卡箍钳"},
        {"step": 4, "desc": "检查水泵皮带", "compliance": "皮带张力适中，无裂纹", "tool": "张力计"},
    ],
    ("冷却系统", "中修"): [
        {"step": 1, "desc": "更换水泵", "compliance": "新泵密封良好，轴承无间隙", "tool": "扳手"},
        {"step": 2, "desc": "更换节温器", "compliance": "开启温度符合规定", "tool": "扳手"},
        {"step": 3, "desc": "更换散热器", "compliance": "新散热器规格匹配", "tool": "扳手"},
        {"step": 4, "desc": "检查风扇电机", "compliance": "电机运转正常，无卡顿", "tool": "万用表"},
        {"step": 5, "desc": "系统加压测试", "compliance": "压力稳定，无泄漏", "tool": "压力表"},
    ],
    ("冷却系统", "大修"): [
        {"step": 1, "desc": "拆卸冷却系统", "compliance": "分步拆卸，做好标记", "tool": "套筒组"},
        {"step": 2, "desc": "更换所有软管", "compliance": "新管规格正确", "tool": "卡箍"},
        {"step": 3, "desc": "更换水泵总成", "compliance": "泵体无砂眼，叶轮完好", "tool": "螺栓"},
        {"step": 4, "desc": "更换散热器总成", "compliance": "散热面积符合要求", "tool": "支架"},
        {"step": 5, "desc": "更换风扇总成", "compliance": "风扇叶片完好，电机正常", "tool": "螺钉"},
        {"step": 6, "desc": "系统组装调试", "compliance": "无气泡，温度正常", "tool": "排气工具"},
    ],
}

PENDING_CASES_FILE = "pending_cases.json"
CORRECTIONS_FILE = "corrections.json"
_json_locks = {PENDING_CASES_FILE: threading.Lock(), CORRECTIONS_FILE: threading.Lock()}

for f in [PENDING_CASES_FILE, CORRECTIONS_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            json.dump([], fp)

def load_json(path):
    lock = _json_locks.get(path, threading.Lock())
    with lock:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def save_json(path, data):
    lock = _json_locks.get(path, threading.Lock())
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- 功能函数（无依赖） ----------
def get_procedure_handler(device, level, username):
    # 输入长度限制，防止 Prompt 注入
    device = device[:50].strip()
    level = level[:20].strip()
    
    template = PROCEDURE_TEMPLATES.get((device, level))
    if template:
        return {"device": device, "level": level, "steps": template, "source": "template"}
    try:
        prompt = f"请为设备“{device}”的“{level}”检修生成详细步骤清单（JSON数组，每项含step、desc、compliance、tool）。只输出JSON，不要输出其他内容。"
        resp = _client.chat.completions.create(
            model=_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = resp.choices[0].message.content
        # 尝试直接解析整个响应为 JSON
        try:
            steps = json.loads(raw)
            if isinstance(steps, list):
                return {"device": device, "level": level, "steps": steps, "source": "ai_generated"}
        except json.JSONDecodeError:
            pass
        # 如果直接解析失败，尝试提取第一个完整的 JSON 数组
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            try:
                steps = json.loads(match.group())
                return {"device": device, "level": level, "steps": steps, "source": "ai_generated"}
            except json.JSONDecodeError:
                pass
        return {"device": device, "level": level, "steps_text": raw, "source": "ai_generated_raw"}
    except Exception as e:
        raise HTTPException(500, f"生成指引失败：{e}")

def submit_case_logic(title, content, type, username):
    cases = load_json(PENDING_CASES_FILE)
    cases.append({
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "type": type,
        "submitter": username,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    })
    save_json(PENDING_CASES_FILE, cases)

def get_pending_cases_logic():
    return [c for c in load_json(PENDING_CASES_FILE) if c["status"] == "pending"]

def review_case_logic(case_id, action):
    cases = load_json(PENDING_CASES_FILE)
    for case in cases:
        if case["id"] == case_id:
            if action == "approve":
                texts = [case["content"]]
                emb = _embed_model.encode(texts).tolist()
                # 审核通过的案例同时加入提交者知识库和全局知识库
                user_col = _get_user_collection(case["submitter"])
                user_col.add(
                    ids=[f"case_{case_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "user_case", "title": case["title"], "type": case["type"]}],
                    documents=texts
                )
                # 加入全局知识库供所有用户检索
                global_col = _get_user_collection("global")
                global_col.add(
                    ids=[f"global_case_{case_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "global_case", "title": case["title"], "type": case["type"], "submitter": case["submitter"]}],
                    documents=texts
                )
                case["status"] = "approved"
            else:
                case["status"] = "rejected"
            save_json(PENDING_CASES_FILE, cases)
            return f"案例已{action}"
    raise HTTPException(404, "案例不存在")

def submit_correction_logic(question, original_answer, correct_answer, username):
    corrections = load_json(CORRECTIONS_FILE)
    corrections.append({
        "id": str(uuid.uuid4()),
        "question": question,
        "original_answer": original_answer,
        "correct_answer": correct_answer,
        "submitter": username,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    })
    save_json(CORRECTIONS_FILE, corrections)

def get_pending_corrections_logic():
    return [c for c in load_json(CORRECTIONS_FILE) if c["status"] == "pending"]

def review_correction_logic(correction_id, action):
    corrections = load_json(CORRECTIONS_FILE)
    for corr in corrections:
        if corr["id"] == correction_id:
            if action == "approve":
                texts = [f"问题：{corr['question']}\n正确答案：{corr['correct_answer']}"]
                emb = _embed_model.encode(texts).tolist()
                # 审核通过的修正加入提交者的个人知识库
                user_col = _get_user_collection(corr["submitter"])
                user_col.add(
                    ids=[f"correction_{correction_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "user_correction"}],
                    documents=texts
                )
                corr["status"] = "approved"
            else:
                corr["status"] = "rejected"
            save_json(CORRECTIONS_FILE, corrections)
            return f"修正已{action}"
    raise HTTPException(404, "修正记录不存在")

# ---------- 动态注册路由 ----------
def init_features(embed_model, get_user_collection, client, text_model, get_current_user, require_admin):
    global _embed_model, _get_user_collection, _client, _TEXT_MODEL, _get_current_user, _require_admin
    _embed_model = embed_model
    _get_user_collection = get_user_collection
    _client = client
    _TEXT_MODEL = text_model
    _get_current_user = get_current_user
    _require_admin = require_admin

    @router.get("/procedure")
    async def procedure(device: str = Query(...), level: str = Query(...), user: Dict = Depends(_get_current_user)):
        return get_procedure_handler(device, level, user["username"])

    @router.post("/submit-case")
    async def submit_case(title: str = Form(...), content: str = Form(...), type: str = Form("经验"), user: Dict = Depends(_get_current_user)):
        submit_case_logic(title, content, type, user["username"])
        return {"message": "案例已提交，等待审核"}

    @router.get("/pending-cases")
    async def pending_cases(admin: Dict = Depends(_require_admin)):
        return {"cases": get_pending_cases_logic()}

    @router.post("/review-case/{case_id}")
    async def review_case(case_id: str, action: str = Query(..., pattern="^(approve|reject)$"), admin: Dict = Depends(_require_admin)):
        msg = review_case_logic(case_id, action)
        return {"message": msg}

    @router.post("/submit-correction")
    async def submit_correction(question: str = Form(...), original_answer: str = Form(...), correct_answer: str = Form(...), user: Dict = Depends(_get_current_user)):
        submit_correction_logic(question, original_answer, correct_answer, user["username"])
        return {"message": "修正建议已提交"}

    @router.get("/pending-corrections")
    async def pending_corrections(admin: Dict = Depends(_require_admin)):
        return {"corrections": get_pending_corrections_logic()}

    @router.post("/review-correction/{correction_id}")
    async def review_correction(correction_id: str, action: str = Query(..., pattern="^(approve|reject)$"), admin: Dict = Depends(_require_admin)):
        msg = review_correction_logic(correction_id, action)
        return {"message": msg}