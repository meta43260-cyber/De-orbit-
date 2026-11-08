# -*- coding: utf-8 -*-
"""
Thai Re-entry Predictor V1.0
เครื่องมือวิเคราะห์การลดระดับวงโคจรจากข้อมูล TLE สาธารณะ
ข้อจำกัด: การคาดการณ์เวลาจากแนวโน้มกึ่งแกนเอกเป็นแบบจำลองเชิงประจักษ์
ไม่ใช่การจำลองแรงต้านบรรยากาศ GMAT/NRLMSISE-00 เต็มรูปแบบ
"""
import io
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sgp4.api import Satrec, jday

st.set_page_config(page_title="Thai Re-entry Predictor", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Noto Sans Thai", "Tahoma", sans-serif; }
.block-container { padding-top: 1.3rem; }
.hero { padding: 1.15rem 1.4rem; border-radius: 14px;
        background: linear-gradient(115deg,#10243d,#173e58); color: white; }
.small-muted { color: #8b98a8; font-size: .88rem; }
.metric-note { font-size: .82rem; color: #687385; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🛰️ Thai Re-entry Predictor V1.0</h1>
<div>ระบบวิเคราะห์การลดระดับวงโคจรและประเมินการกลับเข้าสู่บรรยากาศโลก</div>
<div style="opacity:.8;font-size:.88rem;margin-top:.35rem">ภาษาไทย • ข้อมูล TLE • การวิเคราะห์เชิงประจักษ์ • UTC และเวลาไทย (UTC+7)</div>
</div>
""", unsafe_allow_html=True)

st.warning("ข้อควรทราบ: เครื่องมือนี้เป็นต้นแบบวิจัย ไม่ใช่ระบบเตือนภัยหรือระบบยืนยันจุดตก ผลพยากรณ์เป็นการประมาณจาก TLE และอาจใช้ไม่ได้เมื่อวัตถุมีการจุดเครื่องยนต์ เปลี่ยนท่าทาง หรือข้อมูล TLE ผิดปกติ")

MU = 398600.8  # km^3/s^2, ใกล้เคียงค่าคงที่ WGS-72 ที่ใช้กับ SGP4
EARTH_R = 6378.137
ENTRY_ALT = 120.0

def parse_tle_text(raw):
    lines = [x.strip() for x in raw.replace("\r", "").split("\n") if x.strip()]
    records, errors = [], []
    i = 0
    while i < len(lines):
        name = ""
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i+1].startswith("2 "):
            l1, l2 = lines[i], lines[i+1]
            i += 2
        elif i + 2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            name, l1, l2 = lines[i], lines[i+1], lines[i+2]
            i += 3
        else:
            errors.append(f"ข้ามบรรทัดที่ {i+1}: ไม่พบชุด TLE ที่สมบูรณ์")
            i += 1
            continue
        try:
            sat = Satrec.twoline2rv(l1, l2)
            epoch = datetime(1949, 12, 31, tzinfo=timezone.utc) + timedelta(days=float(sat.jdsatepoch - 2433282.5 + sat.jdsatepochF))
            n_rad_s = sat.no_kozai / 60.0
            a_km = (MU / (n_rad_s ** 2)) ** (1/3)
            e = float(sat.ecco)
            inc = math.degrees(float(sat.inclo))
            records.append({
                "ชื่อ": name or f"NORAD {sat.satnum}",
                "NORAD": int(sat.satnum),
                "เวลา_epoch": epoch,
                "กึ่งแกนเอก_km": a_km,
                "ความสูงเฉลี่ย_km": a_km - EARTH_R,
                "ความเยื้องศูนย์กลาง": e,
                "ความเอียง_deg": inc,
                "คาบ_orbit_min": 2 * math.pi / n_rad_s / 60,
                "TLE_1": l1, "TLE_2": l2, "_sat": sat
            })
        except Exception as ex:
            errors.append(f"อ่าน TLE ไม่สำเร็จ: {ex}")
    return records, errors

def gmst_angle(jd):
    t = (jd - 2451545.0) / 36525.0
    deg = 280.46061837 + 360.98564736629*(jd-2451545.0) + 0.000387933*t*t - t*t*t/38710000.0
    return math.radians(deg % 360)

def ground_track(sat, start_utc, hours=24, step_min=2):
    rows = []
    for mins in np.arange(0, hours*60 + step_min, step_min):
        dt = start_utc + timedelta(minutes=float(mins))
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second + dt.microsecond/1e6)
        err, r, v = sat.sgp4(jd, fr)
        if err != 0:
            continue
        theta = gmst_angle(jd + fr)
        x, y, z = r
        xe = x*math.cos(theta) + y*math.sin(theta)
        ye = -x*math.sin(theta) + y*math.cos(theta)
        lon = math.degrees(math.atan2(ye, xe))
        lat = math.degrees(math.atan2(z, math.sqrt(xe*xe + ye*ye)))
        rows.append({"เวลา UTC": dt, "ละติจูด": lat, "ลองจิจูด": lon})
    return pd.DataFrame(rows)

def regression_forecast(df, threshold_alt=ENTRY_ALT, n_sim=3000):
    # ใช้เฉพาะแนวโน้มช่วงท้าย ไม่ลากเส้นยาวจากข้อมูลทั้งหมด
    d = df.sort_values("เวลา_epoch").copy()
    if len(d) < 4:
        return None
    t0 = d["เวลา_epoch"].iloc[-1]
    d["ชั่วโมงจากล่าสุด"] = (d["เวลา_epoch"] - t0).dt.total_seconds()/3600
    # เลือกข้อมูล 8 จุดท้าย หรือช่วง 72 ชั่วโมงท้าย
    tail = d.tail(8).copy()
    x = tail["ชั่วโมงจากล่าสุด"].to_numpy(float)
    y = tail["ความสูงเฉลี่ย_km"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    pred_hours = (threshold_alt - intercept) / slope if slope < 0 else np.inf
    # Monte Carlo จาก bootstrap residual + slope uncertainty แบบประมาณ
    fit = slope*x + intercept
    resid = y-fit
    rng = np.random.default_rng(42)
    slopes = []
    for _ in range(n_sim):
        ys = fit + rng.choice(resid, size=len(resid), replace=True) + rng.normal(0, max(np.std(resid), 0.02), len(resid))
        try:
            s, b = np.polyfit(x, ys, 1)
            if s < -1e-5:
                h = (threshold_alt-b)/s
                if h > 0 and np.isfinite(h) and h < 24*365:
                    slopes.append(h)
        except Exception:
            pass
    if not np.isfinite(pred_hours) or pred_hours <= 0:
        return {"slope": slope, "pred": None, "samples": np.array([]), "tail": tail}
    samples = np.array(slopes)
    # กรองค่าที่เกิดจากความชันใกล้ศูนย์จนเกินจริง ด้วยเปอร์เซ็นไทล์ 5-95
    if len(samples) >= 30:
        lo, med, hi = np.percentile(samples, [5, 50, 95])
    else:
        lo = med = hi = np.nan
    return {"slope": slope, "pred": t0 + timedelta(hours=float(pred_hours)),
            "samples": samples, "interval": (lo, hi, med), "tail": tail}

def fmt_dt(dt):
    if dt is None: return "ยังประเมินไม่ได้"
    return dt.strftime("%d/%m/%Y %H:%M UTC")

with st.sidebar:
    st.header("⚙️ ตั้งค่าการวิเคราะห์")
    source = st.radio("แหล่งข้อมูล TLE", ["วางข้อมูล / อัปโหลดไฟล์", "ดึงจาก CelesTrak"])
    threshold = st.number_input("ระดับความสูงอ้างอิงสำหรับการเข้าสู่บรรยากาศ (กม.)",
                                min_value=80.0, max_value=150.0, value=120.0, step=5.0)
    track_hours = st.slider("ระยะเวลาวาดแนวเส้นทาง (ชั่วโมง)", 1, 48, 24)
    track_step = st.select_slider("ช่วงเวลาระหว่างจุดบนแผนที่ (นาที)", options=[1,2,5,10], value=2)
    st.caption("ค่าตั้งต้นระดับอ้างอิง 120 กม. เป็นเกณฑ์วิเคราะห์ ไม่ใช่ขอบเขตการเผาไหม้หรือจุดตก")

raw = ""
if source == "วางข้อมูล / อัปโหลดไฟล์":
    up = st.file_uploader("อัปโหลดไฟล์ TLE (.txt)", type=["txt", "tle", "3le"])
    if up:
        raw = up.getvalue().decode("utf-8", errors="ignore")
    raw = st.text_area("หรือวางชุด TLE ที่นี่", value=raw, height=190,
                       placeholder="ตัวอย่างรูปแบบ:\nชื่อวัตถุ\n1 xxxxxU ...\n2 xxxxx ...")
else:
    norad = st.text_input("หมายเลข NORAD", placeholder="เช่น 25544")
    if st.button("ดึง TLE ล่าสุดจาก CelesTrak", type="primary"):
        if norad.strip().isdigit():
            try:
                url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad.strip()}&FORMAT=TLE"
                resp = requests.get(url, timeout=20, headers={"User-Agent":"Thai-Reentry-Predictor/1.0"})
                resp.raise_for_status()
                raw = resp.text
                st.session_state["fetched_tle"] = raw
                st.session_state["fetched_id"] = norad.strip()
            except Exception as ex:
                st.error(f"ดึงข้อมูลไม่สำเร็จ: {ex}")
        else:
            st.error("กรุณากรอกหมายเลข NORAD เป็นตัวเลข")
    raw = st.session_state.get("fetched_tle", "")
    if raw:
        st.success(f"ได้รับข้อมูล TLE สำหรับ NORAD {st.session_state.get('fetched_id','')} แล้ว")
        st.code(raw, language="text")

records, errors = parse_tle_text(raw) if raw.strip() else ([], [])
if errors:
    with st.expander("รายละเอียดการตรวจสอบข้อมูล"):
        for e in errors: st.write("• " + e)

if not records:
    st.info("เริ่มต้นโดยวาง TLE อย่างน้อย 2 ชุดของวัตถุเดียวกัน หรือดึง TLE ล่าสุดจาก CelesTrak จากแถบด้านซ้าย")
    st.markdown("""
    ### วิธีเตรียมข้อมูล
    1. ดาวน์โหลด TLE หลายชุดของวัตถุเดียวกัน โดยแต่ละชุดต้องมี Epoch ต่างกัน
    2. เรียงข้อมูลตามเวลา Epoch (ระบบจะเรียงให้อัตโนมัติ)
    3. ควรมีข้อมูลอย่างน้อย 4 ชุด และควรมีช่วงเวลาครอบคลุมหลายวัน
    4. สำหรับการวิเคราะห์ช่วงท้าย ควรมี TLE ที่ถี่และใหม่พอ
    """)
    st.stop()

df = pd.DataFrame(records).sort_values("เวลา_epoch").reset_index(drop=True)
norads = df["NORAD"].unique()
if len(norads) > 1:
    st.error("พบ TLE มากกว่าหนึ่งวัตถุในข้อมูลชุดเดียว กรุณาแยกไฟล์ตามหมายเลข NORAD ก่อนวิเคราะห์")
    st.stop()

# สร้างข้อมูลเชิงตารางสำหรับกราฟ
df["ช่วงห่าง_ชั่วโมง"] = df["เวลา_epoch"].diff().dt.total_seconds()/3600
df["การเปลี่ยนความสูง_km"] = df["ความสูงเฉลี่ย_km"].diff()
df["อัตราการเปลี่ยน_kmต่อวัน"] = df["การเปลี่ยนความสูง_km"] / df["ช่วงห่าง_ชั่วโมง"] * 24

st.subheader(f"วัตถุที่กำลังวิเคราะห์: {df.iloc[-1]['ชื่อ']}  |  NORAD {int(df.iloc[-1]['NORAD'])}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("จำนวนชุด TLE", f"{len(df)} ชุด")
c2.metric("ความสูงเฉลี่ยล่าสุด", f"{df.iloc[-1]['ความสูงเฉลี่ย_km']:.2f} กม.")
c3.metric("ความเอียงวงโคจร", f"{df.iloc[-1]['ความเอียง_deg']:.2f}°")
c4.metric("คาบการโคจร", f"{df.iloc[-1]['คาบ_orbit_min']:.1f} นาที")

with st.expander("ตารางข้อมูลวงโคจรที่อ่านได้", expanded=False):
    show = df[["NORAD","เวลา_epoch","กึ่งแกนเอก_km","ความสูงเฉลี่ย_km","ความเยื้องศูนย์กลาง","ความเอียง_deg","คาบ_orbit_min","ช่วงห่าง_ชั่วโมง","การเปลี่ยนความสูง_km"]].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)

st.markdown("### 1. วิเคราะห์การลดระดับวงโคจร")
fig = px.line(df, x="เวลา_epoch", y="ความสูงเฉลี่ย_km", markers=True,
              labels={"เวลา_epoch":"เวลา Epoch (UTC)","ความสูงเฉลี่ย_km":"ความสูงเฉลี่ยจากกึ่งแกนเอก (กม.)"},
              title="แนวโน้มความสูงเฉลี่ยจากข้อมูล TLE")
fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"ระดับอ้างอิง {threshold:.0f} กม.")
fig.update_layout(height=390, margin=dict(l=15,r=15,t=55,b=15))
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(df.iloc[1:], x="เวลา_epoch", y="อัตราการเปลี่ยน_kmต่อวัน",
              labels={"เวลา_epoch":"เวลา Epoch (UTC)","อัตราการเปลี่ยน_kmต่อวัน":"อัตราการเปลี่ยนความสูง (กม./วัน)"},
              title="อัตราการเปลี่ยนแปลงระหว่าง TLE แต่ละชุด")
fig2.add_hline(y=0, line_color="gray")
fig2.update_layout(height=300, margin=dict(l=15,r=15,t=55,b=15))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 2. ตรวจสอบคุณภาพและความต่อเนื่องของ TLE")
gaps = df["ช่วงห่าง_ชั่วโมง"].dropna()
declines = df["การเปลี่ยนความสูง_km"].dropna()
q1,q2,q3 = st.columns(3)
q1.metric("ช่วงห่าง Epoch มัธยฐาน", f"{gaps.median():.1f} ชม." if len(gaps) else "—")
q2.metric("ช่วงที่ความสูงลดลง", f"{int((declines<0).sum())}/{len(declines)}")
q3.metric("ช่วงที่ลดลง ≥10 กม.", f"{int((declines<=-10).sum())}")
if len(gaps) and gaps.median() < 12:
    st.warning("ข้อมูลส่วนใหญ่ห่างกันน้อยกว่า 12 ชั่วโมง งานวิจัยต้นฉบับแนะนำช่วงอย่างน้อย 12 ชั่วโมงเพื่อลดสัญญาณรบกวนของแค็ตตาล็อก ยกเว้นช่วงท้ายที่การลดลง ≥10 กม. อาจเด่นกว่าสัญญาณรบกวน")
if len(declines) and (declines > 5).any():
    st.warning("พบความสูงเฉลี่ยเพิ่มขึ้นบางช่วง อาจเกิดจากสัญญาณรบกวนของ TLE การเปลี่ยนวงโคจร หรือการปฏิบัติการของวัตถุ ต้องตรวจสอบก่อนพยากรณ์")

st.markdown("### 3. ประเมินเวลาถึงระดับอ้างอิง")
forecast = regression_forecast(df, threshold)
if forecast is None:
    st.warning("ต้องมี TLE อย่างน้อย 4 ชุดเพื่อประเมินแนวโน้มช่วงท้าย")
elif forecast["pred"] is None:
    st.error("ยังไม่สามารถคำนวณเวลาถึงระดับอ้างอิงได้: แนวโน้มช่วงท้ายไม่ได้ลดลงอย่างชัดเจน หรือวัตถุอาจอยู่ต่ำกว่าระดับอ้างอิงแล้ว")
else:
    pred = forecast["pred"]
    lo,hi,med = forecast["interval"]
    a,b,c = st.columns(3)
    a.metric("เวลาประเมิน (UTC)", pred.strftime("%d/%m/%Y %H:%M"))
    a.caption((pred+timedelta(hours=7)).strftime("เวลาไทย: %d/%m/%Y %H:%M"))
    b.metric("แนวโน้มช่วงท้าย", f"{forecast['slope']*24:.3f} กม./วัน")
    if np.isfinite(lo) and np.isfinite(hi):
        c.metric("ช่วง Monte Carlo 5–95%", f"{lo:.1f}–{hi:.1f} ชม.")
        c.caption("ช่วงเวลาจาก Epoch ล่าสุด ไม่ใช่ช่วงความเชื่อมั่นที่ผ่านการสอบเทียบ")
    else:
        c.metric("จำนวนตัวอย่างจำลอง", f"{len(forecast['samples'])}")
    if len(forecast["samples"]) < 30:
        st.warning("ตัวอย่าง Monte Carlo ที่ใช้ได้มีน้อย ช่วงความไม่แน่นอนจึงไม่ควรนำไปตีความ")
    st.info("นี่คือการลากแนวโน้มเชิงเส้นของความสูงเฉลี่ยช่วงท้ายไปยังระดับอ้างอิง ไม่ใช่การคำนวณแรงต้านแบบ Segmented M/A เต็มรูปแบบ และไม่ใช่เวลายืนยันการแตกสลายหรือการตกถึงพื้น")
    if len(forecast["samples"]) > 20:
        h = forecast["samples"]
        hfig = px.histogram(x=h, nbins=35, labels={"x":"ชั่วโมงจาก Epoch ล่าสุด","count":"จำนวนครั้ง"},
                            title="การกระจายเวลาประเมินจากการสุ่มความคลาดเคลื่อน")
        hfig.update_layout(height=300)
        st.plotly_chart(hfig, use_container_width=True)

st.markdown("### 4. แนวเส้นทางภาคพื้นดินจาก TLE ล่าสุด")
st.caption("คำนวณตำแหน่งดาวเทียมจาก SGP4 และแปลง TEME เป็นพิกัดโลกโดยประมาณ เพื่อแสดงแนว Ground Track ไม่ใช่แนวเศษซากที่ตกถึงพื้น")
latest_sat = df.iloc[-1]["_sat"]
latest_epoch = df.iloc[-1]["เวลา_epoch"]
track = ground_track(latest_sat, latest_epoch, track_hours, track_step)
if len(track):
    track["ส่วนของเส้น"] = (track["ลองจิจูด"].diff().abs() > 180).cumsum()
    mapfig = px.line_geo(track, lat="ละติจูด", lon="ลองจิจูด", color="ส่วนของเส้น",
                         projection="natural earth", hover_data={"เวลา UTC":True,"ละติจูด":":.2f","ลองจิจูด":":.2f","ส่วนของเส้น":False},
                         title=f"แนว Ground Track {track_hours} ชั่วโมงหลัง Epoch ล่าสุด")
    mapfig.update_geos(showland=True, showcountries=True, showocean=True)
    mapfig.update_layout(height=480, showlegend=False)
    st.plotly_chart(mapfig, use_container_width=True)
else:
    st.warning("SGP4 ไม่สามารถสร้างแนวเส้นทางจาก TLE นี้ได้")

st.markdown("### 5. การประเมินสถานะการพยากรณ์")
last_alt = float(df.iloc[-1]["ความสูงเฉลี่ย_km"])
if len(df) < 4:
    status, detail = "ข้อมูลไม่เพียงพอ", "ต้องมีข้อมูล TLE เพิ่มเติม"
elif last_alt <= threshold:
    status, detail = "ต่ำกว่าระดับอ้างอิง", "วัตถุผ่านระดับอ้างอิงที่กำหนดแล้ว แต่ไม่ได้หมายความว่าทราบจุดตก"
elif len(declines) and (declines < 0).sum() >= max(2, int(len(declines)*0.6)):
    status, detail = "พบแนวโน้มลดระดับ", "มีหลักฐานจากข้อมูล TLE ว่าความสูงเฉลี่ยลดลง ควรติดตาม TLE ใหม่"
else:
    status, detail = "ยังไม่พบแนวโน้มลดระดับชัดเจน", "ข้อมูลปัจจุบันยังไม่รองรับการประเมิน Re-entry อย่างมีความหมาย"
st.markdown(f"**สถานะ:** {status}  \n{detail}")

st.markdown("### 6. สิ่งที่ระบบยังไม่สามารถยืนยัน")
st.markdown("""
- จุดเริ่มต้นการเผาไหม้จริง จุดแตกตัว และตำแหน่งชิ้นส่วนที่รอดถึงพื้น
- พื้นที่ตกของเศษซากหรือความเสี่ยงต่อประชาชน
- การเปลี่ยนวงโคจรจากเครื่องยนต์ที่เกิดขึ้นระหว่าง Epoch ของ TLE
- ค่า M/A เชิงกายภาพที่แท้จริง หากไม่มีมวลและแบบจำลองแรงต้านที่ผ่านการสอบเทียบ
- ความแม่นยำระดับนาทีหรือชั่วโมงสำหรับวัตถุทุกชนิด
""")

csv = df.drop(columns=["_sat","TLE_1","TLE_2"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ ดาวน์โหลดตารางวิเคราะห์ CSV", csv,
                   file_name=f"reentry_analysis_{int(df.iloc[-1]['NORAD'])}.csv",
                   mime="text/csv")
tle_out = "\n".join(sum(([r["TLE_1"],r["TLE_2"]] for r in records), []))
st.download_button("⬇️ ดาวน์โหลด TLE ที่อ่านได้", tle_out,
                   file_name=f"tle_{int(df.iloc[-1]['NORAD'])}.txt", mime="text/plain")

st.divider()
st.markdown("""
**หลักการและแหล่งข้อมูล**
- วิธี Segmented M/A Calibration: Remis, J. (2026), *Predicting Atmospheric Re-entries Using Segmented M/A Calibration of Public TLE Data*, arXiv:2607.20128.
- การคำนวณวงโคจร: SGP4 จากชุดข้อมูล TLE
- แหล่ง TLE ออนไลน์: CelesTrak (เมื่อผู้ใช้เลือกดึงข้อมูล)
- ระดับ 120 กม. เป็นเกณฑ์อ้างอิงที่ผู้ใช้ปรับได้ ไม่ใช่เส้นแบ่งที่วัตถุทุกชิ้นจะเริ่มเผาไหม้ ณ ระดับเดียวกัน
""")
st.caption("Thai Re-entry Predictor V1.0 • ต้นแบบเพื่อการศึกษาและวิจัย • ตรวจสอบวันที่ Epoch และแหล่งที่มาของ TLE ก่อนใช้งานทุกครั้ง")
