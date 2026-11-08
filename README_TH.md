# Thai Re-entry Predictor V1.0

เว็บแอปภาษาไทยสำหรับวิเคราะห์การลดระดับวงโคจรจากข้อมูล TLE

## ติดตั้ง
ต้องมี Python 3.10 ขึ้นไป

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

เปิด URL ที่ Streamlit แสดง (โดยปกติ http://localhost:8501)

## วิธีใช้
1. เลือกวาง/อัปโหลด TLE หรือดึง TLE ล่าสุดจาก CelesTrak
2. ใส่ TLE หลาย epoch ของ NORAD เดียวกันเพื่อวิเคราะห์การเปลี่ยนแปลงตามเวลา
3. ตรวจสอบความสูงเฉลี่ย กราฟ อัตราการเปลี่ยน และแนว Ground Track
4. อ่านเวลาประเมินอย่างระมัดระวัง โดยเฉพาะเมื่อมีการเปลี่ยนวงโคจรหรือข้อมูลไม่ต่อเนื่อง

## ข้อจำกัดสำคัญ
- เวลาที่ประเมินเป็นการ extrapolate แนวโน้มเชิงเส้นของความสูงเฉลี่ยจาก TLE ช่วงท้าย ไม่ใช่ผล GMAT/NRLMSISE-00
- ยังไม่ได้ทำ Segmented M/A calibration แบบเต็มตามบทความ เพราะต้องมีการ propagate force model และปรับ effective area ด้วย GMAT หรือแบบจำลองแรงต้านที่ผ่านการตรวจสอบ
- ช่วง Monte Carlo เป็นการสุ่ม residual เชิงสถิติอย่างง่าย ไม่ใช่ probability of impact และไม่ใช่ช่วงความเชื่อมั่นที่ผ่านการ validate
- Ground track เป็นแนวผ่านของวัตถุ ไม่ใช่แนวตกของเศษซาก
- ห้ามใช้แทนคำเตือนภัยหรือการตัดสินใจด้านความปลอดภัย

## แหล่งอ้างอิง
Remis, J. (2026). Predicting Atmospheric Re-entries Using Segmented M/A Calibration of Public TLE Data. arXiv:2607.20128.
