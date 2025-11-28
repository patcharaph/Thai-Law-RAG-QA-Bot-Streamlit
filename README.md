# Thai Law RAG QA Bot (OpenRouter Edition)

## โครงสร้างโปรเจ็กต์
```
thai-law-rag/
├── documents/              # ใส่ไฟล์ PDF กฎหมาย
│   └── Business_Collateral_Act_2558.pdf (ตัวอย่าง)
├── vectorstore/            # ChromaDB ที่สร้างอัตโนมัติ (ไม่ต้องแก้ไขเอง)
├── build_kb.py             # สร้างฐานข้อมูลเวกเตอร์จาก PDF
├── app.py                  # เวอร์ชันเว็บ Streamlit
├── qa.py                   # รันแชตถาม-ตอบกฎหมาย
├── requirements.txt        # รายการไลบรารี
└── README.md               # คู่มือการใช้งาน
```

## เตรียมสภาพแวดล้อม
1) Python 3.10+  
2) ติดตั้งไลบรารี:
```bash
pip install -r requirements.txt
```
3) ตั้งค่าไฟล์สิ่งแวดล้อม:
   - คัดลอก `.env.example` ไปเป็น `.env`
   - ใส่ค่า `OPENROUTER_API_KEY=<your_key>`

## การสร้างฐานข้อมูล (ครั้งแรก)
1) วางไฟล์ PDF ไว้ที่ `documents/`
2) รัน:
```bash
python build_kb.py --pdf-dir documents --persist-dir vectorstore
```
จะได้ ChromaDB ในโฟลเดอร์ `vectorstore/`

## การรันเว็บแอป (Streamlit)
1) ตั้งค่า API key ของ OpenRouter ผ่าน `.env` หรือ environment variable (`OPENROUTER_API_KEY`)
2) รัน Streamlit:
```bash
streamlit run app.py
```
3) ตั้งค่าทางซ้าย (Sidebar)
   - เลือกโมเดล (ค่าเริ่มต้น `openai/gpt-4o-mini`, เปลี่ยนเป็น `google/gemini-flash-1.5` ได้)
   - ปรับ Top-K (จำนวนเอกสารอ้างอิง)
   - เปลี่ยนโฟลเดอร์ ChromaDB ได้หากวางฐานไว้ที่อื่น
   - ปุ่ม Clear Chat เพื่อล้างประวัติ
4) กล่องพิมพ์ใช้ `st.chat_input` และประวัติโชว์ด้วย `st.chat_message` (สนทนาแบบหลายตา)

## การรัน QA แชต
1) ตั้งค่า API key ของ OpenRouter:
```bash
set OPENROUTER_API_KEY=your_key_here   # Windows (PowerShell ใช้ $env:OPENROUTER_API_KEY)
```
หรือใช้ไฟล์ `.env` ตามขั้นตอนเตรียมสภาพแวดล้อม
2) รัน:
```bash
python qa.py --persist-dir vectorstore --model openai/gpt-4o-mini
```
หมายเหตุ: สามารถเปลี่ยนโมเดลเป็น `google/gemini-flash-1.5` ได้

## สรุปคำสั่งสำคัญ
- สร้างฐานข้อมูล: `python build_kb.py --pdf-dir documents --persist-dir vectorstore`
- รัน CLI: `python qa.py --persist-dir vectorstore --model openai/gpt-4o-mini`
- รันเว็บแอป Streamlit: `streamlit run app.py`

## รายละเอียดสำคัญ
- ฝั่งฝังข้อมูล: ใช้ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` รองรับภาษาไทยและรันในเครื่อง  
- การตัดข้อความ: `RecursiveCharacterTextSplitter` ขนาด 1000 ตัวอักษร ซ้อน 200 ตัวอักษร ตัวแบ่ง `["\n\n", "มาตรา", "\n", " ", ""]`  
- เวกเตอร์สโตร์: `Chroma` แบบ persistent ใน `vectorstore/`  
- QA: `ConversationalRetrievalChain` พร้อม memory, condense prompt ภาษาไทย, และ QA prompt บังคับอ้างอิงเลขมาตรา ถ้าไม่พบให้ตอบ “ไม่พบข้อมูล”
- เวอร์ชันเว็บ: ใช้ `@st.cache_resource` โหลด Chroma/LLM ครั้งเดียว, เก็บประวัติใน `st.session_state`, แสดงแหล่งอ้างอิงที่ส่วนตอบกลับ
