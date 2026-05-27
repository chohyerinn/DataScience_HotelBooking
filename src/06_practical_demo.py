"""
Practical demo web application for hotel booking cancellation prediction

This script is an optional application demo. It uses the same final selected
model configuration from the model comparison step
-> StandardScaler + OneHotEncoder + Logistic Regression(C=1.0, class_weight="balanced")

The script trains the pipeline on the cleaned dataset
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Limit the number of CPU workers used by joblib/scikit-learn.
# This helps avoid unnecessary CPU warnings or resource issues on some PCs.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_utils import load_cleaned_data

HOST = "127.0.0.1"

PORT = int(os.environ.get("PORT", "8502"))


class CancellationPredictor:
    """Train and use the final hotel booking cancellation prediction model."""

    def __init__(self):
        """Load cleaned data, build the preprocessing pipeline, and train the model."""

        data = load_cleaned_data()
        self.X = data.drop(columns="is_canceled")
        y = data["is_canceled"]
        categorical = self.X.select_dtypes(include=["object", "string"]).columns.tolist()
        numerical = self.X.select_dtypes(include=["number"]).columns.tolist()

        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numerical,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )

        self.model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        # Fit the full preprocessing + model pipeline.
        self.model.fit(self.X, y)

    def predict(self, values):
        row = {}
        for column in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[column]):
                row[column] = self.X[column].median()
            else:
                row[column] = self.X[column].mode().iloc[0]

        row.update(values)

        row["total_guests"] = row["adults"] + row["children"] + row["babies"]
        row["total_stays"] = (
            row["stays_in_weekend_nights"] + row["stays_in_week_nights"]
        )
        row["is_family"] = int(row["children"] > 0 or row["babies"] > 0)

        if row["total_guests"] <= 0:
            raise ValueError("투숙객은 1명 이상이어야 해요.")
        if row["total_stays"] <= 0:
            raise ValueError("숙박 일수는 1박 이상이어야 해요.")
        if row["adr"] < 0:
            raise ValueError("하루 평균 숙박 요금은 0보다 작을 수 없어요.")

        booking = pd.DataFrame([row], columns=self.X.columns)

        probability = float(self.model.predict_proba(booking)[0, 1])
        if probability >= 0.70:
            return probability, "주의 필요", "취소 가능성이 높습니다. 예약 확정 여부를 미리 확인해 보세요."
        if probability >= 0.40:
            return probability, "확인 권장", "취소 가능성이 중간 정도입니다. 객실 운영 시 참고해 주세요."
        return probability, "안정적", "취소 가능성이 낮은 예약입니다."


# HTML, CSS, and JavaScript for the local demo page
HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>호텔 예약 취소 예측</title>
  <style>
    /* Basic color variables used throughout the demo UI. */
    :root {
      --blue: #3182f6;
      --blue-soft: #edf4ff;
      --text: #191f28;
      --sub: #6b7684;
      --line: #e5e8eb;
      --bg: #f7f8fa;
      --red: #f04452;
      --orange: #f59f00;
      --green: #00a878;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", Arial, sans-serif;
    }
    .wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 44px; }
    h1 { font-size: 34px; letter-spacing: -.06em; margin: 30px 0 9px; line-height: 1.2; }
    .intro { color: var(--sub); font-size: 16px; margin: 0 0 30px; }
    .layout { display: grid; grid-template-columns: 1.08fr .92fr; gap: 18px; align-items: start; }
    .card { background: #fff; border-radius: 26px; padding: 26px; }
    .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    label { display: block; color: #6b7684; font-size: 13px; margin-bottom: 7px; }
    input {
      width: 100%; height: 50px; border: 1px solid var(--line); border-radius: 13px;
      background: #fff; padding: 0 13px; font-size: 15px; color: var(--text); outline: 0;
    }
    input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(49,130,246,.12); }
    /* Custom dropdown UI used instead of the browser default select box. */
    .dropdown { position: relative; }
    .dropdown-toggle {
      width: 100%; height: 50px; border: 1px solid var(--line); border-radius: 14px;
      background: rgba(255,255,255,.72); backdrop-filter: blur(10px);
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 13px; color: var(--text); font-size: 15px; cursor: pointer;
    }
    .dropdown-toggle::after {
      content: ""; width: 8px; height: 8px; border-right: 2px solid #8b95a1;
      border-bottom: 2px solid #8b95a1; transform: rotate(45deg); margin: -4px 4px 0 12px;
    }
    .dropdown.open .dropdown-toggle {
      border-color: #b9d4ff; box-shadow: 0 0 0 3px rgba(49,130,246,.11);
    }
    .dropdown.open .dropdown-toggle::after { transform: translateY(4px) rotate(225deg); }
    .dropdown-menu {
      position: absolute; left: 0; right: 0; top: calc(100% + 8px); z-index: 20;
      display: none; padding: 7px; border: 1px solid rgba(228,232,237,.9);
      border-radius: 17px; background: rgba(255,255,255,.86); backdrop-filter: blur(18px);
      box-shadow: 0 16px 36px rgba(15,23,42,.11);
    }
    .dropdown.open .dropdown-menu { display: block; }
    .dropdown-option {
      width: 100%; border: 0; background: transparent; border-radius: 11px;
      padding: 11px 12px; text-align: left; color: var(--text); font-size: 15px; cursor: pointer;
    }
    .dropdown-option:hover { background: #f2f7ff; color: var(--blue); }
    .dropdown-option.selected { background: var(--blue-soft); color: var(--blue); font-weight: 600; }
    details { margin-top: 20px; border-top: 1px solid #f2f4f6; padding-top: 17px; }
    summary { color: #4e5968; font-size: 14px; cursor: pointer; margin-bottom: 15px; }
    .more { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .submit {
      margin-top: 24px; width: 100%; height: 54px; border: 0; border-radius: 15px;
      background: var(--blue); color: white; font-size: 16px; font-weight: 700; cursor: pointer;
    }
    .submit:hover { background: #1b64da; }
    .result { min-height: 456px; display: flex; flex-direction: column; }
    .placeholder { margin: auto 0; text-align: center; color: #8b95a1; line-height: 1.65; }
    .placeholder .circle {
      margin: 0 auto 18px; width: 48px; height: 48px; border-radius: 50%;
      background: var(--blue-soft); color: var(--blue); display: grid; place-items: center;
      font-size: 25px; font-weight: 700;
    }
    .output { display: none; }
    .output.on { display: block; }
    .caption { margin: 0; font-size: 14px; color: var(--sub); }
    .risk-tag {
      margin: 20px 0 15px; display: inline-flex; align-items: center;
      padding: 8px 13px; font-size: 14px; font-weight: 700; border-radius: 18px;
    }
    .number { font-size: 64px; line-height: 1; font-weight: 700; letter-spacing: -.07em; }
    .number span { font-size: 28px; margin-left: 3px; }
    .bar { margin: 24px 0 25px; height: 8px; background: #f2f4f6; border-radius: 8px; overflow: hidden; }
    .fill { height: 100%; width: 0; border-radius: inherit; transition: width .5s ease; }
    .message { font-size: 16px; line-height: 1.55; margin: 0 0 30px; }
    .tip {
      margin-top: auto; padding: 16px; background: #f7f8fa; border-radius: 15px;
      color: #6b7684; font-size: 13px; line-height: 1.55;
    }
    .error { color: var(--red); font-size: 14px; margin: 12px 0 0; display: none; }
    /* Responsive layout for tablet and mobile screens. */
    @media (max-width: 790px) {
      .layout { grid-template-columns: 1fr; }
      .result { min-height: 340px; }
    }
    @media (max-width: 510px) {
      .wrap { padding: 20px 15px 30px; }
      h1 { font-size: 29px; }
      .card { padding: 21px 18px; border-radius: 22px; }
      .fields, .more { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>호텔 예약 취소 예측</h1>
    <p class="intro">알고 있는 예약 정보를 입력하면 취소 가능성을 확인할 수 있어요.</p>
    <main class="layout">
      <form id="form" class="card">
        <div class="fields">
          <div><label>호텔 종류</label><div class="dropdown" data-name="hotel"><input type="hidden" name="hotel" value="Resort Hotel"><button class="dropdown-toggle" type="button">Resort Hotel</button><div class="dropdown-menu"><button type="button" class="dropdown-option selected" data-value="Resort Hotel">Resort Hotel</button><button type="button" class="dropdown-option" data-value="City Hotel">City Hotel</button></div></div></div>
          <div><label>체크인까지 남은 날짜 (일)</label><input name="lead_time" type="number" min="0" value="5" placeholder="일"></div>
          <div><label>하루 평균 숙박 요금 (EUR)</label><input name="adr" type="number" min="0" value="90"></div>
          <div><label>요청 사항 개수</label><input name="total_of_special_requests" type="number" min="0" value="3"></div>
          <div><label>예약금 / 환불 조건</label><div class="dropdown" data-name="deposit_type"><input type="hidden" name="deposit_type" value="No Deposit"><button class="dropdown-toggle" type="button">예약금 없음</button><div class="dropdown-menu"><button type="button" class="dropdown-option selected" data-value="No Deposit">예약금 없음</button><button type="button" class="dropdown-option" data-value="Non Refund">환불 불가</button><button type="button" class="dropdown-option" data-value="Refundable">환불 가능</button></div></div></div>
          <div><label>예약 방식</label><div class="dropdown" data-name="market_segment"><input type="hidden" name="market_segment" value="Direct"><button class="dropdown-toggle" type="button">직접 예약</button><div class="dropdown-menu"><button type="button" class="dropdown-option selected" data-value="Direct">직접 예약</button><button type="button" class="dropdown-option" data-value="Online TA">온라인 여행사</button><button type="button" class="dropdown-option" data-value="Offline TA/TO">오프라인 여행사</button><button type="button" class="dropdown-option" data-value="Groups">단체 예약</button><button type="button" class="dropdown-option" data-value="Corporate">기업 예약</button></div></div></div>
        </div>
        <details>
          <summary>추가 정보 입력하기</summary>
          <div class="more">
            <div><label>주중 숙박 수 (박)</label><input name="stays_in_week_nights" type="number" min="0" value="2"></div>
            <div><label>주말 숙박 수 (박)</label><input name="stays_in_weekend_nights" type="number" min="0" value="1"></div>
            <div><label>성인 인원</label><input name="adults" type="number" min="0" value="2"></div>
            <div><label>어린이 인원</label><input name="children" type="number" min="0" value="0"></div>
            <div><label>예약 변경 횟수</label><input name="booking_changes" type="number" min="0" value="1"></div>
            <div><label>예약 고객 유형</label><div class="dropdown" data-name="customer_type"><input type="hidden" name="customer_type" value="Transient"><button class="dropdown-toggle" type="button">일반 고객</button><div class="dropdown-menu"><button type="button" class="dropdown-option selected" data-value="Transient">일반 고객</button><button type="button" class="dropdown-option" data-value="Contract">계약 고객</button><button type="button" class="dropdown-option" data-value="Group">단체 고객</button><button type="button" class="dropdown-option" data-value="Transient-Party">동행 고객</button></div></div></div>
          </div>
        </details>
        <input type="hidden" name="babies" value="0">
        <button class="submit" type="submit">취소 가능성 계산하기</button>
        <p id="error" class="error"></p>
      </form>
      <section class="card result">
        <div id="placeholder" class="placeholder">
          <div class="circle">?</div>
          아직 예측 결과가 없어요.<br>왼쪽에 예약 정보를 입력해 주세요.
        </div>
        <div id="output" class="output">
          <p class="caption">예측된 취소 가능성</p>
          <div id="risk-tag" class="risk-tag"></div>
          <div class="number"><strong id="score">0.0</strong><span>%</span></div>
          <div class="bar"><div id="fill" class="fill"></div></div>
          <p id="message" class="message"></p>
          <div class="tip">입력 화면에 없는 항목은 데이터의 일반적인 값으로 계산합니다.</div>
        </div>
      </section>
    </main>
  </div>
  <script>
    // Form element that sends booking inputs to the Python server.
    const form = document.getElementById("form");
    // These fields must be converted from strings to numbers before sending.
    const numberFields = ["lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "booking_changes", "total_of_special_requests"];
    const labels = {
      hotel: {"Resort Hotel": "Resort Hotel", "City Hotel": "City Hotel"},
      deposit_type: {"No Deposit": "예약금 없음", "Non Refund": "환불 불가", "Refundable": "환불 가능"},
      market_segment: {"Direct": "직접 예약", "Online TA": "온라인 여행사", "Offline TA/TO": "오프라인 여행사", "Groups": "단체 예약", "Corporate": "기업 예약"},
      customer_type: {"Transient": "일반 고객", "Contract": "계약 고객", "Group": "단체 고객", "Transient-Party": "동행 고객"}
    };
    // Update hidden input values and visible dropdown labels.
    function setDropdown(name, value) {
      const dropdown = document.querySelector(`.dropdown[data-name="${name}"]`);
      if (!dropdown) return;
      dropdown.querySelector("input").value = value;
      dropdown.querySelector(".dropdown-toggle").textContent = labels[name]?.[value] || value;
      dropdown.querySelectorAll(".dropdown-option").forEach(option => {
        option.classList.toggle("selected", option.dataset.value === value);
      });
    }
    document.querySelectorAll(".dropdown").forEach(dropdown => {
      dropdown.querySelector(".dropdown-toggle").addEventListener("click", () => {
        document.querySelectorAll(".dropdown.open").forEach(open => {
          if (open !== dropdown) open.classList.remove("open");
        });
        dropdown.classList.toggle("open");
      });
      dropdown.querySelectorAll(".dropdown-option").forEach(option => {
        option.addEventListener("click", () => {
          setDropdown(dropdown.dataset.name, option.dataset.value);
          dropdown.classList.remove("open");
        });
      });
    });
    document.addEventListener("click", event => {
      if (!event.target.closest(".dropdown")) {
        document.querySelectorAll(".dropdown.open").forEach(dropdown => dropdown.classList.remove("open"));
      }
    });
    // Send form input to the /predict API endpoint and display the result.
    async function calculate(event) {
      if (event) event.preventDefault();
      const values = Object.fromEntries(new FormData(form).entries());
      numberFields.forEach(key => values[key] = Number(values[key]));
      const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(values)
      });
      const result = await response.json();
      const error = document.getElementById("error");
      if (!response.ok) {
        error.textContent = result.error;
        error.style.display = "block";
        return;
      }
      error.style.display = "none";
      const color = result.risk === "주의 필요" ? "#f04452" : result.risk === "확인 권장" ? "#f59f00" : "#00a878";
      document.getElementById("placeholder").style.display = "none";
      document.getElementById("output").classList.add("on");
      document.getElementById("risk-tag").textContent = result.risk;
      document.getElementById("risk-tag").style.cssText = `color:${color};background:${color}16;`;
      document.getElementById("score").textContent = result.percent.toFixed(1);
      document.getElementById("fill").style.cssText = `width:${result.percent}%;background:${color};`;
      document.getElementById("message").textContent = result.message;
    }
    form.addEventListener("submit", calculate);
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local prediction web app."""

    # Shared trained model object
    # It is initialized once in main() and reused for all requests
    predictor = None

    def do_GET(self):
        """Serve the main HTML page."""

        if self.path != "/":
            self.send_error(404)
            return
        content = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        """Receive booking input, run prediction, and return JSON response."""

        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            # Read and parse JSON request body from the browser
            length = int(self.headers.get("Content-Length", "0"))
            values = json.loads(self.rfile.read(length))
            probability, risk, message = self.predictor.predict(values)
            self.send_json(
                {
                    "probability": probability,
                    "percent": probability * 100,
                    "risk": risk,
                    "message": message,
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, status=400)

    def send_json(self, payload, status=200):
        """Send a JSON response to the browser."""
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Disable default HTTP request logs to keep the terminal output clean
        return


def main():
    """Train the model and start the local HTTP server."""

    print("Training the final cancellation prediction model...")
    # Train the prediction model once before starting the server
    AppHandler.predictor = CancellationPredictor()

    # ThreadingHTTPServer can handle multiple browser requests concurrently
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Reservation prediction web app running at http://{HOST}:{PORT}")
    try:
        # Keep the local web server running until the user stops it
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
