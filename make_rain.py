"""Official KNMI radar_forecast/2.0 -> Apple Watch PNG and ComplicaJSON JSON."""
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import requests
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

BASE = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/2.0/files"
LON, LAT = 5.0019, 52.0975  # Vlambloemlaan (approximate street midpoint), Vleuterweide
TZ = ZoneInfo("Europe/Amsterdam")
OUT = Path("docs")
OUT.mkdir(exist_ok=True)
API_KEY = os.environ["KNMI_API_KEY"]
SESSION = requests.Session()
SESSION.headers.update({"Authorization": API_KEY})


def get(url, **kwargs):
    r = SESSION.get(url, timeout=45, **kwargs)
    r.raise_for_status()
    return r


def scalar(v):
    return v[0] if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else v


def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def font(size, bold=False):
    candidates = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                  ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for name in candidates:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fetch_values():
    response = get(BASE, params={"maxKeys": 1, "orderBy": "created", "sorting": "desc"}).json()
    files = response.get("files", [])
    if not files:
        raise RuntimeError("KNMI returned no forecast files")
    latest = files[0]
    created = datetime.fromisoformat(latest["created"])
    if created.tzinfo is None:
        raise RuntimeError("KNMI creation time has no timezone")
    age = datetime.now(timezone.utc) - created
    if not timedelta(minutes=-5) <= age <= timedelta(minutes=60):
        raise RuntimeError(f"KNMI file is not current: {age}")

    file_name = latest["filename"]
    download_info = get(f"{BASE}/{file_name}/url").json()
    download_url = download_info["temporaryDownloadUrl"]
    # Do not print URL; it is temporary and may be signed.
    download_response = requests.get(download_url, timeout=60)
    download_response.raise_for_status()
    values = []
    with h5py.File(io.BytesIO(download_response.content), "r") as f:
        geo = f["geographic"]
        proj4 = decode(f["geographic/map_projection"].attrs["projection_proj4_params"])
        x, y = Transformer.from_crs("EPSG:4326", proj4, always_xy=True).transform(LON, LAT)
        col_offset = float(scalar(geo.attrs["geo_column_offset"]))
        row_offset = float(scalar(geo.attrs["geo_row_offset"]))
        dx = float(scalar(geo.attrs["geo_pixel_size_x"]))
        dy = float(scalar(geo.attrs["geo_pixel_size_y"]))
        # KNMI's north polar stereographic y points in the opposite direction
        # from the image row coordinate. The row offset is stored in km.
        col = round((x - col_offset) / dx)
        row = round((-y - row_offset) / abs(dy))
        names = sorted((k for k in f if re.fullmatch(r"image\d+", k)), key=lambda k: int(k[5:]))
        if len(names) < 20:
            raise RuntimeError(f"Too few forecast frames: {len(names)}")
        for name in names:
            image = f[name]
            data = image["image_data"]
            if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
                raise RuntimeError(f"Vleuterweide outside radar grid: {row}, {col}")
            raw = int(data[row, col])
            calibration = image["calibration"].attrs
            missing = int(scalar(calibration["calibration_missing_data"]))
            outside = int(scalar(calibration["calibration_out_of_image"]))
            if raw in (missing, outside):
                mm = None
            else:
                formula = decode(calibration["calibration_formulas"])
                m = re.fullmatch(r"GEO\s*=\s*([+-]?[\d.]+)\s*\*\s*PV\s*\+\s*([+-]?[\d.]+)", formula)
                if not m:
                    raise RuntimeError(f"Unknown calibration: {formula}")
                mm = float(m.group(1)) * raw + float(m.group(2))
            time_string = decode(image.attrs["image_datetime_valid"])
            valid_utc = datetime.strptime(time_string, "%d-%b-%Y;%H:%M:%S.%f").replace(tzinfo=timezone.utc)
            values.append((valid_utc, mm))
    print(f"File: {file_name}; grid: {row},{col}; frames: {len(values)}")
    return created, values


def render(created, values):
    # Render at 2x for a crisp, small wide complication.
    W, H = 640, 160
    img = Image.new("RGB", (W, H), (8, 14, 24))
    d = ImageDraw.Draw(img)
    white = (241, 247, 255)
    muted = (161, 182, 202)
    blue = (54, 159, 249)
    d.text((16, 8), "VLEUTERWEIDE  ·  KNMI", font=font(21, True), fill=white)
    update = created.astimezone(TZ).strftime("%H:%M")
    d.text((W - 16, 12), f"KNMI {update}", font=font(17), fill=muted, anchor="ra")
    left, right, top, bottom = 18, 622, 42, 124
    d.line((left, bottom, right, bottom), fill=(66, 80, 100), width=1)
    finite = [v for _, v in values if v is not None]
    if not finite:
        raise RuntimeError("No valid rainfall samples for Vlambloemlaan")
    ymax = max(0.1, max(finite) * 1.1)
    n = len(values)
    pitch = (right - left) / n
    for i, (_, mm) in enumerate(values):
        x0 = round(left + i * pitch + 2)
        x1 = round(left + (i + 1) * pitch - 2)
        if mm is None:
            d.line((x0, bottom - 4, x1, bottom - 4), fill=(245, 184, 75), width=2)
            continue
        height = max(0, round(mm / ymax * (bottom - top)))
        if mm > 0:
            height = max(2, height)
        if height:
            d.rounded_rectangle((x0, bottom - height, x1, bottom), radius=3, fill=blue)
    d.text((left, 131), "NU", font=font(17, True), fill=muted)
    d.text((W // 2, 131), "+1 UUR", font=font(17), fill=muted, anchor="ma")
    d.text((right, 131), "+2 UUR", font=font(17), fill=muted, anchor="ra")
    img.save(OUT / "rain.png", optimize=True)

def rain_text(values, created):
    """24 regenposities van elk 5 minuten, met begintijd erboven."""
    from datetime import datetime, timedelta

    now = datetime.now(created.tzinfo)
    elapsed = max(0, int((now - created).total_seconds() // 300))

    future = list(values[elapsed:elapsed + 24])
    future += [None] * (24 - len(future))

    def symbol(mm):
        if mm is None:
            return "·"
        mm_per_hour = max(0, mm) * 12
        if mm_per_hour < 0.1:
            return "•"
        levels = "▁▂▃▄▅▆▇█"
        thresholds = [0.5, 1, 2, 4, 6, 10, 20]
        level = sum(mm_per_hour >= t for t in thresholds)
        return levels[level]

    bars = "".join(symbol(mm) for mm in future)

    first_rain = next(
        (i for i, mm in enumerate(future)
         if mm is not None and mm * 12 >= 0.1),
        None
    )

    if first_rain is not None:
        starts = created + timedelta(minutes=5 * (elapsed + first_rain))
        title = "Regen " + starts.astimezone(TZ).strftime("%H:%M")
    elif any(mm is not None for mm in future):
        title = "Droog"
    else:
        title = "Geen gegevens"

    return title + "\n" + bars

def main():
    created, values = fetch_values()
    render(created, values)
    # A plain top-level key can be selected as 'Image URL' in ComplicaJSON.
    # Use a stable URL with a changing query value to discourage image caching.
    stamp = int(created.timestamp())
    url = f"https://lighthub-2027.github.io/-knmi-watch/rain.png?v={stamp}"
    payload = {
        "image_url": url,
        "regenbalkjes": rain_text(values, created),
        "location": "Vlambloemlaan, Vleuterweide",
        "source": "KNMI radar_forecast/2.0",
        "updated_at": created.isoformat(),
        "updated_local": created.astimezone(TZ).strftime("%d-%m-%Y %H:%M %Z"),
    }
    (OUT / "complicajson.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>KNMI regen Vleuterweide</title><style>body{background:#080e18;color:#f1f7ff;font:16px system-ui;max-width:720px;margin:40px auto;padding:20px}img{max-width:100%;height:auto}</style>'
        f'<h1>KNMI regenverwachting Vleuterweide</h1><p>KNMI-bestand: {created.astimezone(TZ).strftime("%d-%m-%Y %H:%M %Z")}</p>'
        f'<img src="rain.png?v={stamp}" alt="KNMI regenverwachting voor de komende twee uur">'
        '<p>Let op: controleer altijd het tijdstip van de laatste update. Bij een mislukte update kan deze pagina verouderd zijn.</p></html>\n',
        encoding="utf-8"
    )
    print("Created docs/rain.png, docs/complicajson.json, docs/index.html")


if __name__ == "__main__":
    main()
