"""
plate_reader.py
Pembaca teks plat kendaraan pakai PaddleOCR (default GPU kalau paddlepaddle-gpu
terinstal; otomatis fallback ke CPU dengan warning kalau tidak).

Output di-format ke konvensi plat Indonesia: huruf depan + angka + huruf belakang
(contoh: "B 1234 ABC"). Karakter non-alfanumerik dibuang via regex allowlist.

Preprocessing pipeline (di-apply ke crop sebelum OCR):
1. Deskew    -> luruskan plat miring via minAreaRect kontur teks
2. CLAHE     -> contrast equalization adaptif (bantu plat redup/silau)
3. Upscale   -> kalau lebar < 200px, perbesar pakai cubic
4. Unsharp   -> sharpening tipis biar edge huruf lebih tegas
"""

import itertools

import cv2
import numpy as np
from paddleocr import PaddleOCR


# Kode wilayah plat Indonesia yang sah (huruf depan, 1-2 huruf). Dipakai untuk
# memvalidasi/menolak bacaan head yang tak mungkin (mis. 'I' tunggal). Perhatikan
# huruf tunggal C I J O Q U V X Y BUKAN kode yang sah.
VALID_HEAD_CODES = {
    # 1 huruf
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    # 2 huruf (Jawa + Sumatra + Kalimantan + Sulawesi + Bali/Nusra + Maluku/Papua)
    'AA', 'AB', 'AD', 'AE', 'AG',
    'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP',
    'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DR', 'DT', 'DW',
    'EA', 'EB', 'ED',
    'KB', 'KH', 'KT', 'KU',
    'PA', 'PB',
}


def _paddle_has_cuda():
    """Cek build paddle support CUDA. Lazy import + safe fallback."""
    try:
        import paddle
        return bool(paddle.is_compiled_with_cuda())
    except Exception:
        return False


class PlateReader:
    def __init__(self, gpu=True, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                 min_confidence=0.3, preprocess=False,
                 deskew_max_angle=15.0,
                 clahe_clip=2.0, clahe_grid=(8, 8),
                 unsharp_amount=0.5,
                 crop_bottom_ratio=0.0,
                 upscale_width=0,
                 min_height_ratio=0.6,
                 strict_region=True):
        # ponytail: preprocess+crop_bottom default OFF. Ablasi di crop plat
        # nunjukin deskew/CLAHE/unsharp netral-ke-merugikan dan crop_bottom=0.22
        # MOTONG huruf plat (2/8 benar -> 8/8 pas dimatiin). Strip pajak ID
        # ('07-23') tetap dibuang oleh line-clustering di read(). Kalau ketemu
        # plat ID yang strip pajaknya bocor ke OCR, naikin crop_bottom_ratio lagi.
        # Paddle build harus compiled-with-cuda untuk pakai GPU.
        # Kalau user minta gpu tapi build CPU-only, fallback eksplisit ke CPU
        # supaya tidak masuk konfigurasi setengah-jalan yang trigger bug oneDNN.
        if gpu and not _paddle_has_cuda():
            print('[PlateReader] WARNING: paddlepaddle bukan GPU build. '
                  'Install paddlepaddle-gpu untuk akselerasi GPU. '
                  'Sementara fallback ke CPU.')
            gpu = False
        device = 'gpu' if gpu else 'cpu'
        print(f'[PlateReader] Loading PaddleOCR (device={device})...')
        # use_textline_orientation/doc_*: dimatikan, plat 1 baris dan crop sudah lurus.
        # enable_mkldnn=False: hindari bug onednn di Paddle 3.x (PIR runtime).
        self.reader = PaddleOCR(
            lang='en',
            device=device,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )
        self.allowlist = set(allowlist)
        self.min_confidence = min_confidence
        self.preprocess = preprocess
        self.deskew_max_angle = deskew_max_angle
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.unsharp_amount = unsharp_amount
        self.crop_bottom_ratio = max(0.0, min(0.5, crop_bottom_ratio))
        self.upscale_width = max(0, int(upscale_width))  # 0 = native (tanpa upscale)
        # Ambil hanya teks yang tingginya >= ratio ini dari teks tertinggi dalam
        # box. Teks lebih kecil (tanggal pajak) dibuang. Tahan-miring.
        self.min_height_ratio = max(0.1, min(1.0, min_height_ratio))
        # True = tolak bacaan yang huruf depannya bukan kode wilayah sah
        # (kembalikan '' -> tracker menunggu frame yang lebih baik).
        self.strict_region = strict_region
        print(f'[PlateReader] PaddleOCR siap. Preprocess={"ON" if preprocess else "OFF"}, '
              f'crop_bottom={self.crop_bottom_ratio:.0%}')

    def read(self, frame, bbox):
        """
        Baca teks plat dari crop bbox.
        Return string hasil baca, atau '' kalau tidak ada hasil.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return ''

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # LURUSKAN DULU sebelum crop strip pajak. Kalau plat miring, potong-bawah
        # polos jadi diagonal -> motong huruf di satu sisi & nyisain pajak di sisi
        # lain. Deskew bikin strip pajak jadi pita horizontal yang bersih dipotong.
        if self.preprocess or self.crop_bottom_ratio > 0:
            gray = self._deskew(gray)
        if self.preprocess:
            gray = self.clahe.apply(gray)

        # Baru buang strip masa-berlaku-pajak (contoh "07-23"/"04-30"), setelah
        # lurus. Ini yang bikin OCR concat ngawur (D 1886 AP + 04-30 -> ...CL30).
        if self.crop_bottom_ratio > 0:
            keep_h = max(8, int(gray.shape[0] * (1.0 - self.crop_bottom_ratio)))
            gray = gray[:keep_h, :]

        # Upscale HANYA plat yang beneran kecil (kalau diaktifkan). Plat yang
        # sudah >~100px malah rusak kalau diregangkan (artefak interpolasi ->
        # bacaan tak konsisten). Default OFF (upscale_width=0 = native).
        ch, cw = gray.shape[:2]
        if self.upscale_width > 0 and cw < self.upscale_width:
            scale = self.upscale_width / cw
            gray = cv2.resize(gray, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)

        if self.preprocess and self.unsharp_amount > 0:
            gray = self._unsharp(gray, self.unsharp_amount)

        # PaddleOCR butuh 3-channel
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        try:
            results = self.reader.predict(img)
        except Exception:
            return ''

        if not results:
            return ''

        res = results[0]
        texts = res.get('rec_texts') or []
        scores = res.get('rec_scores') or []
        polys = res.get('rec_polys') or []

        if not texts:
            return ''

        # Kumpulkan kandidat dengan posisi vertikal, untuk grouping multi-line.
        kept = []
        for i, txt in enumerate(texts):
            conf = scores[i] if i < len(scores) else 0.0
            if conf < self.min_confidence:
                continue
            poly = polys[i] if i < len(polys) else None
            if poly is not None:
                ys = [p[1] for p in poly]
                xs = [p[0] for p in poly]
                y_center = (min(ys) + max(ys)) / 2
                line_h = max(ys) - min(ys)
                x_min = min(xs)
            else:
                y_center = float(i)
                line_h = 1.0
                x_min = float(i)
            kept.append({
                'y': y_center, 'h': line_h, 'x': x_min, 'text': txt,
            })

        if not kept:
            return ''

        # Seleksi "teks besar menang": ambil hanya potongan yang tingginya >=
        # min_height_ratio dari potongan TERTINGGI. Ini membuang tanggal pajak
        # (font kecil) apa pun posisinya -> tahan plat miring, tak seperti
        # clustering by-y yang gampang gagal saat plat mereng.
        max_h = max(r['h'] for r in kept)
        big = [r for r in kept if r['h'] >= max_h * self.min_height_ratio]
        big.sort(key=lambda r: r['x'])  # urut kiri ke kanan

        # Bangun token berurutan kiri->kanan. PaddleOCR sering memisah plat di
        # batas zona (region terpisah, atau spasi literal di dalam teks) -> pakai
        # itu sebagai penanda zona AA|1234|XXX, jangan dibuang.
        tokens = []
        for r in big:
            for sub in str(r['text']).upper().split():
                sub = ''.join(c for c in sub if c in self.allowlist)
                if sub:
                    tokens.append(sub)
        if not tokens:
            return ''

        return self._finalize(tokens, self.strict_region)

    def _deskew(self, gray):
        """Deteksi sudut miring plat via minAreaRect dari piksel teks (Otsu)."""
        # Threshold inverted: teks gelap di plat terang -> jadi foreground putih
        _, thr = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thr > 0))
        if coords.shape[0] < 50:
            return gray  # terlalu sedikit piksel, skip

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        # OpenCV minAreaRect angle convention: [-90, 0)
        # Normalisasi ke "kemiringan dari horizontal".
        if angle < -45:
            angle = 90 + angle

        # Hindari over-rotate kalau deteksi noise
        if abs(angle) < 0.5 or abs(angle) > self.deskew_max_angle:
            return gray

        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def _unsharp(gray, amount):
        """Unsharp mask: gray + amount * (gray - blur(gray))."""
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
        return sharp

    # Peta koreksi confusion OCR per-zona. Di zona huruf, karakter yang kebaca
    # angka tapi bentuknya mirip huruf dikembalikan ke huruf; di zona angka,
    # sebaliknya. Ini yang "memaksa" hasil baca ikut template Indonesia.
    _DIGIT_TO_ALPHA = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A',
                       '5': 'S', '6': 'G', '8': 'B'}
    _ALPHA_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
                       'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}

    # Kandidat HURUF untuk tiap DIGIT saat di zona head (huruf depan). Satu digit
    # bisa punya beberapa kandidat huruf; dicari kombinasi yang jadi kode sah.
    _HEAD_DIGIT_CAND = {'0': 'ODQ', '1': 'ILT', '2': 'Z', '4': 'A',
                        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'}

    @classmethod
    def _to_alpha(cls, c):
        return cls._DIGIT_TO_ALPHA.get(c, c)

    @classmethod
    def _to_digit(cls, c):
        return cls._ALPHA_TO_DIGIT.get(c, c)

    @classmethod
    def _parse_plate(cls, s):
        """
        Pas-kan string alfanumerik ke template plat Indonesia:
        [1-2 huruf][1-4 angka][1-3 huruf]. Coba semua pembagian batas yang
        valid, pilih yang paling banyak karakternya sudah sesuai tipe zona,
        lalu koreksi sisanya via peta confusion. Return (head, num, tail)
        atau None kalau panjang di luar 3..9.
        """
        n = len(s)
        if n < 3 or n > 9:
            return None
        # Tanpa satu pun angka, ini kemungkinan bukan plat kebaca — jangan
        # paksa pecah jadi template (hindari "XYZ" -> "X Y Z").
        if not any(c.isdigit() for c in s):
            return None

        best = None
        best_score = -1
        for head_len in (1, 2):
            for tail_len in (1, 2, 3):
                num_len = n - head_len - tail_len
                if num_len < 1 or num_len > 4:
                    continue
                head = s[:head_len]
                num = s[head_len:head_len + num_len]
                tail = s[head_len + num_len:]
                # Skor = jumlah karakter yang sudah cocok tipe zonanya.
                score = (sum(c.isalpha() for c in head)
                         + sum(c.isdigit() for c in num)
                         + sum(c.isalpha() for c in tail))
                if score > best_score:
                    best_score = score
                    best = (head, num, tail)

        if best is None:
            return None

        head, num, tail = best
        head = ''.join(cls._to_alpha(c) for c in head)
        num = ''.join(cls._to_digit(c) for c in num)
        tail = ''.join(cls._to_alpha(c) for c in tail)
        return head, num, tail

    @classmethod
    def _format_indonesian(cls, s):
        """Paksa ke template 'XX 1234 YYY' (huruf-angka-huruf). Kalau panjang
        di luar batas template, return raw apa adanya."""
        parsed = cls._parse_plate(s)
        if parsed:
            head, num, tail = parsed
            return f'{head} {num} {tail}'
        return s

    @classmethod
    def _split_by_gaps(cls, tokens):
        """Tentukan zona head|num|tail dari token yang sudah dipisah OCR (via
        region terpisah / spasi). Token angka = yang paling banyak digit; sebelum
        = head (raw, belum divalidasi), sesudah = tail. Return (head, num, tail)
        atau None kalau tak ada struktur yang jelas (caller fallback ke
        _parse_plate kombinatorik)."""
        tokens = [t for t in tokens if t]
        if len(tokens) < 2:
            return None
        digit_counts = [sum(c.isdigit() for c in t) for t in tokens]
        if max(digit_counts) == 0:
            return None
        num_idx = digit_counts.index(max(digit_counts))
        head = ''.join(tokens[:num_idx])
        num = ''.join(cls._to_digit(c) for c in tokens[num_idx])
        tail = ''.join(cls._to_alpha(c) for c in ''.join(tokens[num_idx + 1:]))
        if not head or not tail:
            return None
        return head, num, tail

    @classmethod
    def _finalize(cls, tokens, strict):
        """Token -> teks plat final. Coba split via gap dulu, fallback ke tebak
        kombinatorik. Validasi kode wilayah berlaku di KEDUA jalur (dulu cuma di
        jalur gap -> bacaan macam '1963 SSJ' lolos jadi 'I 963 SSJ' via fallback).
        Return '' kalau head tak sah & strict (tolak; tunggu frame lebih baik)."""
        parsed = cls._split_by_gaps(tokens)
        if parsed is None:
            formatted = cls._format_indonesian(''.join(tokens))
            parts = formatted.split()
            if len(parts) != 3:
                return formatted  # bukan template plat -> apa adanya (biar settle-gate yang nyaring)
            head, num, tail = parts
        else:
            head, num, tail = parsed
        head, ok = cls._correct_region_code(head)
        if not ok and strict:
            return ''
        return f'{head} {num} {tail}'

    @classmethod
    def _region_candidates(cls, head):
        """Semua kombinasi huruf yang mungkin dari head (digit -> kandidat huruf,
        huruf tetap). Untuk head 1-2 karakter jumlahnya kecil."""
        per_char = []
        for ch in head:
            per_char.append(cls._HEAD_DIGIT_CAND.get(ch, ch) if ch.isdigit() else ch)
        return [''.join(p) for p in itertools.product(*per_char)]

    @classmethod
    def _correct_region_code(cls, head):
        """Validasi/koreksi kode wilayah (huruf depan). Return (head_out, ok).
        ok=False artinya head bukan kode sah dan tak bisa dikoreksi deterministik
        (mis. 'I' tunggal) -> caller boleh menolak bacaan kalau strict."""
        head = head.upper()
        if head in VALID_HEAD_CODES:
            return head, True
        for cand in cls._region_candidates(head):
            if cand in VALID_HEAD_CODES:
                return cand, True
        # tak ada kandidat sah: kembalikan versi huruf apa adanya (untuk mode
        # non-strict), tandai tidak valid.
        return ''.join(cls._to_alpha(c) for c in head), False


def _demo():
    """Self-check fungsi murni (tanpa memuat PaddleOCR). Jalankan:
    venv\\Scripts\\python.exe src\\plate_reader.py"""
    assert PlateReader._split_by_gaps(['D', '1666', 'XYZ']) == ('D', '1666', 'XYZ')
    assert PlateReader._split_by_gaps(['T', '1598', 'EO']) == ('T', '1598', 'EO')
    assert PlateReader._split_by_gaps(['HWSIVSU']) is None
    assert PlateReader._split_by_gaps(['NAI3', 'NRU']) is None
    assert PlateReader._correct_region_code('D') == ('D', True)
    assert PlateReader._correct_region_code('I')[1] is False
    fixed, ok = PlateReader._correct_region_code('0')
    assert ok and fixed in VALID_HEAD_CODES, (fixed, ok)
    assert PlateReader._correct_region_code('B0') == ('BD', True)  # 0->D via kandidat
    print('plate_reader self-check OK')


if __name__ == '__main__':
    _demo()
