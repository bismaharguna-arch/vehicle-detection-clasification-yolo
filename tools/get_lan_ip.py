"""
get_lan_ip.py
Print IP LAN aktif (bukan 127.0.0.1) supaya gampang dipakai di simulasi.

Pakai trik UDP socket: koneksi dummy ke IP eksternal -> OS pilih interface
LAN, kita ambil source IP-nya. Tidak ada paket yang benar-benar dikirim.
"""

import socket


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


if __name__ == '__main__':
    print(get_lan_ip())
