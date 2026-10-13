"""Tests for the vectorized E57 reader (E57ImportWorker._fast_binary_read).

Synthetic E57 files are generated with explicit packet layouts and field
encodings; libE57Format (via pye57) is used as the reference decoder.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

pye57 = pytest.importorskip("pye57")
pytest.importorskip("PySide6")

from locul3d.plugins.importers.e57 import E57ImportWorker  # noqa: E402


# ── Minimal E57 writer ───────────────────────────────────────────────

PAGE_SIZE = 1024
DATA_PER_PAGE = 1020


def _crc32c_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return table


_CRC_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _paginate(logical: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(logical), DATA_PER_PAGE):
        page = logical[i:i + DATA_PER_PAGE].ljust(DATA_PER_PAGE, b"\0")
        out += page + struct.pack(">I", _crc32c(page))
    return bytes(out)


def _log_to_phys(l: int) -> int:
    return (l // DATA_PER_PAGE) * PAGE_SIZE + l % DATA_PER_PAGE


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    """bitPackCodec: little-endian, LSB-first bit stream."""
    big = 0
    for i, v in enumerate(values.tolist()):
        big |= int(v) << (i * bits)
    return big.to_bytes((len(values) * bits + 7) // 8, "little")


def write_e57(path, xyz, rgb, packet_sizes, xyz_bits=None, scale=0.001,
              normals=False):
    """Write a single-scan E57 file.

    xyz_bits=None stores coordinates as single precision Float, otherwise as
    ScaledInteger with a symmetric range of that bit width.  packet_sizes
    gives the number of records in each data packet.  normals=True adds
    namespaced extension fields (nor:normalX..Z) between xyz and colors.
    """
    n = len(xyz)
    assert sum(packet_sizes) == n

    if xyz_bits is None:
        xyz_proto = "".join(
            f'<{a} type="Float" precision="single"/>'
            for a in ("cartesianX", "cartesianY", "cartesianZ"))
        streams = [xyz[:, i].astype("<f4").tobytes() for i in range(3)]
        rec_bits = [32, 32, 32]
    else:
        lo, hi = -(1 << (xyz_bits - 1)), (1 << (xyz_bits - 1)) - 1
        xyz_proto = "".join(
            f'<{a} type="ScaledInteger" minimum="{lo}" maximum="{hi}" '
            f'scale="{scale!r}" offset="0"/>'
            for a in ("cartesianX", "cartesianY", "cartesianZ"))
        raw = np.clip(np.round(xyz / scale), lo, hi).astype(np.int64) - lo
        streams = [_pack_bits(raw[:, i], xyz_bits) for i in range(3)]
        rec_bits = [xyz_bits] * 3
    if normals:
        xyz_proto += "".join(
            f'<nor:{a} type="Float" precision="single"/>'
            for a in ("normalX", "normalY", "normalZ"))
        streams += [np.full(n, 0.5, dtype="<f4").tobytes()] * 3
        rec_bits += [32, 32, 32]
    rgb_proto = "".join(
        f'<{c} type="Integer" minimum="0" maximum="255"/>'
        for c in ("colorRed", "colorGreen", "colorBlue"))
    streams += [rgb[:, i].astype(np.uint8).tobytes() for i in range(3)]
    rec_bits += [8, 8, 8]

    # Data packets: each packet carries the next packet_sizes[k] records
    packets = bytearray()
    start = 0
    for count in packet_sizes:
        end = start + count
        chunks = [s[start * b // 8:end * b // 8] for s, b in zip(streams, rec_bits)]
        header_len = 6 + 2 * len(chunks)
        length = header_len + sum(len(c) for c in chunks)
        length += (4 - length % 4) % 4
        packets += struct.pack("<BBHH", 1, 0, length - 1, len(chunks))
        packets += struct.pack(f"<{len(chunks)}H", *(len(c) for c in chunks))
        for c in chunks:
            packets += c
        packets += b"\0" * (length - header_len - sum(len(c) for c in chunks))
        start = end

    section = struct.pack("<B7xQQQ", 1, 32 + len(packets), _log_to_phys(80), 0)
    xml_start = 48 + len(section) + len(packets)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<e57Root type="Structure" xmlns="http://www.astm.org/COMMIT/E57/2010-e57-v1.0" '
        'xmlns:nor="http://www.libe57.org/E57_NOR_surface_normals.txt">'
        '<formatName type="String"><![CDATA[ASTM E57 3D Imaging Data File]]></formatName>'
        '<guid type="String"><![CDATA[test-file]]></guid>'
        '<versionMajor type="Integer">1</versionMajor>'
        '<versionMinor type="Integer">0</versionMinor>'
        '<data3D type="Vector" allowHeterogeneousChildren="1"><vectorChild type="Structure">'
        '<guid type="String"><![CDATA[test-scan]]></guid>'
        f'<points type="CompressedVector" fileOffset="48" recordCount="{n}">'
        f'<prototype type="Structure">{xyz_proto}{rgb_proto}</prototype>'
        '<codecs type="Vector" allowHeterogeneousChildren="1"/></points>'
        '</vectorChild></data3D>'
        '<images2D type="Vector" allowHeterogeneousChildren="1"/></e57Root>\n'
    ).encode()

    logical = bytearray(48) + section + packets + xml
    physical_len = -(-len(logical) // DATA_PER_PAGE) * PAGE_SIZE
    logical[0:48] = struct.pack("<8sIIQQQQ", b"ASTM-E57", 1, 0, physical_len,
                                _log_to_phys(xml_start), len(xml), PAGE_SIZE)
    with open(path, "wb") as f:
        f.write(_paginate(bytes(logical)))


# ── Helpers ──────────────────────────────────────────────────────────


def _fast_read(path):
    worker = E57ImportWorker.__new__(E57ImportWorker)
    worker._log = lambda msg: None
    e57 = pye57.E57(str(path))
    header = e57.get_header(0)
    n, fields = header.point_count, list(header.point_fields)
    e57.close()
    return worker._fast_binary_read(str(path), n, True, fields)


def _reference(path):
    e57 = pye57.E57(str(path))
    data = e57.read_scan_raw(0, ignore_unsupported_fields=True)
    e57.close()
    xyz = np.column_stack([data["cartesianX"], data["cartesianY"],
                           data["cartesianZ"]]).astype(np.float32)
    rgb = np.column_stack([data["colorRed"], data["colorGreen"],
                           data["colorBlue"]]).astype(np.uint8)
    return xyz, rgb


def _points(n, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-50, 50, (n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    return xyz, rgb


def _assert_matches_reference(path, n):
    xyz, rgb = _fast_read(path)
    ref_xyz, ref_rgb = _reference(path)
    assert len(xyz) == n
    np.testing.assert_array_equal(xyz, ref_xyz)
    np.testing.assert_array_equal(rgb, ref_rgb)
    return xyz, rgb


# ── _unpack_bits ─────────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [1, 2, 7, 8, 11, 12, 16, 20, 24, 31, 32, 33, 48, 56, 64])
def test_unpack_bits(bits):
    rng = np.random.default_rng(bits)
    values = rng.integers(0, 2 ** min(bits, 63), 777, dtype=np.uint64)
    buf = np.frombuffer(_pack_bits(values, bits), dtype=np.uint8)
    np.testing.assert_array_equal(
        E57ImportWorker._unpack_bits(buf, len(values), bits), values)


# ── Packet layouts ───────────────────────────────────────────────────


def test_uniform_packets(tmp_path):
    xyz, rgb = _points(12000)
    path = tmp_path / "uniform.e57"
    write_e57(path, xyz, rgb, [3000] * 4)
    got_xyz, got_rgb = _assert_matches_reference(path, 12000)
    np.testing.assert_array_equal(got_xyz, xyz)
    np.testing.assert_array_equal(got_rgb, rgb)


def test_short_last_packet_is_not_dropped(tmp_path):
    xyz, rgb = _points(5000)
    path = tmp_path / "tail.e57"
    write_e57(path, xyz, rgb, [3000, 2000])
    got_xyz, _ = _assert_matches_reference(path, 5000)
    np.testing.assert_array_equal(got_xyz[-2000:], xyz[-2000:])


def test_variable_packet_sizes(tmp_path):
    # Incrementally written files: every flush ends with a short packet
    sizes = [1, 2, 3, 60, 3000, 3000, 1240, 7, 3000, 3000, 3000, 333]
    xyz, rgb = _points(sum(sizes))
    path = tmp_path / "variable.e57"
    write_e57(path, xyz, rgb, sizes)
    got_xyz, got_rgb = _assert_matches_reference(path, sum(sizes))
    np.testing.assert_array_equal(got_xyz, xyz)
    np.testing.assert_array_equal(got_rgb, rgb)


# ── Field encodings ──────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [16, 20, 24, 32])
def test_scaled_integer_widths(tmp_path, bits):
    n = 8 * 1000
    xyz, rgb = _points(n)
    # Keep coordinates inside the representable range at 1 mm resolution
    xyz = xyz * min(1.0, (2 ** (bits - 1) - 1) * 0.001 / 50)
    path = tmp_path / f"scaled{bits}.e57"
    # Packet record counts are multiples of 8 so bit-packed records never
    # straddle a packet boundary
    write_e57(path, xyz, rgb, [2992, 2400, n - 2992 - 2400],
              xyz_bits=bits, scale=0.001)
    got_xyz, _ = _assert_matches_reference(path, n)
    np.testing.assert_allclose(got_xyz, xyz, atol=0.0006)


def test_namespaced_extension_fields(tmp_path):
    # NavVis files carry nor:normalX..Z before the color streams
    xyz, rgb = _points(6000)
    path = tmp_path / "normals.e57"
    write_e57(path, xyz, rgb, [2000, 2000, 2000], normals=True)
    _, got_rgb = _assert_matches_reference(path, 6000)
    np.testing.assert_array_equal(got_rgb, rgb)
